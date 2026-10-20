// A flat C API over libllama for backend.py, which calls it through ctypes: no structs by
// value, no llama.h layout to repeat in Python. dh_load loads a GGUF; dh_forward decodes a
// state once and each question after it, and copies out the questions' final (post-norm)
// hidden states, which the pointer head reads. The Dockerfile builds it against llama.cpp.

#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// The most tokens per llama_decode. llama's compute buffers grow with it: uncapped, a
// 4,096-token state ran out of 10 GB.
#define DH_UBATCH 512
// Questions decoded together, one sequence each. llama allows at most 256 sequences, and
// the cache need only hold the state and one group's questions.
#define DH_GROUP 32

typedef struct {
    struct llama_model * model;
    int                  n_embd;
    int                  n_threads;
} dh_handle;

// llama logs hundreds of lines per context below WARN, about 150 KB a request, which
// CloudWatch Logs bills at ingestion. A CONT record continues the line before it.
static enum ggml_log_level dh_log_level = GGML_LOG_LEVEL_NONE;
static void dh_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level != GGML_LOG_LEVEL_CONT) dh_log_level = level;
    if (dh_log_level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

// Load a GGUF. Returns NULL on failure; sets *n_embd_out to the hidden size.
void * dh_load(const char * model_path, int n_threads, int * n_embd_out) {
    llama_log_set(dh_log, NULL);
    llama_backend_init();
    struct llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) return NULL;
    dh_handle * h = (dh_handle *) calloc(1, sizeof(dh_handle));
    h->model     = model;
    h->n_embd    = llama_model_n_embd_out(model);
    h->n_threads = n_threads > 0 ? n_threads : 1;
    *n_embd_out  = h->n_embd;
    return h;
}

// Decode b's tokens and copy each one's hidden state to out at row *row onward.
static int dh_flush(struct llama_context * ctx, struct llama_batch * b, int n_embd, float * out,
                    int * row) {
    if (b->n_tokens == 0) return 0;
    if (llama_decode(ctx, *b) != 0) return 4;
    for (int i = 0; i < b->n_tokens; i++, (*row)++) {
        const float * e = llama_get_embeddings_ith(ctx, i);
        if (!e) return 5;
        memcpy(out + (size_t) *row * n_embd, e, sizeof(float) * n_embd);
    }
    b->n_tokens = 0;
    return 0;
}

// Decode the state on sequence 0, once. Then, per group of DH_GROUP questions, copy
// sequence 0's memory (the attention cache and the linear-attention state) to one sequence
// per question, decode the questions on their own sequences, and drop those sequences.
// Writes every question token's hidden state, in question order, to out
// ([sum(q_lens), n_embd]). 0 on success.
//
// llama keeps a row of vocabulary logits, about 1 MB here, for every token it outputs, and
// with embeddings on every token is an output. So the state decodes with embeddings off,
// and the questions at most DH_UBATCH tokens per llama_decode.
int dh_forward(void * handle, const int * state_ids, int n_state, const int * q_ids,
               const int * q_lens, int n_q, float * out) {
    if (!handle || n_state <= 0 || n_q <= 0) return 1;
    dh_handle * h = (dh_handle *) handle;
    int longest_group = 0;
    for (int g = 0; g < n_q; g += DH_GROUP) {
        int t = 0;
        for (int s = g; s < n_q && s < g + DH_GROUP; s++) t += q_lens[s];
        if (t > longest_group) longest_group = t;
    }
    const int n_batch = n_state > DH_UBATCH ? n_state : DH_UBATCH;
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = n_state + longest_group;
    cparams.n_batch         = n_batch;
    cparams.n_ubatch        = DH_UBATCH;
    cparams.n_seq_max       = (n_q < DH_GROUP ? n_q : DH_GROUP) + 1;
    cparams.kv_unified      = true;  // the questions share the state's cells
    cparams.n_threads       = h->n_threads;
    cparams.n_threads_batch = h->n_threads;
    // Pooling stays unset: for a model without a pooling type, a GGUF converted from a causal LM,
    // that is none, one hidden state per token. Setting it explicitly logs a warning per context.
    cparams.embeddings      = false;
    cparams.no_perf         = true;
    struct llama_context * ctx = llama_init_from_model(h->model, cparams);
    if (!ctx) return 2;
    struct llama_batch b = llama_batch_init(n_batch, 0, 1);

    b.n_tokens = n_state;
    for (int i = 0; i < n_state; i++) {
        b.token[i] = state_ids[i]; b.pos[i] = i;
        b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 0;
    }
    int rc = llama_decode(ctx, b) == 0 ? 0 : 3;
    b.n_tokens = 0;

    llama_set_embeddings(ctx, true);
    llama_memory_t mem = llama_get_memory(ctx);
    int off = 0, row = 0;
    for (int g = 0; g < n_q && rc == 0; g += DH_GROUP) {
        const int g_end = g + DH_GROUP < n_q ? g + DH_GROUP : n_q;
        for (int s = g; s < g_end; s++) llama_memory_seq_cp(mem, 0, s - g + 1, -1, -1);
        for (int s = g; s < g_end && rc == 0; s++) {
            for (int j = 0; j < q_lens[s]; j++) {
                if (b.n_tokens == DH_UBATCH && (rc = dh_flush(ctx, &b, h->n_embd, out, &row))) break;
                const int k = b.n_tokens++;
                b.token[k] = q_ids[off + j]; b.pos[k] = n_state + j;
                b.n_seq_id[k] = 1; b.seq_id[k][0] = s - g + 1; b.logits[k] = 1;
            }
            off += q_lens[s];
        }
        if (rc == 0) rc = dh_flush(ctx, &b, h->n_embd, out, &row);
        for (int s = g; s < g_end; s++) llama_memory_seq_rm(mem, s - g + 1, -1, -1);
    }
    llama_batch_free(b);
    llama_free(ctx);
    return rc;
}
