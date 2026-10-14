// decider_hidden.c
//
// Flat C shim over libllama for the Lambda app. A ctypes-friendly API (no
// struct-by-value, no header layout to reproduce in Python): load a GGUF once,
// then decode exact token ids with embeddings=true and pooling NONE, and copy
// the per-token post-final-norm hidden states into a caller buffer. Those are
// the torso's last_hidden_state, which the pointer head reads.
//
// Built by container_build.sh against the pinned llama.cpp:
//   gcc -O3 -fPIC -shared -o libdecider_hidden.so decider_hidden.c \
//       -I<src>/include -I<src>/ggml/include -L<libs> -lllama

#include "llama.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef DH_UBATCH
#define DH_UBATCH 512
#endif
// Questions decoded together, one seq each, against the shared state.
#ifndef DH_GROUP
#define DH_GROUP 32
#endif

// Opaque handle so Python holds a single void* and never sees a struct.
typedef struct {
    struct llama_model * model;
    int                  n_embd;
    int                  n_threads;
} dh_handle;

// llama logs hundreds of lines per context at INFO and DEBUG, about 150 KB a request,
// which CloudWatch bills at ingestion. Pass on warnings and errors only.
static enum ggml_log_level dh_log_level = GGML_LOG_LEVEL_NONE;
static void dh_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    if (level == GGML_LOG_LEVEL_CONT) level = dh_log_level;  // continues the last line
    else dh_log_level = level;
    if (level >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

// Load a GGUF from a filesystem path (a /proc/self/fd/<n> path backed by a memfd
// works -- llama mmaps it). Returns NULL on failure. n_embd_out is set to the
// model's embedding-output dimension (2048 for this torso).
void * dh_load(const char * model_path, int n_threads, int * n_embd_out) {
    llama_log_set(dh_log, NULL);
    llama_backend_init();
    struct llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;
    struct llama_model * model = llama_model_load_from_file(model_path, mparams);
    if (!model) return NULL;
    dh_handle * h = (dh_handle *) calloc(1, sizeof(dh_handle));
    h->model     = model;
    h->n_embd    = llama_model_n_embd_out(model);
    h->n_threads = n_threads > 0 ? n_threads : 1;
    if (n_embd_out) *n_embd_out = h->n_embd;
    return h;
}

int dh_n_embd(void * handle) {
    return handle ? ((dh_handle *) handle)->n_embd : -1;
}

// Decode b and append its per-token hidden states to out at *row. Every token in
// b is an output (embeddings=true makes llama output all of them anyway).
static int dh_flush(struct llama_context * ctx, struct llama_batch * b, int n_embd,
                    float * out, int * row) {
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

// Run one forward pass over `n_tokens` ids and write [n_tokens, n_embd] float32
// row-major into `out` (caller-allocated). A fresh context is created and freed
// per call so each request is independent (no KV bleed between requests).
// The model weights stay resident in the handle across calls.
// Returns 0 on success, non-zero on failure.
int dh_forward(void * handle, const int * ids, int n_tokens, float * out) {
    if (!handle || n_tokens <= 0) return 1;
    dh_handle * h = (dh_handle *) handle;

    // Decoded DH_UBATCH tokens per llama_decode: every token is an output, and llama
    // reserves n_vocab floats of logits per output (see dh_forward_shared).
    const int n_chunk = n_tokens < DH_UBATCH ? n_tokens : DH_UBATCH;
    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = n_tokens + 8;
    cparams.n_batch         = n_chunk;
    cparams.n_ubatch        = n_chunk;
    cparams.n_threads       = h->n_threads;
    cparams.n_threads_batch = h->n_threads;
    cparams.embeddings      = true;
    cparams.pooling_type    = LLAMA_POOLING_TYPE_NONE;
    cparams.no_perf         = true;

    struct llama_context * ctx = llama_init_from_model(h->model, cparams);
    if (!ctx) return 2;

    struct llama_batch batch = llama_batch_init(n_chunk, 0, 1);
    const int n_embd = h->n_embd;
    int row = 0, rc = 0;
    for (int i = 0; i < n_tokens && rc == 0; i++) {
        int k = batch.n_tokens++;
        batch.token[k]     = (llama_token) ids[i];
        batch.pos[k]       = i;
        batch.n_seq_id[k]  = 1;
        batch.seq_id[k][0] = 0;
        batch.logits[k]    = 1;
        if (batch.n_tokens == n_chunk || i == n_tokens - 1) rc = dh_flush(ctx, &batch, n_embd, out, &row);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    return rc;
}

// Shared-prefix forward: decode the state prefix ONCE on seq 0, copy its memory
// (attention KV + linear-attention recurrent state) to one seq per question, then
// decode every question suffix on its own seq. Writes the hidden states of the
// SUFFIX tokens only, concatenated in question order, into `out`
// ([sum(q_lens), n_embd]). The same idea as SystemOneEngine's prefix cache.
// `sequential` != 0 decodes each question in its own llama_decode calls.
//
// Two things bound the memory. llama reserves n_vocab floats of logits for every
// output token (about 1 MB each for this vocabulary), and with embeddings=true
// every token is an output. So the state decodes with embeddings off and no
// outputs, and the suffixes decode at most DH_UBATCH tokens per llama_decode.
// And questions go through in groups of DH_GROUP seqs, whose seqs are dropped and
// re-forked from seq 0 for the next group: llama allows at most 256 seqs, and the
// KV cells only have to hold the state plus one group's suffixes.
int dh_forward_shared(void * handle, const int * state_ids, int n_state,
                      const int * q_ids, const int * q_lens, int n_q,
                      int sequential, float * out) {
    if (!handle || n_state <= 0 || n_q <= 0) return 1;
    dh_handle * h = (dh_handle *) handle;
    int max_group = 0;
    for (int g = 0; g < n_q; g += DH_GROUP) {
        int t = 0;
        for (int s = g; s < n_q && s < g + DH_GROUP; s++) t += q_lens[s];
        if (t > max_group) max_group = t;
    }
    const int n_group = n_q < DH_GROUP ? n_q : DH_GROUP;
    const int n_big = n_state > DH_UBATCH ? n_state : DH_UBATCH;

    struct llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx           = n_state + max_group + 16;
    cparams.n_batch         = n_big + 8;
    // Cap the micro-batch: compute buffers scale ~quadratically with n_ubatch
    // (4096-token pages OOMed at 10 GB). llama splits n_batch into ubatches and
    // carries KV + recurrent state across them.
    cparams.n_ubatch        = DH_UBATCH;
    cparams.n_seq_max       = n_group + 1;
    cparams.kv_unified      = true;   // prefix cells are shared, not copied
    cparams.n_threads       = h->n_threads;
    cparams.n_threads_batch = h->n_threads;
    cparams.embeddings      = false;  // switched on for the suffixes
    cparams.pooling_type    = LLAMA_POOLING_TYPE_NONE;
    cparams.no_perf         = true;

    struct llama_context * ctx = llama_init_from_model(h->model, cparams);
    if (!ctx) return 2;

    // 1) state prefix on seq 0, no outputs.
    struct llama_batch b = llama_batch_init(n_big, 0, 1);
    b.n_tokens = n_state;
    for (int i = 0; i < n_state; i++) {
        b.token[i] = (llama_token) state_ids[i]; b.pos[i] = i;
        b.n_seq_id[i] = 1; b.seq_id[i][0] = 0; b.logits[i] = 0;
    }
    if (llama_decode(ctx, b) != 0) { llama_batch_free(b); llama_free(ctx); return 3; }
    b.n_tokens = 0;
    llama_set_embeddings(ctx, true);

    // 2) per group: fork the prefix memory into one seq per question, decode the
    //    suffixes in chunks of at most DH_UBATCH tokens, then drop the group's seqs.
    llama_memory_t mem = llama_get_memory(ctx);
    const int n_embd = h->n_embd;
    int off = 0, row = 0, rc = 0;
    for (int g = 0; g < n_q && rc == 0; g += DH_GROUP) {
        const int g_end = g + DH_GROUP < n_q ? g + DH_GROUP : n_q;
        for (int s = g; s < g_end; s++) llama_memory_seq_cp(mem, 0, s - g + 1, -1, -1);
        for (int s = g; s < g_end && rc == 0; s++) {
            if (sequential) rc = dh_flush(ctx, &b, n_embd, out, &row);
            for (int j = 0; j < q_lens[s] && rc == 0; j++) {
                if (b.n_tokens == DH_UBATCH) rc = dh_flush(ctx, &b, n_embd, out, &row);
                int k = b.n_tokens++;
                b.token[k] = (llama_token) q_ids[off + j]; b.pos[k] = n_state + j;
                b.n_seq_id[k] = 1; b.seq_id[k][0] = s - g + 1; b.logits[k] = 1;
            }
            off += q_lens[s];
        }
        if (rc == 0) rc = dh_flush(ctx, &b, n_embd, out, &row);
        for (int s = g; s < g_end; s++) llama_memory_seq_rm(mem, s - g + 1, -1, -1);
    }
    llama_batch_free(b);
    llama_free(ctx);
    return rc;
}
