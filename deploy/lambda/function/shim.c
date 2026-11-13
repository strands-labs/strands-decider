// A flat C API over libllama for backend.py, which calls it through ctypes: no structs by
// value, no llama.h layout to repeat in Python. dh_load loads a GGUF; dh_forward decodes a
// state once and each question after it, and copies out the questions' final (post-norm)
// hidden states, which the pointer head reads. dh_load_mm loads a vision tower beside the
// model, and dh_forward_mm does what dh_forward does for a state that holds images. The
// Dockerfile builds it against llama.cpp.

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper.h"
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
    mtmd_context *       mtmd;  // the vision tower, after dh_load_mm; NULL without one
} dh_handle;

// llama logs hundreds of lines per context below WARN, about 150 KB a request, which
// CloudWatch Logs bills at ingestion. A CONT record continues the line before it, so it is
// shown at that line's level. ggml can log from its worker threads, so the level is atomic.
// An image moves the positions of the text after it by its grid size, not its token count,
// and the linear-attention memory warns of every such jump: expected, so not shown.
static _Atomic int dh_log_level = GGML_LOG_LEVEL_NONE;
static void dh_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    const int shown = level == GGML_LOG_LEVEL_CONT ? dh_log_level : (dh_log_level = level);
    if (shown >= GGML_LOG_LEVEL_WARN && !strstr(text, "non-consecutive token position")) {
        fputs(text, stderr);
    }
}

// Load a GGUF. Returns NULL on failure; sets *n_embd_out to the hidden size.
void * dh_load(const char * model_path, int n_threads, int * n_embd_out) {
    llama_log_set(dh_log, NULL);
    llama_backend_init();
    struct llama_model * model = llama_model_load_from_file(model_path, llama_model_default_params());
    if (!model) return NULL;
    dh_handle * h = (dh_handle *) calloc(1, sizeof(dh_handle));
    if (!h) {
        llama_model_free(model);
        return NULL;
    }
    h->model     = model;
    h->n_embd    = llama_model_n_embd_out(model);
    h->n_threads = n_threads > 0 ? n_threads : 1;
    *n_embd_out  = h->n_embd;
    return h;
}

// Load a vision tower (an mmproj GGUF) for the model. 0 on success.
int dh_load_mm(void * handle, const char * mmproj_path) {
    if (!handle) return 1;
    dh_handle * h = (dh_handle *) handle;
    mtmd_helper_log_set(dh_log, NULL);
    struct mtmd_context_params p = mtmd_context_params_default();
    p.use_gpu       = false;
    p.print_timings = false;
    p.n_threads     = h->n_threads;
    h->mtmd = mtmd_init_from_file(mmproj_path, h->model, p);
    if (!h->mtmd) return 2;
    // This shim decodes text after an image at the position mtmd returns, which is right
    // only for M-RoPE models, as Qwen3.5's vision tower is.
    return mtmd_support_vision(h->mtmd) && mtmd_decode_use_mrope(h->mtmd) ? 0 : 3;
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

// A context for a state of n_state tokens and n_q questions after it, decoded n_batch tokens
// at a time. Embeddings start off: the state is decoded with no outputs.
static struct llama_context * dh_context(const dh_handle * h, int n_state, const int * q_lens,
                                         int n_q, int n_batch) {
    int longest_group = 0;
    for (int g = 0; g < n_q; g += DH_GROUP) {
        int t = 0;
        for (int s = g; s < n_q && s < g + DH_GROUP; s++) t += q_lens[s];
        if (t > longest_group) longest_group = t;
    }
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
    return llama_init_from_model(h->model, cparams);
}

// Decode n text tokens on sequence 0 from position *pos, with no outputs, and move *pos past
// them. 0 on success.
static int dh_state_text(struct llama_context * ctx, struct llama_batch * b, const int * ids,
                         int n, llama_pos * pos) {
    if (n == 0) return 0;
    b->n_tokens = n;
    for (int i = 0; i < n; i++) {
        b->token[i] = ids[i]; b->pos[i] = *pos + i;
        b->n_seq_id[i] = 1; b->seq_id[i][0] = 0; b->logits[i] = 0;
    }
    const int rc = llama_decode(ctx, *b) == 0 ? 0 : 3;
    b->n_tokens = 0;
    *pos += n;
    return rc;
}

// After the state, on sequence 0, up to position `base`: per group of DH_GROUP questions,
// copy sequence 0's memory (the attention cache and the linear-attention state) to one
// sequence per question, decode the questions on their own sequences from `base`, and drop
// those sequences. Writes every question token's hidden state, in question order, to out
// ([sum(q_lens), n_embd]). 0 on success.
//
// llama keeps a row of vocabulary logits, about 1 MB here, for every token it outputs, and
// with embeddings on every token is an output. So the state decodes with embeddings off,
// and the questions at most DH_UBATCH tokens per llama_decode.
static int dh_questions(struct llama_context * ctx, struct llama_batch * b, int n_embd,
                        llama_pos base, const int * q_ids, const int * q_lens, int n_q,
                        float * out) {
    llama_set_embeddings(ctx, true);
    llama_memory_t mem = llama_get_memory(ctx);
    int off = 0, row = 0, rc = 0;
    for (int g = 0; g < n_q && rc == 0; g += DH_GROUP) {
        const int g_end = g + DH_GROUP < n_q ? g + DH_GROUP : n_q;
        for (int s = g; s < g_end; s++) llama_memory_seq_cp(mem, 0, s - g + 1, -1, -1);
        for (int s = g; s < g_end && rc == 0; s++) {
            for (int j = 0; j < q_lens[s]; j++) {
                if (b->n_tokens == DH_UBATCH && (rc = dh_flush(ctx, b, n_embd, out, &row))) break;
                const int k = b->n_tokens++;
                b->token[k] = q_ids[off + j]; b->pos[k] = base + j;
                b->n_seq_id[k] = 1; b->seq_id[k][0] = s - g + 1; b->logits[k] = 1;
            }
            off += q_lens[s];
        }
        if (rc == 0) rc = dh_flush(ctx, b, n_embd, out, &row);
        for (int s = g; s < g_end; s++) llama_memory_seq_rm(mem, s - g + 1, -1, -1);
    }
    return rc;
}

// Decode the state on sequence 0, once, then the questions after it (dh_questions). 0 on
// success.
int dh_forward(void * handle, const int * state_ids, int n_state, const int * q_ids,
               const int * q_lens, int n_q, float * out) {
    if (!handle || n_state <= 0 || n_q <= 0) return 1;
    dh_handle * h = (dh_handle *) handle;
    const int n_batch = n_state > DH_UBATCH ? n_state : DH_UBATCH;
    struct llama_context * ctx = dh_context(h, n_state, q_lens, n_q, n_batch);
    if (!ctx) return 2;
    struct llama_batch b = llama_batch_init(n_batch, 0, 1);
    llama_pos pos = 0;
    int rc = dh_state_text(ctx, &b, state_ids, n_state, &pos);
    if (rc == 0) rc = dh_questions(ctx, &b, h->n_embd, pos, q_ids, q_lens, n_q, out);
    llama_batch_free(b);
    llama_free(ctx);
    return rc;
}

// dh_forward for a state that holds images: n_images + 1 text segments with image i between
// segment i and segment i + 1. The segments are token ids, seg_lens[i] each, end to end in
// seg_ids; image i is rgb[i], w[i] * h[i] * 3 bytes. mtmd wraps each image in
// <|vision_start|> and <|vision_end|>, runs it through the vision tower and decodes its
// embeddings at M-RoPE positions, which move by the image's grid size; the text after an
// image continues from there. Sets img_tokens[i] to image i's token count, its start and end
// included. 0 on success.
int dh_forward_mm(void * handle, const int * seg_ids, const int * seg_lens,
                  const unsigned char * const * rgb, const int * w, const int * hgt, int n_images,
                  const int * q_ids, const int * q_lens, int n_q, float * out, int * img_tokens) {
    if (!handle || n_images <= 0 || n_q <= 0) return 1;
    dh_handle * h = (dh_handle *) handle;
    if (!h->mtmd) return 6;
    mtmd_input_chunks ** chunks = (mtmd_input_chunks **) calloc(n_images, sizeof(*chunks));
    if (!chunks) return 9;
    int rc = 0, n_state = 0, longest = DH_UBATCH;
    for (int i = 0; i < n_images && rc == 0; i++) {
        mtmd_bitmap * bitmap = mtmd_bitmap_init((uint32_t) w[i], (uint32_t) hgt[i], rgb[i]);
        chunks[i] = mtmd_input_chunks_init();
        if (!bitmap || !chunks[i]) {
            rc = 9;
        } else {
            const mtmd_input_part part = { NULL, bitmap };
            const mtmd_input_part * parts[1] = { &part };
            if (mtmd_tokenize_from_parts(h->mtmd, chunks[i], parts, 1, false) != 0) rc = 7;
        }
        if (bitmap) mtmd_bitmap_free(bitmap);
        if (rc == 0) {
            img_tokens[i] = (int) mtmd_helper_get_n_tokens(chunks[i]);
            n_state += img_tokens[i];
            if (img_tokens[i] > longest) longest = img_tokens[i];
        }
    }
    for (int i = 0; i <= n_images; i++) {
        n_state += seg_lens[i];
        if (seg_lens[i] > longest) longest = seg_lens[i];
    }
    struct llama_context * ctx = NULL;
    if (rc == 0 && !(ctx = dh_context(h, n_state, q_lens, n_q, longest))) rc = 2;
    struct llama_batch b = llama_batch_init(longest, 0, 1);
    llama_pos pos = 0;
    for (int i = 0, off = 0; i <= n_images && rc == 0; off += seg_lens[i], i++) {
        rc = dh_state_text(ctx, &b, seg_ids + off, seg_lens[i], &pos);
        if (rc == 0 && i < n_images &&
            mtmd_helper_eval_chunks(h->mtmd, ctx, chunks[i], pos, 0, longest, false, &pos) != 0) {
            rc = 8;
        }
    }
    if (rc == 0) rc = dh_questions(ctx, &b, h->n_embd, pos, q_ids, q_lens, n_q, out);
    llama_batch_free(b);
    if (ctx) llama_free(ctx);
    for (int i = 0; i < n_images; i++) {
        if (chunks[i]) mtmd_input_chunks_free(chunks[i]);
    }
    free(chunks);
    return rc;
}
