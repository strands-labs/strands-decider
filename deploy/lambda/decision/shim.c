// A flat C API over libllama and mtmd for strands_decider.backends.llama_cpp, which calls it
// through ctypes. It runs the Gemma 4 decoder of a decision checkpoint under the prefix_bidir mask
// (docs/spec.md, section 6) and copies out the final-norm hidden state at each question's last
// token, which the core's readout reads. With a vision tower loaded (sd_load_vision), the state
// may hold images, whose rows take the place of their placeholders (section 11.1).
//
// prefix_bidir, with S state tokens: a state token sees every state token, in both directions,
// and no question token; a question token sees the whole prompt, in both directions; no layer
// limits attention by distance. llama.cpp gives it as two non-causal decodes on one context:
//   1. The state, all S positions in one non-causal ubatch on an empty cache. Text positions go
//      in by token id and image positions by their rows, in one mixed batch (llama_batch_ext), so
//      every state position, text or image, sees every other. mtmd's own helper decodes each text
//      run and each image separately, so earlier positions would never see later ones.
//   2. Each question, one non-causal ubatch at positions S.. on top of the cached state. Its
//      tokens see the state's cells and each other; the state's cells were computed before the
//      question existed, so they never see it. Its cells are then removed, so the next question
//      computes exactly what a fresh pass would.
//
// Two load-time metadata overrides make every layer of the stock GGUF follow the mask:
//   gemma4.attention.non_causal_type = 0 (LLAMA_NON_CAUSAL_TYPE_ALL). Gemma 4's default,
//     SWA_ONLY, keeps the full-attention layers causal under llama_set_causal_attn(ctx, false).
//     The key comes from patches/0001 (stock llama.cpp has no way to change it).
//   gemma4.attention.sliding_window = 1 << 20, so no sliding-window layer cuts a key.
//
// Only the last token of each question is an output. llama.cpp then runs Gemma 4's KV-shared
// layers (the last 20 of E2B's 35) on that one row, and sizes its buffers for one row of the
// 262,144-token LM head (n_outputs_max = 1), not one per batch token. The hidden state is read
// from the graph's "result_norm" tensor through the eval callback: asking for embeddings would make
// every token an output.
//
// An image row is what llama.cpp's Gemma 4 graph expects of an embedding row: it is not scaled by
// sqrt(n_embd), and its per-layer embedding is the pad token's, as in transformers.

#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "mtmd.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct {
    struct llama_model *           model;
    mtmd_context *                 mtmd;  // the vision tower, after sd_load_vision; NULL without one
    int                            n_embd;
    int                            n_embd_inp;
    int                            n_threads;
    struct llama_model_kv_override overrides[3];
} sd_handle;

typedef struct {
    float * dst;  // n_embd floats for the decode's one output row, or NULL to drop it
    int     n_embd;
    int     got;
} sd_capture;

// llama logs hundreds of lines per context below WARN, which CloudWatch Logs bills at ingestion. A
// CONT record continues the line before it, so it is shown at that line's level. ggml can log from
// its worker threads, so the level is atomic.
static _Atomic int sd_log_level = GGML_LOG_LEVEL_NONE;
static int         sd_verbose   = 0;

static void sd_log(enum ggml_log_level level, const char * text, void * user_data) {
    (void) user_data;
    const int shown = level == GGML_LOG_LEVEL_CONT ? sd_log_level : (sd_log_level = level);
    if (sd_verbose || shown >= GGML_LOG_LEVEL_WARN) fputs(text, stderr);
}

// Observe only "result_norm", the final-norm hidden state of the output rows.
static bool sd_eval_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    if (strcmp(t->name, "result_norm") != 0) return !ask;
    if (ask) return true;
    sd_capture * c = (sd_capture *) user_data;
    if (c->dst && t->type == GGML_TYPE_F32 && t->ne[0] == c->n_embd && t->ne[1] >= 1 &&
        ggml_is_contiguous(t)) {
        ggml_backend_tensor_get(t, c->dst, (size_t) (t->ne[1] - 1) * t->nb[1], sizeof(float) * c->n_embd);
        c->got = 1;
    }
    return true;
}

static double sd_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec / 1e6;
}

static void sd_override_int(struct llama_model_kv_override * o, const char * key, int64_t value) {
    o->tag = LLAMA_KV_OVERRIDE_TYPE_INT;
    snprintf(o->key, sizeof(o->key), "%s", key);
    o->val_i64 = value;
}

// Load a Gemma 4 GGUF, mapped. The per-layer embedding table, half of E2B's weights, is read
// lazily (LLAMA_LAZY_MODE_ON): only the rows of a batch's tokens, from the mapped file, never a
// copy of the whole table. Sets *n_embd_out to the hidden size. NULL on failure.
void * sd_load(const char * path, int n_threads, int * n_embd_out) {
    sd_verbose = getenv("STRANDS_DECIDER_LLAMA_VERBOSE") != NULL;
    llama_log_set(sd_log, NULL);
    llama_backend_init();
    sd_handle * h = (sd_handle *) calloc(1, sizeof(sd_handle));
    if (!h) return NULL;
    sd_override_int(&h->overrides[0], "gemma4.attention.non_causal_type", 0);
    sd_override_int(&h->overrides[1], "gemma4.attention.sliding_window", 1 << 20);
    h->overrides[2].key[0] = 0;

    struct llama_model_params mp = llama_model_default_params();
    mp.kv_overrides = h->overrides;
    mp.lazy_mode    = LLAMA_LAZY_MODE_ON;
    mp.load_mode    = LLAMA_LOAD_MODE_MMAP;
    h->model = llama_model_load_from_file(path, mp);
    if (!h->model) {
        free(h);
        return NULL;
    }
    h->n_embd     = llama_model_n_embd_out(h->model);
    h->n_embd_inp = llama_model_n_embd_inp(h->model);
    h->n_threads  = n_threads > 0 ? n_threads : 1;
    *n_embd_out   = h->n_embd;
    return h;
}

void sd_free(void * handle) {
    sd_handle * h = (sd_handle *) handle;
    if (!h) return;
    if (h->mtmd) mtmd_free(h->mtmd);
    llama_model_free(h->model);
    free(h);
}

// Load a vision tower (an mmproj GGUF) beside the decoder. image_max_tokens caps an image's soft
// tokens (280 for Gemma 4). 0 on success; 2 if mtmd cannot load it; 3 if it has no vision encoder
// or wants M-RoPE positions, which Gemma 4's one-dimensional positions are not.
int sd_load_vision(void * handle, const char * mmproj_path, int image_max_tokens) {
    sd_handle * h = (sd_handle *) handle;
    if (!h || h->mtmd) return 1;
    mtmd_log_set(sd_log, NULL);
    struct mtmd_context_params p = mtmd_context_params_default();
    p.use_gpu       = false;
    p.print_timings = false;
    p.n_threads     = h->n_threads;
    p.warmup        = false;  // the caller warms up with a request
    if (image_max_tokens > 0) p.image_max_tokens = image_max_tokens;
    mtmd_context * m = mtmd_init_from_file(mmproj_path, h->model, p);
    if (!m) return 2;
    if (!mtmd_support_vision(m) || mtmd_decode_use_mrope(m)) {
        mtmd_free(m);
        return 3;
    }
    h->mtmd = m;
    return 0;
}

// The width of an image row: the decoder's input embedding size.
int sd_n_embd_inp(void * handle) { return ((sd_handle *) handle)->n_embd_inp; }

// Run one image, w x hgt RGB bytes already at the size the image processor gives it, through the
// vision tower and its projection. *n_tok gets its soft-token count, and its rows are copied to out
// if out holds that many (cap_rows). wrap[0] and wrap[1] get the token mtmd puts before and after
// the image, or -1. ms[0] gets mtmd's preprocessing time and ms[1] the encode time. 0 on success.
int sd_encode_image(void * handle, const unsigned char * rgb, int w, int hgt, float * out, int cap_rows,
                    int * n_tok, int * wrap, double * ms) {
    sd_handle * h = (sd_handle *) handle;
    if (!h || !h->mtmd || !rgb || w <= 0 || hgt <= 0) return 1;
    double        t0     = sd_ms();
    mtmd_bitmap * bitmap = mtmd_bitmap_init((uint32_t) w, (uint32_t) hgt, rgb);
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    int rc = bitmap && chunks ? 0 : 2;
    if (rc == 0) {
        const mtmd_input_part   part     = { NULL, bitmap };
        const mtmd_input_part * parts[1] = { &part };
        if (mtmd_tokenize_from_parts(h->mtmd, chunks, parts, 1, false) != 0) rc = 3;
    }
    if (bitmap) mtmd_bitmap_free(bitmap);
    const mtmd_input_chunk * image = NULL;
    wrap[0] = wrap[1] = -1;
    for (size_t i = 0; rc == 0 && i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) {
            image = c;
        } else {
            size_t              n = 0;
            const llama_token * t = mtmd_input_chunk_get_tokens_text(c, &n);
            if (n > 0) wrap[image ? 1 : 0] = t[image ? 0 : n - 1];
        }
    }
    if (rc == 0 && !image) rc = 4;
    double t1 = sd_ms();
    if (rc == 0) {
        *n_tok = (int) mtmd_input_chunk_get_n_tokens(image);
        if (cap_rows < *n_tok) {
            rc = 6;
        } else if (mtmd_encode_chunk(h->mtmd, image) != 0) {
            rc = 5;
        } else {
            memcpy(out, mtmd_get_output_embd(h->mtmd), sizeof(float) * (size_t) *n_tok * h->n_embd_inp);
        }
    }
    if (ms) {
        ms[0] = t1 - t0;
        ms[1] = sd_ms() - t1;
    }
    if (chunks) mtmd_input_chunks_free(chunks);
    return rc;
}

static void sd_fill(struct llama_batch * b, const int * ids, int n, int pos0) {
    b->n_tokens = n;
    for (int i = 0; i < n; i++) {
        b->token[i]     = ids[i];
        b->pos[i]       = pos0 + i;
        b->n_seq_id[i]  = 1;
        b->seq_id[i][0] = 0;
        b->logits[i]    = i == n - 1;  // one output row: the last token
    }
}

// The state as one mixed batch: positions 0..n_state-1, each position whose id is image_token
// replaced, in order, by the next row of embd; rows[i] is image i's row count, which must equal the
// length of its run of placeholders. 0 on success.
static int sd_decode_mixed_state(sd_handle * h, struct llama_context * ctx, const int * state_ids, int n_state,
                                 int image_token, const float * embd, const int * rows, int n_images) {
    struct llama_batch_ext * be = llama_batch_ext_init(ctx);
    if (!be) return 12;
    const size_t ne  = (size_t) h->n_embd_inp;
    int          rc  = 0, img = 0, row = 0;
    size_t       off = 0;
    for (int i = 0; i < n_state && rc == 0; i++) {
        int idx;
        if (state_ids[i] == image_token) {
            if (img >= n_images) {
                rc = 10;
                break;
            }
            struct llama_embd e = { embd + (off + row) * ne, 1, ne };
            idx = llama_batch_ext_add_embd(be, 0, e);
            if (++row == rows[img]) {
                off += rows[img];
                row = 0;
                img++;
            }
        } else {
            if (row != 0) {  // a run of placeholders shorter than its image
                rc = 11;
                break;
            }
            idx = llama_batch_ext_add_token(be, 0, state_ids[i]);
        }
        if (idx < 0) {
            rc = 12;
            break;
        }
        const llama_pos p = i;
        llama_batch_ext_set_pos(be, idx, &p);
        if (i == n_state - 1) llama_batch_ext_set_output_logits(be, idx, true);
    }
    if (rc == 0 && (img != n_images || row != 0)) rc = 13;  // more images than runs, or a longer run
    if (rc == 0 && llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, be) != 0) rc = 3;
    llama_synchronize(ctx);
    llama_batch_ext_free(be);
    return rc;
}

// Decode the state (n_state ids) once, then each question after it on the same cache, and write
// each question's last-token final hidden state to out ([n_q, n_embd], in question order).
// n_images > 0 needs a vision tower: the state's runs of image_token are then filled with embd, the
// images' rows end to end, rows[i] of them for image i (see sd_decode_mixed_state). ms, if not
// NULL, gets n_q + 2 timings in milliseconds: the context's creation, the state's decode, then each
// question's decode. 0 on success.
int sd_forward(void * handle, const int * state_ids, int n_state, int image_token, const float * embd,
               const int * rows, int n_images, const int * q_ids, const int * q_lens, int n_q, float * out,
               double * ms) {
    sd_handle * h = (sd_handle *) handle;
    if (!h || n_state <= 0 || n_q <= 0 || n_images < 0 || (n_images > 0 && (!h->mtmd || !embd || !rows))) {
        return 1;
    }
    int longest = 0;
    for (int i = 0; i < n_q; i++) {
        if (q_lens[i] <= 0) return 1;
        if (q_lens[i] > longest) longest = q_lens[i];
    }
    // A non-causal decode must fit one ubatch, so the ubatch holds the whole state.
    const int  n_batch = n_state > longest ? n_state : longest;
    sd_capture cap     = { NULL, h->n_embd, 0 };

    double                      t0 = sd_ms();
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = n_state + longest;
    cp.n_batch           = n_batch;
    cp.n_ubatch          = n_batch;
    cp.n_seq_max         = 1;  // lets Gemma 4 run its KV-shared layers on the output row only
    cp.n_outputs_max     = 1;  // buffers for one output row, not n_batch rows of LM-head logits
    cp.n_threads         = h->n_threads;
    cp.n_threads_batch   = h->n_threads;
    cp.embeddings        = false;
    cp.no_perf           = true;
    cp.swa_full          = true;
    cp.attention_type    = LLAMA_ATTENTION_TYPE_NON_CAUSAL;
    cp.flash_attn_type   = LLAMA_FLASH_ATTN_TYPE_AUTO;
    cp.cb_eval           = sd_eval_cb;
    cp.cb_eval_user_data = &cap;
    struct llama_context * ctx = llama_init_from_model(h->model, cp);
    if (!ctx) return 2;
    llama_set_causal_attn(ctx, false);
    struct llama_batch b   = llama_batch_init(n_batch, 0, 1);
    llama_memory_t     mem = llama_get_memory(ctx);
    double             t1  = sd_ms();
    if (ms) ms[0] = t1 - t0;

    int rc;
    if (n_images > 0) {
        rc = sd_decode_mixed_state(h, ctx, state_ids, n_state, image_token, embd, rows, n_images);
    } else {
        sd_fill(&b, state_ids, n_state, 0);
        rc = llama_decode(ctx, b) == 0 ? 0 : 3;
        llama_synchronize(ctx);  // llama_decode can return before the compute ends; time each phase whole
    }
    double t2 = sd_ms();
    if (ms) ms[1] = t2 - t1;

    int off = 0;
    for (int i = 0; i < n_q && rc == 0; i++) {
        double tq = sd_ms();
        sd_fill(&b, q_ids + off, q_lens[i], n_state);
        off     += q_lens[i];
        cap.dst  = out + (size_t) i * h->n_embd;
        cap.got  = 0;
        if (llama_decode(ctx, b) != 0) rc = 4;
        llama_synchronize(ctx);
        if (rc == 0 && !cap.got) rc = 5;
        cap.dst = NULL;
        if (!llama_memory_seq_rm(mem, 0, n_state, -1)) rc = rc ? rc : 6;
        if (ms) ms[2 + i] = sd_ms() - tq;
    }
    llama_batch_free(b);
    llama_free(ctx);
    return rc;
}
