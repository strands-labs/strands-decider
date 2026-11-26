// A flat C API over libllama and mtmd for strands_decider.backends.llama_cpp, which calls it
// through ctypes. It runs the Gemma 4 decoder of a model checkpoint under the prefix_bidir mask
// and copies out the final-norm hidden state at each question's last token, which the core's
// readout reads. With a vision tower loaded (sd_load_vision), the state may hold images, whose
// rows take the place of their placeholders. A call that can fail returns NULL on success, or what
// failed.
//
// prefix_bidir, with S state tokens: a state token sees every state token, in both directions,
// and no question token; a question token sees the whole prompt, in both directions; no layer
// limits attention by distance. llama.cpp gives it as two non-causal decodes on one context:
//   1. The state, all S positions in one non-causal ubatch on an empty cache. With images, text
//      positions go in by token id and image positions by their rows, in one mixed batch
//      (llama_batch_ext), so every state position sees every other. mtmd's own helper decodes
//      each text run and each image separately, so earlier positions would never see later ones.
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

#define _GNU_SOURCE  // dladdr
#include "ggml-backend.h"
#include "ggml.h"
#include "llama.h"
#include "mtmd.h"

#include <dlfcn.h>
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

// The CPU backend variant in use: the libggml-cpu-<tag>.so ggml picked for this host, "static" when
// the CPU backend is linked in, or "none" before the first sd_load.
static char sd_cpu_variant[64] = "none";

// A --lambda-variants build (scripts/build_native.sh) links no CPU backend: ggml loads the best
// libggml-cpu-*.so its host supports, from this library's own directory, since ggml's default
// search (the executable's directory and the working directory) is Python's, not ours.
static void sd_load_backends(void) {
    if (ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU)) {
        if (strcmp(sd_cpu_variant, "none") == 0) snprintf(sd_cpu_variant, sizeof(sd_cpu_variant), "static");
        return;
    }
    Dl_info info;
    char    dir[4096] = ".";
    if (dladdr((void *) sd_load_backends, &info) && info.dli_fname) {
        snprintf(dir, sizeof(dir), "%s", info.dli_fname);
        char * slash = strrchr(dir, '/');
        if (slash) *slash = 0; else snprintf(dir, sizeof(dir), ".");
    }
    ggml_backend_load_all_from_path(dir);
    ggml_backend_dev_t cpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    if (!cpu) {
        fprintf(stderr, "strands-decider: no CPU backend in %s runs on this host\n", dir);
        return;
    }
    // The module that defines the CPU registration function is the variant ggml chose.
    ggml_backend_reg_t reg = ggml_backend_dev_backend_reg(cpu);
    void * fn = ggml_backend_reg_get_proc_address(reg, "ggml_backend_set_n_threads");
    const char * name = "loaded";
    if (fn && dladdr(fn, &info) && info.dli_fname) {
        const char * base = strrchr(info.dli_fname, '/');
        name = base ? base + 1 : info.dli_fname;
    }
    snprintf(sd_cpu_variant, sizeof(sd_cpu_variant), "%s", name);
}

const char * sd_cpu_backend(void) { return sd_cpu_variant; }

// Load a Gemma 4 GGUF, mapped, and give its hidden size and its input embedding size (an image
// row's). The per-layer embedding table, half of E2B's weights, is read lazily
// (LLAMA_LAZY_MODE_ON): only the rows of a batch's tokens, from the mapped file, never a copy of
// the whole table. NULL on failure, and for a GGUF whose general.architecture is not gemma4: the
// overrides name gemma4's keys, and another architecture would ignore them and run causally.
void * sd_load(const char * path, int n_threads, int * n_embd, int * n_embd_inp) {
    sd_verbose = getenv("STRANDS_DECIDER_LLAMA_VERBOSE") != NULL;
    llama_log_set(sd_log, NULL);
    sd_load_backends();
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
    // STRANDS_DECIDER_NO_REPACK=1 keeps ggml from repacking weights for its GEMM kernels (llama.cpp's --no-repack), for measuring.
    if (getenv("STRANDS_DECIDER_NO_REPACK")) mp.use_extra_bufts = false;
    h->model = llama_model_load_from_file(path, mp);
    if (!h->model) {
        free(h);
        return NULL;
    }
    char arch[64] = { 0 };
    if (llama_model_meta_val_str(h->model, "general.architecture", arch, sizeof(arch)) < 0 ||
        strcmp(arch, "gemma4") != 0) {
        fprintf(stderr, "strands-decider: %s is a %s model, not gemma4\n", path, arch[0] ? arch : "unnamed");
        llama_model_free(h->model);
        free(h);
        return NULL;
    }
    h->n_embd     = llama_model_n_embd_out(h->model);
    h->n_embd_inp = llama_model_n_embd_inp(h->model);
    h->n_threads  = n_threads > 0 ? n_threads : 1;
    *n_embd       = h->n_embd;
    *n_embd_inp   = h->n_embd_inp;
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
// tokens (280 for Gemma 4).
const char * sd_load_vision(void * handle, const char * mmproj_path, int image_max_tokens) {
    sd_handle * h = (sd_handle *) handle;
    if (h->mtmd) return "a vision tower is already loaded";
    mtmd_log_set(sd_log, NULL);
    struct mtmd_context_params p = mtmd_context_params_default();
    p.use_gpu          = false;
    p.print_timings    = false;
    p.n_threads        = h->n_threads;
    p.warmup           = false;  // the caller warms up with a request
    p.image_max_tokens = image_max_tokens;
    mtmd_context * m   = mtmd_init_from_file(mmproj_path, h->model, p);
    if (!m) return "mtmd could not load it";
    if (!mtmd_support_vision(m) || mtmd_decode_use_mrope(m)) {
        mtmd_free(m);
        return "it has no vision encoder, or wants M-RoPE positions, which Gemma 4's are not";
    }
    h->mtmd = m;
    return NULL;
}

// Run one image, width x height RGB bytes already at the size the image processor gives it,
// through the vision tower and its projection, and copy its soft-token rows to out: n_rows rows
// of n_embd_inp floats, n_rows being the image's placeholders in the prompt.
const char * sd_encode_image(void * handle, const unsigned char * rgb, int width, int height, int n_rows,
                             float * out) {
    sd_handle *         h      = (sd_handle *) handle;
    mtmd_bitmap *       bitmap = mtmd_bitmap_init((uint32_t) width, (uint32_t) height, rgb);
    mtmd_input_chunks * chunks = mtmd_input_chunks_init();
    const char *        err    = bitmap && chunks ? NULL : "could not allocate the image";
    if (!err) {
        const mtmd_input_part   part     = { NULL, bitmap };
        const mtmd_input_part * parts[1] = { &part };
        if (mtmd_tokenize_from_parts(h->mtmd, chunks, parts, 1, false) != 0) err = "mtmd could not tokenize it";
    }
    if (bitmap) mtmd_bitmap_free(bitmap);
    // mtmd puts its own begin and end tokens around the image as text chunks; the prompt has them.
    const mtmd_input_chunk * image = NULL;
    for (size_t i = 0; !err && !image && i < mtmd_input_chunks_size(chunks); i++) {
        const mtmd_input_chunk * c = mtmd_input_chunks_get(chunks, i);
        if (mtmd_input_chunk_get_type(c) == MTMD_INPUT_CHUNK_TYPE_IMAGE) image = c;
    }
    if (!err && !image) err = "mtmd made no image chunk";
    if (!err && mtmd_input_chunk_get_n_tokens(image) != (size_t) n_rows) {
        err = "mtmd gives it another number of soft tokens than the prompt holds";
    }
    if (!err && mtmd_encode_chunk(h->mtmd, image) != 0) err = "the vision tower failed";
    if (!err) memcpy(out, mtmd_get_output_embd(h->mtmd), sizeof(float) * (size_t) n_rows * h->n_embd_inp);
    if (chunks) mtmd_input_chunks_free(chunks);
    return err;
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

// The state as one mixed batch: each position whose id is image_token takes the next of embd's
// n_rows rows, in order, and every other position its token.
static const char * sd_decode_mixed_state(sd_handle * h, struct llama_context * ctx, const int * state_ids,
                                          int n_state, int image_token, const float * embd, int n_rows) {
    int n_placeholders = 0;
    for (int i = 0; i < n_state; i++) n_placeholders += state_ids[i] == image_token;
    if (n_placeholders != n_rows) return "the state's image placeholders and the images' rows differ in number";
    struct llama_batch_ext * be = llama_batch_ext_init(ctx);
    if (!be) return "could not build the mixed batch";
    const char * err = NULL;
    int          row = 0;
    for (int i = 0; i < n_state && !err; i++) {
        int idx;
        if (state_ids[i] == image_token) {
            struct llama_embd e = { embd + (size_t) row++ * h->n_embd_inp, 1, (size_t) h->n_embd_inp };
            idx = llama_batch_ext_add_embd(be, 0, e);
        } else {
            idx = llama_batch_ext_add_token(be, 0, state_ids[i]);
        }
        if (idx < 0) {
            err = "could not build the mixed batch";
            break;
        }
        const llama_pos p = i;
        llama_batch_ext_set_pos(be, idx, &p);
        if (i == n_state - 1) llama_batch_ext_set_output_logits(be, idx, true);
    }
    if (!err && llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, be) != 0) err = "the state's decode failed";
    llama_synchronize(ctx);
    llama_batch_ext_free(be);
    return err;
}

// Decode the state (n_state ids) once, then each question after it on the same cache, and write
// each question's last-token final hidden state to out ([n_q, n_embd], in question order). With
// n_rows > 0 the state's image_token positions take embd's rows, in order. ms gets n_q + 2 timings
// in milliseconds: the context's creation, the state's decode, then each question's decode.
const char * sd_forward(void * handle, const int * state_ids, int n_state, int image_token, const float * embd,
                        int n_rows, const int * q_ids, const int * q_lens, int n_q, float * out, double * ms) {
    sd_handle * h = (sd_handle *) handle;
    if (n_state <= 0 || n_q <= 0) return "an empty state, or no questions";
    int longest = 0;
    for (int i = 0; i < n_q; i++) {
        if (q_lens[i] <= 0) return "an empty question";
        if (q_lens[i] > longest) longest = q_lens[i];
    }
    // A non-causal decode must fit one ubatch, so the ubatch holds the whole state.
    const int  n_batch = n_state > longest ? n_state : longest;
    sd_capture cap     = { NULL, h->n_embd, 0 };

    double                      t0 = sd_ms();
    struct llama_context_params cp = llama_context_default_params();
    cp.n_ctx             = n_state + longest;  // the state and one question
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
    // For measuring precision: STRANDS_DECIDER_KV_F32=1 (an f32 KV cache), STRANDS_DECIDER_FLASH_ATTN=0 (no flash attention).
    if (getenv("STRANDS_DECIDER_KV_F32")) cp.type_k = cp.type_v = GGML_TYPE_F32;
    if (getenv("STRANDS_DECIDER_FLASH_ATTN") && strcmp(getenv("STRANDS_DECIDER_FLASH_ATTN"), "0") == 0) cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    cp.cb_eval           = sd_eval_cb;
    cp.cb_eval_user_data = &cap;
    struct llama_context * ctx = llama_init_from_model(h->model, cp);
    if (!ctx) return "could not create a llama.cpp context";
    llama_set_causal_attn(ctx, false);
    struct llama_batch b   = llama_batch_init(n_batch, 0, 1);
    llama_memory_t     mem = llama_get_memory(ctx);
    double             t1  = sd_ms();
    ms[0]                  = t1 - t0;
    if (!b.token || !b.pos || !b.n_seq_id || !b.seq_id || !b.logits) {
        llama_batch_free(b);
        llama_free(ctx);
        return "could not allocate the batch";
    }

    const char * err = NULL;
    if (n_rows > 0) {
        err = sd_decode_mixed_state(h, ctx, state_ids, n_state, image_token, embd, n_rows);
    } else {
        sd_fill(&b, state_ids, n_state, 0);
        if (llama_decode(ctx, b) != 0) err = "the state's decode failed";
        llama_synchronize(ctx);  // llama_decode can return before the compute ends; time each phase whole
    }
    ms[1] = sd_ms() - t1;

    int off = 0;
    for (int i = 0; i < n_q && !err; i++) {
        double tq = sd_ms();
        sd_fill(&b, q_ids + off, q_lens[i], n_state);
        off     += q_lens[i];
        cap.dst  = out + (size_t) i * h->n_embd;
        cap.got  = 0;
        if (llama_decode(ctx, b) != 0) err = "a question's decode failed";
        llama_synchronize(ctx);
        if (!err && !cap.got) err = "a question's decode gave no hidden state";
        cap.dst = NULL;
        if (!llama_memory_seq_rm(mem, 0, n_state, -1) && !err) err = "could not remove a question's cells";
        ms[2 + i] = sd_ms() - tq;
    }
    llama_batch_free(b);
    llama_free(ctx);
    return err;
}
