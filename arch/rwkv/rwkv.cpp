// Defines fileno on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#include <cstdint>
#include <cstdio>
#endif

#include "../../ggml.h"
#include "../arch-util.h"
#include "rwkv.h"

#include <array>
#include <ctime>
#include <cinttypes>
#include <fstream>
#include <random>
#include <map>
#include <unordered_map>
#include <queue>
#include <cassert>
#include <cstring>
#include <climits>
#include <memory>
#include <algorithm>
#include <initializer_list>
#include <thread>
#include <atomic>
#include <mutex>
#include <sstream>
#include <regex>

#define RWKV_USE_SCRATCH
#define RWKV_MAX_SCRATCH_BUFFERS 16

// RWKV Raven models
enum e_model {
    MODEL_UNKNOWN,
    MODEL_169M,
    MODEL_430M,
    MODEL_3B,
    MODEL_7B,
    MODEL_14B,
    //MODEL_20B,
};

static const size_t MiB = 1024*1024;

// TODO: dynamically determine these sizes
// TODO: To load the stablelm 3B model on my test XR will require some tricks, small ggml context size, mmap support, among others, but is maybe feasible, is a smaller n_ctx required? 512 instead of 2048/4096? Does mmap work as desired on iOS?
//       needs modifications in ggml

// TODO: How are these values actually determined?
// TODO: This is now priority
static const std::map<e_model, size_t> & MEM_REQ_SCRATCH0()
{
    static std::map<e_model, size_t> _MEM_REQ_SCRATCH0 = {
        { MODEL_169M,  256ull * MiB },
        { MODEL_430M,  256ull * MiB },
        { MODEL_3B,    256ull * MiB },
        { MODEL_7B,    512ull * MiB },
        { MODEL_14B,   512ull * MiB },
        //{ MODEL_20B,   512ull * MiB },
    };
    return _MEM_REQ_SCRATCH0;
}

// TODO: How are these values actually determined?
static const std::map<e_model, size_t> & MEM_REQ_SCRATCH1()
{
    static std::map<e_model, size_t> _MEM_REQ_SCRATCH1 = {
        { MODEL_169M,  256ull * MiB },
        { MODEL_430M,  256ull * MiB },
        { MODEL_3B,    256ull * MiB },
        { MODEL_7B,    512ull * MiB },
        { MODEL_14B,   512ull * MiB },
        //{ MODEL_20B,   512ull * MiB },
    };
    return _MEM_REQ_SCRATCH1;
}

// TODO: How are these values actually determined?
// 2*n_embd*n_ctx*n_layer*sizeof(float16)
// llama 7B: 2 * 768 * 32 * 2 = 98304
static const std::map<e_model, size_t> & MEM_REQ_KV_SELF()
{
    static std::map<e_model, size_t> _MEM_REQ_KV_SELF = {
        { MODEL_169M, 512ull * MiB },
        { MODEL_430M, 512ull * MiB },
        { MODEL_3B,   512ull * MiB },
        { MODEL_7B,   1026ull * MiB },
        { MODEL_14B,  1608ull * MiB },
        //{ MODEL_20B,  1608ull * MiB },
    };
    return _MEM_REQ_KV_SELF;
}

// TODO: How are these values actually determined?
// this is mostly needed for temporary mul_mat buffers to dequantize the data
// not actually needed if BLAS is disabled
static const std::map<e_model, size_t> & MEM_REQ_EVAL()
{
    static std::map<e_model, size_t> _MEM_REQ_EVAL = {
        { MODEL_169M, 512ull * MiB },
        { MODEL_430M, 512ull * MiB },
        { MODEL_3B,   512ull * MiB },
        { MODEL_7B,   768ull * MiB },
        { MODEL_14B, 1024ull * MiB },
        //{ MODEL_20B, 1024ull * MiB },
    };
    return _MEM_REQ_EVAL;
}

// default hparams (GPT-NeoX oasst 12B)
struct rwkv_hparams {
    uint32_t n_vocab = 50277;
    uint32_t n_ctx   = 1024;   // this is provided as user input?
    uint32_t n_embd  = 4096;
    //uint32_t n_head  = 40;
    uint32_t n_layer = 32;
    uint32_t rescale_every = 6;
    //uint32_t n_rot   = 32;
    //uint32_t use_parallel_residual = 1; // 1 = true, 0 = false
    enum ggml_ftype ftype = GGML_FTYPE_MOSTLY_F16;

    bool operator!=(const rwkv_hparams & other) const {
        return memcmp(this, &other, sizeof(rwkv_hparams));
    }
};

// rwkv.embeddings.weight
// rwkv.blocks.0.pre_ln.weight
// rwkv.blocks.0.pre_ln.bias

// rwkv.blocks.0.ln1.weight
// rwkv.blocks.0.ln1.bias
// rwkv.blocks.0.ln2.weight
// rwkv.blocks.0.ln2.bias
// rwkv.blocks.0.attention.time_decay
// rwkv.blocks.0.attention.time_first
// rwkv.blocks.0.attention.time_mix_key
// rwkv.blocks.0.attention.time_mix_value
// rwkv.blocks.0.attention.time_mix_receptance
// rwkv.blocks.0.attention.key.weight
// rwkv.blocks.0.attention.value.weight
// rwkv.blocks.0.attention.receptance.weight
// rwkv.blocks.0.attention.output.weight
// rwkv.blocks.0.feed_forward.time_mix_key
// rwkv.blocks.0.feed_forward.time_mix_receptance
// rwkv.blocks.0.feed_forward.key.weight
// rwkv.blocks.0.feed_forward.receptance.weight
// rwkv.blocks.0.feed_forward.value.weight

// rwkv.blocks.1.ln1.weight ...
// ...

// rwkv.ln_out.weight
// rwkv.ln_out.bias
// head.weight

struct rwkv_layer {
    // ln1
    struct ggml_tensor * ln_1_g;
    struct ggml_tensor * ln_1_b;

    // ln2
    struct ggml_tensor * ln_2_g;
    struct ggml_tensor * ln_2_b;

    // attention
    struct ggml_tensor * attn_time_decay;
    struct ggml_tensor * attn_time_first;
    struct ggml_tensor * attn_time_mix_k;
    struct ggml_tensor * attn_time_mix_v;
    struct ggml_tensor * attn_time_mix_r;
    struct ggml_tensor * attn_k_w;
    struct ggml_tensor * attn_v_w;
    struct ggml_tensor * attn_r_w;
    struct ggml_tensor * attn_out_w;

    // ff
    struct ggml_tensor * ff_time_mix_k;
    struct ggml_tensor * ff_time_mix_r;
    struct ggml_tensor * ff_k_w;
    struct ggml_tensor * ff_r_w;
    struct ggml_tensor * ff_v_w;
};

struct rwkv_state {
    // Input token
    struct ggml_tensor * token;
    
    // Output logits
    struct ggml_tensor * logits;
    
    // Layers states
    struct ggml_tensor * input_state;
    struct ggml_tensor * output_state;
    std::unique_ptr<struct rwkv_layer_state []> input_layers;
    std::unique_ptr<struct rwkv_layer_state []> output_layers;
    
    struct ggml_context * ctx = NULL;
    arch_util_buffer buf;
    //int n; // number of tokens currently in the state (5)

    ~rwkv_state() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

struct rwkv_model {
    e_model type = MODEL_UNKNOWN;

    rwkv_hparams hparams;
    
    // word embedding
    struct ggml_tensor * wte;
    
    // pre normalization
    struct ggml_tensor * ln_pre_g;
    struct ggml_tensor * ln_pre_b;

    // final normalization
    struct ggml_tensor * ln_out_g;
    struct ggml_tensor * ln_out_b;

    // language model head
    struct ggml_tensor * lmh_w;

    std::vector<rwkv_layer> layers;

    // context
    struct ggml_context * ctx = NULL;
    
    // state
    struct rwkv_state state;
    
    // rescale_every
    bool layers_are_rescaled = false;

    // the model memory buffer
    arch_util_buffer buf;

    // model memory mapped file
    std::unique_ptr<arch_util_mmap> mapping;

    // objects representing data potentially being locked in memory
    arch_util_mlock mlock_buf;
    arch_util_mlock mlock_mmap;

    // for quantize-stats only
    std::vector<std::pair<std::string, struct ggml_tensor *>> tensors_by_name;

    ~rwkv_model() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

struct rwkv_vocab {
    using id    = int32_t;
    using token = std::string;

    /*struct token_score {
        token tok;
        float score;
    };*/

    std::unordered_map<token, id> token_to_id;
    std::vector<token /*token_score*/> id_to_token;
};

struct rwkv_context {
    std::mt19937 rng;

    int64_t t_load_us = 0;
    int64_t t_start_us = 0;
    bool has_evaluated_once = false;

    int64_t t_sample_us = 0;
    int64_t t_eval_us   = 0;
    int64_t t_p_eval_us = 0;

    int32_t n_sample = 0; // number of tokens sampled
    int32_t n_eval   = 0; // number of eval calls
    int32_t n_p_eval = 0; // number of tokens in eval calls for the prompt (with batch size > 1)

    rwkv_model model;
    rwkv_vocab vocab;
    
    struct ggml_context * ctx = NULL;
    struct ggml_cgraph cg = {};

    size_t mem_per_token = 0;

    // decode output (1-dimensional array: [n_vocab])
    std::vector<float> logits;

    // input embedding (1-dimensional array: [n_embd])
    std::vector<float> embedding;

    // memory buffers used to evaluate the model
    // TODO: move in rwkv_state?
    arch_util_buffer buf_compute;
    arch_util_buffer buf_scratch[RWKV_MAX_SCRATCH_BUFFERS];

    int    buf_last = 0;
    size_t buf_max_size[RWKV_MAX_SCRATCH_BUFFERS] = { 0 };
    
    ~rwkv_context() {
        if (ctx) {
            ggml_free(ctx);
        }
    }

    void use_buf(struct ggml_context * ctx, int i) {
#if defined(RWKV_USE_SCRATCH)
        size_t last_size = 0;

        if (i == -1) {
            last_size = ggml_set_scratch(ctx, { 0, 0, nullptr, });
        } else {
            auto & buf = buf_scratch[i];
            last_size = ggml_set_scratch(ctx, { 0, buf.size, buf.addr, });
        }

        if (buf_last >= 0) {
            buf_max_size[buf_last] = std::max(buf_max_size[buf_last], last_size);
        }

        buf_last = i;
#else
        (void) i;
        (void) ctx;
#endif
    }

    size_t get_buf_max_mem(int i) const {
#if defined(RWKV_USE_SCRATCH)
        return buf_max_size[i];
#else
        (void) i;
        return 0;
#endif
    }
};

template <typename T>
static T checked_mul(T a, T b) {
    T ret = a * b;
    if (a != 0 && ret / a != b) {
        throw format("overflow multiplying %llu * %llu",
                     (unsigned long long) a, (unsigned long long) b);
    }
    return ret;
}

static size_t checked_div(size_t a, size_t b) {
    if (b == 0 || a % b != 0) {
        throw format("error dividing %zu / %zu", a, b);
    }
    return a / b;
}

static std::string rwkv_format_tensor_shape(const std::vector<uint32_t> & ne) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%5u", ne.at(0));
    for (size_t i = 1; i < ne.size(); i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), " x %5u", ne.at(i));
    }
    return buf;
}

static size_t rwkv_calc_tensor_size(const std::vector<uint32_t> & ne, enum ggml_type type) {
    size_t size = ggml_type_size(type);
    for (uint32_t dim : ne) {
        size = checked_mul<size_t>(size, dim);
    }
    return size / ggml_blck_size(type);
}

struct rwkv_load_tensor_shard {
    std::vector<uint32_t> ne;
    size_t size;
    enum ggml_type type;
    size_t file_idx;
    size_t file_off;

    void calc_size() {
        size = rwkv_calc_tensor_size(ne, type);
    }
};

enum rwkv_split_type {
    SPLIT_NONE,
    SPLIT_BY_COLUMNS,
    SPLIT_BY_ROWS
};

struct rwkv_load_tensor {
    std::vector<rwkv_load_tensor_shard> shards;

    std::string name;
    enum ggml_type type = GGML_TYPE_F32;
    rwkv_split_type split_type = SPLIT_NONE;
    std::vector<uint32_t> ne;
    size_t size;
    struct ggml_tensor * ggml_tensor = NULL;
    uint8_t * data;

    rwkv_load_tensor(const std::string & name) : name(name) {}

    void calc_all() {
        calc_type();
        calc_split_type();
        calc_ne();
        calc_size();
    }

    void calc_type() {
        const auto & first_shard = shards.at(0);
        for (const auto & shard : shards) {
            if (shard.type != first_shard.type) {
                throw format("inconsistent tensor shard type in '%s'", name.c_str());
            }
        }
        type = first_shard.type;
    }

    void calc_split_type() {
        if (shards.at(0).ne.size() == 1 || // 1D tensors are just duplicated in every file
            shards.size() == 1) { // only one file?
            split_type = SPLIT_NONE;
        } else if (name.find("rwkv.embeddings.") == 0 ||
            name.find(".attention.wo.weight") != std::string::npos ||
            name.find(".feed_forward.w2.weight") != std::string::npos) {
            split_type = SPLIT_BY_COLUMNS;
        } else {
            split_type = SPLIT_BY_ROWS;
        }
    }

    void calc_ne() {
        const auto & first_shard = shards.at(0);
        for (const auto & shard : shards) {
            if (shard.ne != first_shard.ne) {
                throw format("inconsistent tensor shard shape in '%s': first was %s, other was %s",
                             name.c_str(), rwkv_format_tensor_shape(first_shard.ne).c_str(), rwkv_format_tensor_shape(shard.ne).c_str());
            }
        }
        ne = first_shard.ne;
        ARCH_ASSERT(shards.size() <= UINT32_MAX);
        uint32_t n_shards = (uint32_t) shards.size();
        switch (split_type) {
            case SPLIT_NONE:
                ne = first_shard.ne;
                break;
            case SPLIT_BY_COLUMNS:
                ne = {checked_mul<uint32_t>(first_shard.ne[0], n_shards),
                      first_shard.ne[1]};
                break;
            case SPLIT_BY_ROWS:
                ne = {first_shard.ne[0],
                      checked_mul<uint32_t>(first_shard.ne[1], n_shards)};
                break;
        }
    }

    void calc_size() {
        size = rwkv_calc_tensor_size(ne, type);
    }
};

struct rwkv_load_tensors_map {
    // tensors is kept in a separate vector to preserve file order
    std::vector<rwkv_load_tensor> tensors;
    std::unordered_map<std::string, size_t> name_to_idx;
};

enum ggml_file_version {
    GGML_FILE_VERSION_GGML,
    GGML_FILE_VERSION_GGMF_V1, // added version field and scores in vocab
    GGML_FILE_VERSION_GGJT_V1, // added padding
};

struct rwkv_file_loader {
    arch_util_file file;
    ggml_file_version file_version;
    rwkv_hparams hparams;
    rwkv_vocab vocab;

    rwkv_file_loader(const char * fname, size_t file_idx, rwkv_load_tensors_map & tensors_map)
        : file(fname, "rb") {
        fprintf(stderr, "rwkv: loading model from %s\n", fname);
        read_magic();
        read_hparams();
        read_vocab();
        read_tensor_metadata(file_idx, tensors_map);
    }
    void read_magic() {
        uint32_t magic = file.read_u32();
        uint32_t version = 0;

        if (magic != 'ggml') {
            version = file.read_u32();
        }

        if (magic == 'ggml' && version == 0) {
            file_version = GGML_FILE_VERSION_GGML;
        } else if (magic == 'ggmf' && version == 1) {
            file_version = GGML_FILE_VERSION_GGMF_V1;
        } else if (magic == 'ggjt' && version == 1) {
            file_version = GGML_FILE_VERSION_GGJT_V1;
        } else {
            throw format("unknown (magic, version) combination: %08x, %08x; is this really a GGML file?",
                         magic, version);
        }
    }
    void read_hparams() {
        hparams.n_vocab = file.read_u32();
        hparams.n_ctx = file.read_u32();
        hparams.n_embd = file.read_u32();
        //hparams.n_head = file.read_u32();
        hparams.n_layer = file.read_u32();
        hparams.rescale_every = file.read_u32();
        //hparams.n_rot = file.read_u32();
        //hparams.use_parallel_residual = file.read_u32();
        hparams.ftype = (enum ggml_ftype) file.read_u32();
    }
    void read_vocab() {
        vocab.id_to_token.resize(hparams.n_vocab);

        for (uint32_t i = 0; i < hparams.n_vocab; i++) {
            uint32_t len = file.read_u32();
            std::string word = file.read_string(len);

            /*float score = 0.0f;
            if (file_version >= GGML_FILE_VERSION_GGMF_V1) {
                file.read_raw(&score, sizeof(score));
            }*/

            vocab.token_to_id[word] = i;

            vocab.id_to_token[i] = word;
            /*
            auto & tok_score = vocab.id_to_token[i];
            tok_score.tok = std::move(word);
            tok_score.score = score;
             */
        }
    }
    void read_tensor_metadata(size_t file_idx, rwkv_load_tensors_map & tensors_map) {
        while (file.tell() < file.size) {
            rwkv_load_tensor_shard shard;
            uint32_t n_dims = file.read_u32();
            uint32_t name_len = file.read_u32();
            shard.type = (enum ggml_type) file.read_u32();
            shard.ne.resize(n_dims);
            file.read_raw(shard.ne.data(), sizeof(shard.ne[0]) * n_dims);
            std::string name = file.read_string(name_len);
            if (n_dims < 1 || n_dims > 2) {
                throw format("rwkv: tensor '%s' should not be %u-dimensional", name.c_str(), n_dims);
            }
            switch (shard.type) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q4_2:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    break;
                default: {
                    throw format("unrecognized tensor type %u\n", shard.type);
                }
            }

            if (file_version >= GGML_FILE_VERSION_GGJT_V1) {
                // skip to the next multiple of 32 bytes
                file.seek(-file.tell() & 31, SEEK_CUR);
            }
            shard.file_idx = file_idx;
            shard.file_off = file.tell();

            shard.calc_size();
            file.seek(shard.size, SEEK_CUR);

            auto it = tensors_map.name_to_idx.find(name);
            size_t idx;
            if (it != tensors_map.name_to_idx.end()) {
                idx = it->second;
            } else {
                tensors_map.tensors.emplace_back(name);
                idx = tensors_map.tensors.size() - 1;
                tensors_map.name_to_idx.emplace(name, idx);
            }
            tensors_map.tensors.at(idx).shards.push_back(shard);
        }
    }
};

struct rwkv_file_saver {
    arch_util_file file;
    rwkv_file_loader * any_file_loader;
    rwkv_file_saver(const char * fname, rwkv_file_loader * any_file_loader, enum ggml_ftype new_ftype)
        : file(fname, "wb"), any_file_loader(any_file_loader) {
        fprintf(stderr, "rwkv: saving model to %s\n", fname);
        write_magic();
        write_hparams(new_ftype);
        write_vocab();
    }
    void write_magic() {
        file.write_u32('ggjt'); // magic
        file.write_u32(1); // version
    }
    void write_hparams(enum ggml_ftype new_ftype) {
        const rwkv_hparams & hparams = any_file_loader->hparams;
        file.write_u32(hparams.n_vocab);
        file.write_u32(hparams.n_ctx);
        file.write_u32(hparams.n_embd);
        //file.write_u32(hparams.n_head);
        file.write_u32(hparams.n_layer);
        file.write_u32(hparams.rescale_every);
        //file.write_u32(hparams.n_rot);
        //file.write_u32(hparams.use_parallel_residual);
        file.write_u32(new_ftype);
    }
    void write_vocab() {
        if (any_file_loader->file_version == GGML_FILE_VERSION_GGML) {
            fprintf(stderr, "rwkv: WARNING: input is an old file that doesn't have scores; will add dummy scores\n");
        }
        uint32_t n_vocab = any_file_loader->hparams.n_vocab;
        for (uint32_t i = 0; i < n_vocab; i++) {
            const auto & token = any_file_loader->vocab.id_to_token.at(i);
            file.write_u32((uint32_t) token.size());
            file.write_raw(token.data(), token.size());
            /*
            const auto & token_score = any_file_loader->vocab.id_to_token.at(i);
            file.write_u32((uint32_t) token_score.tok.size());
            file.write_raw(token_score.tok.data(), token_score.tok.size());
            file.write_raw(&token_score.score, sizeof(token_score.score));
             */
        }
    }
    void write_tensor(rwkv_load_tensor & tensor, enum ggml_type new_type, const void * new_data, size_t new_size) {
        switch (new_type) {
            case GGML_TYPE_F32:
            case GGML_TYPE_F16:
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q4_2:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_0:
                break;
            default: ARCH_ASSERT(false);
        }
        file.write_u32((uint32_t) tensor.ne.size());
        file.write_u32((uint32_t) tensor.name.size());
        file.write_u32(new_type);
        file.write_raw(tensor.ne.data(), sizeof(tensor.ne[0]) * tensor.ne.size());
        file.write_raw(tensor.name.data(), tensor.name.size());
        file.seek(-file.tell() & 31, SEEK_CUR);
        ARCH_ASSERT(new_size == rwkv_calc_tensor_size(tensor.ne, new_type));
        file.write_raw(new_data, new_size);
    }
};

struct rwkv_model_loader {
    std::vector<std::unique_ptr<rwkv_file_loader>> file_loaders;
    rwkv_load_tensors_map tensors_map;
    bool use_mmap;
    size_t num_ggml_tensors_created = 0;
    struct ggml_context * ggml_ctx = NULL;
    std::unique_ptr<arch_util_mmap> mapping;

    rwkv_model_loader(const std::string & fname_base, bool use_mmap, bool vocab_only) {
        auto first_file = new rwkv_file_loader(fname_base.c_str(), 0, tensors_map);
        file_loaders.emplace_back(first_file);
        uint32_t n_parts = vocab_only ? 1 : guess_n_parts();
        for (uint32_t i = 1; i < n_parts; i++) {
            std::string fname = fname_base + "." + std::to_string(i);
            auto ith_file = new rwkv_file_loader(fname.c_str(), i, tensors_map);
            file_loaders.emplace_back(ith_file);
            if (ith_file->hparams != first_file->hparams) {
                throw format("rwkv: hparams inconsistent between files");
            }
        }
        if (!arch_util_mmap::SUPPORTED) {
            use_mmap = false;
        }
        if (use_mmap && alignment_prevents_mmap()) {
            fprintf(stderr, "rwkv: can't use mmap because tensors are not aligned; convert to new format to avoid this\n");
            use_mmap = false;
        }
        this->use_mmap = use_mmap;
        for (rwkv_load_tensor & lt : tensors_map.tensors) {
            lt.calc_all();
        }
    }

    bool alignment_prevents_mmap() {
        for (const rwkv_load_tensor & lt : tensors_map.tensors) {
            for (const rwkv_load_tensor_shard & shard : lt.shards) {
                if (shard.file_off & 3) {
                    return true;
                }
            }
        }
        return false;
    }

    uint32_t guess_n_parts() const {
        auto it = tensors_map.name_to_idx.find("rwkv.embeddings.weight");
        if (it == tensors_map.name_to_idx.end()) {
            throw std::string("missing rwkv.embeddings.weight");
        }
        const rwkv_load_tensor & lt = tensors_map.tensors.at(it->second);
        return file_loaders.at(0)->hparams.n_embd / lt.shards.at(0).ne.at(0);
    }

    void calc_sizes(size_t * ctx_size_p, size_t * mmapped_size_p) const {
        *ctx_size_p = *mmapped_size_p = 0;
        for (const rwkv_load_tensor & lt : tensors_map.tensors) {
            *ctx_size_p += sizeof(struct ggml_tensor) + GGML_OBJECT_SIZE;
            *(use_mmap ? mmapped_size_p : ctx_size_p) += lt.size;
        }
    }

    struct ggml_tensor * get_tensor(const std::string & name, std::vector<uint32_t> ne) {
        auto it = tensors_map.name_to_idx.find(name);
        if (it == tensors_map.name_to_idx.end()) {
            throw format("rwkv: tensor '%s' is missing from model", name.c_str());
        }
        rwkv_load_tensor & lt = tensors_map.tensors.at(it->second);
        if (lt.ne != ne) {
            throw format("rwkv: tensor '%s' has wrong shape; expected %s, got %s",
                         name.c_str(), rwkv_format_tensor_shape(ne).c_str(), rwkv_format_tensor_shape(lt.ne).c_str());
        }
        
        // Debug tensor
        /*printf("%48s - %14s, type = %4s\n",
               lt.name.c_str(),
               rwkv_format_tensor_shape(lt.ne).c_str(),
               ggml_type_name(lt.type));*/

        return get_tensor_for(lt);
    }

    struct ggml_tensor * get_tensor_for(rwkv_load_tensor & lt) {
        struct ggml_tensor * tensor;
        if (lt.ne.size() == 2) {
            tensor = ggml_new_tensor_2d(ggml_ctx, lt.type, lt.ne.at(0), lt.ne.at(1));
        } else {
            ARCH_ASSERT(lt.ne.size() == 1);
            tensor = ggml_new_tensor_1d(ggml_ctx, lt.type, lt.ne.at(0));
        }
        ARCH_ASSERT(lt.ggml_tensor == NULL); // if this fails, we called get_tensor twice on the same tensor
        lt.ggml_tensor = tensor;
        num_ggml_tensors_created++;
        return tensor;
    }

    void done_getting_tensors() {
        if (num_ggml_tensors_created != tensors_map.tensors.size()) {
            throw std::string("rwkv: file contained more tensors than expected");
        }
    }

    void load_all_data(rwkv_progress_callback progress_callback, void *  progress_callback_user_data, arch_util_mlock * lmlock) {
        size_t data_size = 0;
        for (const rwkv_load_tensor & lt : tensors_map.tensors) {
            data_size += lt.size;
        }

        if (use_mmap) {
            mapping.reset(new arch_util_mmap(&file_loaders.at(0)->file));
            if (!lmlock) {
                // Don't call the callback since the actual loading will be lazy
                // and we can't measure it.
                progress_callback = NULL;
            }
            if (lmlock) {
                lmlock->init(mapping->addr);
            }
        }

        size_t done_size = 0;
        for (rwkv_load_tensor & lt : tensors_map.tensors) {
            if (progress_callback) {
                progress_callback((float) done_size / data_size, progress_callback_user_data);
            }
            ARCH_ASSERT(lt.ggml_tensor); // unused tensors should have been caught by load_data already
            lt.data = (uint8_t *) lt.ggml_tensor->data;
            load_data_for(lt);
            lt.ggml_tensor->data = lt.data;
            done_size += lt.size;
            if (use_mmap && lmlock) {
                lmlock->grow_to(done_size);
            }
        }
        if (progress_callback) {
            progress_callback(1.0f, progress_callback_user_data);
        }
    }

    void load_data_for(rwkv_load_tensor & lt) {
        if (use_mmap) {
            ARCH_ASSERT(lt.shards.size() == 1);
            lt.data = (uint8_t *) mapping->addr + lt.shards.at(0).file_off;
        } else if (lt.split_type == SPLIT_NONE) {
            arch_util_file & file = file_loaders.at(lt.shards.at(0).file_idx)->file;
            file.seek(lt.shards.at(0).file_off, SEEK_SET);
            file.read_raw(lt.data, lt.size);
        } else if (lt.split_type == SPLIT_BY_ROWS) {
            size_t offset = 0;
            for (rwkv_load_tensor_shard & shard : lt.shards) {
                arch_util_file & file = file_loaders.at(shard.file_idx)->file;
                file.seek(shard.file_off, SEEK_SET);
                file.read_raw(lt.data + offset, shard.size);
                offset += shard.size;
            }
            ARCH_ASSERT(offset == lt.size);
        } else if (lt.split_type == SPLIT_BY_COLUMNS) {
            // Let's load the data into temporary buffers to ensure the OS performs large loads.
            std::vector<arch_util_buffer> tmp_bufs;
            tmp_bufs.resize(lt.shards.size());
            for (size_t i = 0; i < lt.shards.size(); i++) {
                rwkv_load_tensor_shard & shard = lt.shards.at(i);
                arch_util_file & file = file_loaders.at(shard.file_idx)->file;
                file.seek(shard.file_off, SEEK_SET);
                tmp_bufs.at(i).resize(shard.size);
                file.read_raw(tmp_bufs.at(i).addr, shard.size);
            }
            // Then reshape.
            size_t num_rows = lt.ne.at(1);
            size_t per_shard_row_size = lt.shards.at(0).size / num_rows;
            size_t out_offset = 0;
            for (size_t row = 0; row < num_rows; row++) {
                for (arch_util_buffer & tmp_buf : tmp_bufs) {
                    memcpy(lt.data + out_offset,
                           tmp_buf.addr + row * per_shard_row_size,
                           per_shard_row_size);
                    out_offset += per_shard_row_size;
                }
            }
            ARCH_ASSERT(out_offset == lt.size);
        }
        if (0) {
            print_checksum(lt);
        }
    }

    static void print_checksum(rwkv_load_tensor & lt) {
        uint32_t sum = 0;
        for (size_t i = 0; i < lt.size; i++) {
            uint8_t byte = lt.data[i];
            sum = byte + (sum << 6) + (sum << 16) - sum; // sdbm hash
        }
        fprintf(stderr, "%s checksum: %#08x (%s, size %zu)\n", lt.name.c_str(), sum,
                rwkv_format_tensor_shape(lt.ne).c_str(), lt.size);
    }

};


//
// rwkv state
//

struct rwkv_layer_state {
    struct ggml_tensor * attn;
    struct ggml_tensor * num;
    struct ggml_tensor * den;
    struct ggml_tensor * max;
    struct ggml_tensor * ff;
};

static bool rwkv_state_init(const struct rwkv_hparams & hparams,
                                    struct rwkv_state & state,
                                            ggml_type   wtype) {
    const int n_embd  = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_vocab = hparams.n_vocab;
    
    const int64_t n_elements = n_embd * (int64_t)n_layer;

    // TODO: Extra overflow MiB needed?
    size_t state_size = 2 * 5 * n_elements * ggml_type_size(wtype);
    size_t logits_size = n_vocab * ggml_type_size(GGML_TYPE_F32);
    size_t token_size = 1 * ggml_type_size(GGML_TYPE_I32);
    state.buf.resize(state_size + logits_size + token_size + MiB);

    struct ggml_init_params params;
    params.mem_size   = state.buf.size;
    params.mem_buffer = state.buf.addr;
    params.no_alloc   = false;

    state.ctx = ggml_init(params);

    if (!state.ctx) {
        fprintf(stderr, "%s: failed to allocate memory for rwkv state\n", __func__);
        return false;
    }
    
    struct ggml_tensor * token = ggml_new_i32(state.ctx, 0);
    struct ggml_tensor * logits = ggml_new_tensor_1d(state.ctx, GGML_TYPE_F32, n_vocab);
    
    struct ggml_tensor * input  = ggml_new_tensor_1d(state.ctx, wtype, n_embd * 5 * n_layer);
    struct ggml_tensor * output = ggml_new_tensor_1d(state.ctx, wtype, n_embd * 5 * n_layer);
    
    std::unique_ptr<struct rwkv_layer_state []> inputs(new(std::nothrow) struct rwkv_layer_state [n_layer]);
    std::unique_ptr<struct rwkv_layer_state []> outputs(new(std::nothrow) struct rwkv_layer_state [n_layer]);
    
    for(int i = 0; i < n_layer; i++) {
        struct rwkv_layer_state & input_state = inputs[i];
        input_state.attn = ggml_view_1d(state.ctx, input, n_embd, n_embd * (i * 5 + 0) * sizeof(float));
        input_state.num  = ggml_view_1d(state.ctx, input, n_embd, n_embd * (i * 5 + 1) * sizeof(float));
        input_state.den  = ggml_view_1d(state.ctx, input, n_embd, n_embd * (i * 5 + 2) * sizeof(float));
        input_state.max  = ggml_view_1d(state.ctx, input, n_embd, n_embd * (i * 5 + 3) * sizeof(float));
        input_state.ff   = ggml_view_1d(state.ctx, input, n_embd, n_embd * (i * 5 + 4) * sizeof(float));

        struct rwkv_layer_state & output_state = outputs[i];
        output_state.attn = ggml_view_1d(state.ctx, output, n_embd, n_embd * (i * 5 + 0) * sizeof(float));
        output_state.num  = ggml_view_1d(state.ctx, output, n_embd, n_embd * (i * 5 + 1) * sizeof(float));
        output_state.den  = ggml_view_1d(state.ctx, output, n_embd, n_embd * (i * 5 + 2) * sizeof(float));
        output_state.max  = ggml_view_1d(state.ctx, output, n_embd, n_embd * (i * 5 + 3) * sizeof(float));
        output_state.ff   = ggml_view_1d(state.ctx, output, n_embd, n_embd * (i * 5 + 4) * sizeof(float));
        
        ggml_set_f32(input_state.attn,  0.0f);
        ggml_set_f32(input_state.num,   0.0f);
        ggml_set_f32(input_state.den,   0.0f);
        ggml_set_f32(input_state.max, -1e30f); //-1e38f);
        ggml_set_f32(input_state.ff,    0.0f);
        
        ggml_set_f32(output_state.attn,  0.0f);
        ggml_set_f32(output_state.num,   0.0f);
        ggml_set_f32(output_state.den,   0.0f);
        ggml_set_f32(output_state.max, -1e30f); //-1e38f);
        ggml_set_f32(output_state.ff,    0.0f);
        
        strcpy(input_state.attn->name, (std::to_string(i) + " in attn").c_str());
        strcpy(input_state.num->name, (std::to_string(i) + " in num").c_str());
        strcpy(input_state.den->name, (std::to_string(i) + " in den").c_str());
        strcpy(input_state.max->name, (std::to_string(i) + " in max").c_str());
        strcpy(input_state.ff->name, (std::to_string(i) + " in ff").c_str());
        
        strcpy(output_state.attn->name, (std::to_string(i) + " out attn").c_str());
        strcpy(output_state.num->name, (std::to_string(i) + " out num").c_str());
        strcpy(output_state.den->name, (std::to_string(i) + " out den").c_str());
        strcpy(output_state.max->name, (std::to_string(i) + " out max").c_str());
        strcpy(output_state.ff->name, (std::to_string(i) + " out ff").c_str());
    }
    
    state.token         = token;
    state.logits        = logits;
    state.input_state   = input;
    state.output_state  = output;
    state.input_layers  = std::move(inputs);
    state.output_layers = std::move(outputs);

    return true;
}


struct rwkv_context_params rwkv_context_default_params() {
    struct rwkv_context_params result = {
        /*.n_ctx                       =*/ 512,
        /*.n_parts                     =*/ -1,
        /*.seed                        =*/ 0,
        /*.f16_rwkv_state              =*/ false,
        /*.logits_all                  =*/ //false,
        /*.vocab_only                  =*/ false,
        /*.use_mmap                    =*/ true,
        /*.use_mlock                   =*/ false,
        /*.embedding                   =*/ false,
        /*.progress_callback           =*/ nullptr,
        /*.progress_callback_user_data =*/ nullptr,
    };

    return result;
}

bool rwkv_mmap_supported() {
    return arch_util_mmap::SUPPORTED;
}

bool rwkv_mlock_supported() {
    return arch_util_mlock::SUPPORTED;
}

//
// model loading
//

static const char *ggml_file_version_name(ggml_file_version version) {
    switch (version) {
        case GGML_FILE_VERSION_GGML: return "'ggml' (old version with low tokenizer quality and no mmap support)";
        case GGML_FILE_VERSION_GGMF_V1: return "ggmf v1 (old version with no mmap support)";
        case GGML_FILE_VERSION_GGJT_V1: return "ggjt v1 (latest)";
        default: ARCH_ASSERT(false);
    }
}

static const char *ggml_ftype_name(enum ggml_ftype ftype) {
    switch (ftype) {
        case GGML_FTYPE_ALL_F32:     return "all F32";
        case GGML_FTYPE_MOSTLY_F16:  return "mostly F16";
        case GGML_FTYPE_MOSTLY_Q4_0: return "mostly Q4_0";
        case GGML_FTYPE_MOSTLY_Q4_1: return "mostly Q4_1";
        case GGML_FTYPE_MOSTLY_Q4_1_SOME_F16:
                                      return "mostly Q4_1, some F16";
        case GGML_FTYPE_MOSTLY_Q4_2: return "mostly Q4_2";
        //case GGML_FTYPE_MOSTLY_Q4_3: return "mostly Q4_3";
        case GGML_FTYPE_MOSTLY_Q5_0: return "mostly Q5_0";
        case GGML_FTYPE_MOSTLY_Q5_1: return "mostly Q5_1";
        case GGML_FTYPE_MOSTLY_Q8_0: return "mostly Q8_0";
        default:                      return "unknown, may not work";
    }
}

static const char *rwkv_model_type_name(e_model type) {
    switch (type) {
        case MODEL_169M: return "169M";
        case MODEL_430M: return "430M";
        case MODEL_3B: return "3B";
        case MODEL_7B: return "7B";
        case MODEL_14B: return "14B";
        //case MODEL_20B: return "20B";
        case MODEL_UNKNOWN: return "UNKNOWN";
        default: ARCH_ASSERT(false);
    }
}

static void rwkv_model_load_internal(
        const std::string & fname,
        rwkv_context & lctx,
        int n_ctx,
        ggml_type memory_type,
        bool use_mmap,
        bool use_mlock,
        bool vocab_only,
        rwkv_progress_callback progress_callback,
        void * progress_callback_user_data) {

    lctx.t_start_us = ggml_time_us();

    std::unique_ptr<rwkv_model_loader> ml(new rwkv_model_loader(fname, use_mmap, vocab_only));

    lctx.vocab = std::move(ml->file_loaders.at(0)->vocab);
    auto & model = lctx.model;
    model.hparams = ml->file_loaders.at(0)->hparams;
    ggml_file_version file_version = ml->file_loaders.at(0)->file_version;
    auto & hparams = model.hparams;
    
    {
        switch (hparams.n_layer) {
            //case 16: model.type = e_model::MODEL_3B; break;
            case 12: model.type = e_model::MODEL_169M; break;
            case 24: model.type = e_model::MODEL_430M; break;
            case 32: {
                if (hparams.n_embd == 2560) model.type = e_model::MODEL_3B;
                if (hparams.n_embd == 4096) model.type = e_model::MODEL_7B;
                break;
            }
            case 40: model.type = e_model::MODEL_14B; break;
            //case 44: model.type = e_model::MODEL_20B; break;
        }

        hparams.n_ctx = n_ctx;
    }

    {
        fprintf(stderr, "%s: format     = %s\n",  __func__, ggml_file_version_name(file_version));
        fprintf(stderr, "%s: n_vocab    = %u\n",  __func__, hparams.n_vocab);
        fprintf(stderr, "%s: n_ctx      = %u\n",  __func__, hparams.n_ctx);
        fprintf(stderr, "%s: n_embd     = %u\n",  __func__, hparams.n_embd);
        //fprintf(stderr, "%s: n_head     = %u\n",  __func__, hparams.n_head);
        fprintf(stderr, "%s: n_layer    = %u\n",  __func__, hparams.n_layer);
        fprintf(stderr, "%s: rescale_every = %u\n",  __func__, hparams.rescale_every);
        //fprintf(stderr, "%s: n_rot      = %u\n",  __func__, hparams.n_rot);
        //fprintf(stderr, "%s: use_parallel_residual = %d\n", __func__, hparams.use_parallel_residual);
        fprintf(stderr, "%s: ftype      = %u (%s)\n", __func__, hparams.ftype, ggml_ftype_name(hparams.ftype));
        fprintf(stderr, "%s: n_parts    = %zu\n", __func__, ml->file_loaders.size());
        fprintf(stderr, "%s: model size = %s\n",  __func__, rwkv_model_type_name(model.type));
    }

    if (vocab_only) {
        return;
    }

    size_t ctx_size, mmapped_size;
    ml->calc_sizes(&ctx_size, &mmapped_size);
    fprintf(stderr, "%s: ggml ctx size = %6.2f KiB\n", __func__, ctx_size/1024.0);

    // print memory requirements
    {
        const size_t scale = memory_type == GGML_TYPE_F32 ? 2 : 1;

        // this is the total memory required to run the inference
        const size_t mem_required =
            ctx_size +
            mmapped_size +
            MEM_REQ_SCRATCH0().at(model.type) +
            MEM_REQ_SCRATCH1().at(model.type) +
            MEM_REQ_EVAL().at(model.type);

        // this is the memory required by one rwkv_state
        const size_t mem_required_state =
            scale*MEM_REQ_KV_SELF().at(model.type);

        fprintf(stderr, "%s: mem required  = %7.2f MiB (+ %7.2f MiB per state)\n", __func__,
                mem_required / 1024.0 / 1024.0, mem_required_state / 1024.0 / 1024.0);
    }

    // create the ggml context
    {
        lctx.model.buf.resize(ctx_size);
        if (use_mlock) {
            lctx.model.mlock_buf.init(lctx.model.buf.addr);
            lctx.model.mlock_buf.grow_to(lctx.model.buf.size);
        }

        struct ggml_init_params params = {
            /*.mem_size   =*/ lctx.model.buf.size,
            /*.mem_buffer =*/ lctx.model.buf.addr,
            /*.no_alloc   =*/ ml->use_mmap,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            throw format("ggml_init() failed");
        }
    }
    
    auto & ctx = model.ctx;

    // prepare memory for the weights
    {
        const auto & hparams = model.hparams;

        const uint32_t n_embd  = hparams.n_embd;
        const uint32_t n_layer = hparams.n_layer;
        const uint32_t n_vocab = hparams.n_vocab;

        ml->ggml_ctx = ctx;

        model.wte       = ml->get_tensor("rwkv.embeddings.weight",      {n_embd, n_vocab});
        strcpy(model.wte->name, "wte");
        model.ln_pre_g  = ml->get_tensor("rwkv.blocks.0.pre_ln.weight", {n_embd});
        strcpy(model.ln_pre_g->name, "ln_pre_g");
        model.ln_pre_b  = ml->get_tensor("rwkv.blocks.0.pre_ln.bias",   {n_embd});
        strcpy(model.ln_pre_b->name, "ln_pre_b");
        
        model.ln_out_g  = ml->get_tensor("rwkv.ln_out.weight",          {n_embd});
        strcpy(model.ln_out_g->name, "ln_out_g");
        model.ln_out_b  = ml->get_tensor("rwkv.ln_out.bias",            {n_embd});
        strcpy(model.ln_out_b->name, "ln_out_b");
        model.lmh_w     = ml->get_tensor("head.weight",                 {n_embd, n_vocab});
        strcpy(model.lmh_w->name, "lmh_w");

        model.layers.resize(n_layer);
        for (uint32_t i = 0; i < n_layer; i++) {
            auto & layer = model.layers[i];

            std::string str_i = std::to_string(i);
            std::string layers_i = "rwkv.blocks." + str_i;

            layer.ln_1_g = ml->get_tensor(layers_i + ".ln1.weight", {n_embd});
            strcpy(layer.ln_1_g->name, (str_i + ".ln_1_g").c_str());
            layer.ln_1_b = ml->get_tensor(layers_i + ".ln1.bias", {n_embd});
            strcpy(layer.ln_1_b->name, (str_i + ".ln_1_b").c_str());
            
            layer.attn_time_mix_k = ml->get_tensor(layers_i + ".attention.time_mix_key", {n_embd});
            strcpy(layer.attn_time_mix_k->name, (str_i + ".attn_time_mix_k").c_str());
            layer.attn_time_mix_v = ml->get_tensor(layers_i + ".attention.time_mix_value", {n_embd});
            strcpy(layer.attn_time_mix_v->name, (str_i + ".attn_time_mix_v").c_str());
            layer.attn_time_mix_r = ml->get_tensor(layers_i + ".attention.time_mix_receptance", {n_embd});
            strcpy(layer.attn_time_mix_r->name, (str_i + ".attn_time_mix_r").c_str());
            layer.attn_time_first = ml->get_tensor(layers_i + ".attention.time_first", {n_embd});
            strcpy(layer.attn_time_first->name, (str_i + ".attn_time_first").c_str());
            layer.attn_time_decay = ml->get_tensor(layers_i + ".attention.time_decay", {n_embd});
            strcpy(layer.attn_time_decay->name, (str_i + ".attn_time_decay").c_str());
            
            layer.attn_k_w = ml->get_tensor(layers_i + ".attention.key.weight", {n_embd, n_embd});
            strcpy(layer.attn_k_w->name, (str_i + ".attn_k_w").c_str());
            layer.attn_v_w = ml->get_tensor(layers_i + ".attention.value.weight", {n_embd, n_embd});
            strcpy(layer.attn_v_w->name, (str_i + ".attn_v_w").c_str());
            layer.attn_r_w = ml->get_tensor(layers_i + ".attention.receptance.weight", {n_embd, n_embd});
            strcpy(layer.attn_r_w->name, (str_i + ".attn_r_w").c_str());
            layer.attn_out_w = ml->get_tensor(layers_i + ".attention.output.weight", {n_embd, n_embd});
            strcpy(layer.attn_out_w->name, (str_i + ".attn_out_w").c_str());
            
            layer.ln_2_g = ml->get_tensor(layers_i + ".ln2.weight", {n_embd});
            strcpy(layer.ln_2_g->name, (str_i + ".ln_2_g").c_str());
            layer.ln_2_b = ml->get_tensor(layers_i + ".ln2.bias", {n_embd});
            strcpy(layer.ln_2_b->name, (str_i + ".ln_2_b").c_str());
            
            layer.ff_time_mix_k = ml->get_tensor(layers_i + ".feed_forward.time_mix_key", {n_embd});
            strcpy(layer.ff_time_mix_k->name, (str_i + ".ff_time_mix_k").c_str());
            layer.ff_time_mix_r = ml->get_tensor(layers_i + ".feed_forward.time_mix_receptance", {n_embd});
            strcpy(layer.ff_time_mix_r->name, (str_i + ".ff_time_mix_r").c_str());
            
            layer.ff_k_w = ml->get_tensor(layers_i + ".feed_forward.key.weight", {n_embd, n_embd * 4});
            strcpy(layer.ff_k_w->name, (str_i + ".ff_k_w").c_str());
            layer.ff_r_w = ml->get_tensor(layers_i + ".feed_forward.receptance.weight", {n_embd, n_embd});
            strcpy(layer.ff_r_w->name, (str_i + ".ff_r_w").c_str());
            layer.ff_v_w = ml->get_tensor(layers_i + ".feed_forward.value.weight", {n_embd * 4, n_embd});
            strcpy(layer.ff_v_w->name, (str_i + ".ff_v_w").c_str());
        }
    }

    ml->done_getting_tensors();

    // populate `tensors_by_name`
    for (rwkv_load_tensor & lt : ml->tensors_map.tensors) {
        model.tensors_by_name.emplace_back(lt.name, lt.ggml_tensor);
    }

    ml->load_all_data(progress_callback, progress_callback_user_data, use_mlock ? &lctx.model.mlock_mmap : NULL);

    model.mapping = std::move(ml->mapping);
    
    // Create empty cgraph
    //model.cg.reset(new(std::nothrow) struct ggml_cgraph());

    // loading time will be recalculate after the first eval, so
    // we take page faults deferred by mmap() into consideration
    lctx.t_load_us = ggml_time_us() - lctx.t_start_us;
}

static bool rwkv_model_load(
        const std::string & fname,
        rwkv_context & lctx,
        int n_ctx,
        ggml_type memory_type,
        bool use_mmap,
        bool use_mlock,
        bool vocab_only,
        rwkv_progress_callback progress_callback,
        void *progress_callback_user_data) {
    try {
        rwkv_model_load_internal(fname, lctx, n_ctx, memory_type, use_mmap, use_mlock,
                                  vocab_only, progress_callback, progress_callback_user_data);
        return true;
    } catch (const std::string & err) {
        fprintf(stderr, "error loading model: %s\n", err.c_str());
        return false;
    }
}

// ggml ext functions - now built-in to ggml because map_unary is having some func addressing issues when pre-building graph
/*
// max
void ggml_ext_max_impl(const int n_cols, float * dst, const float * src0, const float * src1) {
    for (int i = 0; i < n_cols; i++) { dst[i] = fmaxf(src0[i], src1[i]); }
}

struct ggml_tensor * ggml_ext_max(ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * y) {
    struct ggml_tensor * t = ggml_map_binary_f32(ctx, x, y, ggml_ext_max_impl);
    strcpy(t->name, "ext max");
    return t;
}

// exp
void ggml_ext_exp_impl(const int n_cols, float * dst, const float * src) {
    for(int i = 0; i < n_cols; i++) { dst[i] = expf(src[i]); }
}

struct ggml_tensor * ggml_ext_exp(ggml_context * ctx, struct ggml_tensor * x) {
    struct ggml_tensor * t = ggml_map_unary_f32(ctx, x, ggml_ext_exp_impl);
    strcpy(t->name, "ext exp");
    return t;
}

// 1 - x
void ggml_ext_one_minus_x_impl(const int n_cols, float * dst, const float * src) {
    for (int i = 0; i < n_cols; i++) { dst[i] = 1.0f - src[i]; }
}

struct ggml_tensor * ggml_ext_one_minus_x(ggml_context * ctx, struct ggml_tensor * x) {
    struct ggml_tensor * t = ggml_map_unary_f32(ctx, x, ggml_ext_1_minus_x_impl);
    strcpy(t->name, "ext 1 - x");
    return t;
}

// sigmoid
void ggml_ext_sigmoid_impl(const int n_cols, float * dst, const float * src) {
    //for (int i = 0; i < n_cols; i++) { dst[i] = (float)(1.0 / (1.0 + exp((double)-src[i]))); }
    for (int i = 0; i < n_cols; i++) { dst[i] = 1.0 / (1.0 + expf(-src[i])); }
}

struct ggml_tensor * ggml_ext_sigmoid(ggml_context * ctx, struct ggml_tensor * x) {
    struct ggml_tensor * t = ggml_map_unary_f32(ctx, x, ggml_ext_sigmoid_impl);
    strcpy(t->name, "ext sigmoid");
    return t;
}
 */

// time decay
/*void ggml_ext_time_decay_impl(const int n_cols, float * dst, const float * src0, const float * src1) {
    //for (int i = 0; i < n_cols; i++) { dst[i] = (float)(((double)src0[i]) - exp((double)src1[i])); }
    //for (int i = 0; i < n_cols; i++) { dst[i] = (float)(((double)src0[i]) + ((double)src1[i])); }
    for (int i = 0; i < n_cols; i++) { dst[i] = src0[i] + src1[i]; }
}

struct ggml_tensor * ggml_ext_time_decay(ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * y) {
    struct ggml_tensor * t = ggml_map_binary_f32(ctx, x, y, ggml_ext_time_decay_impl);
    strcpy(t->name, "ext time decay");
    return t;
}*/

/*static void debug_state(std::string name, struct ggml_tensor * state_t, int n_layer, int n_embd) {
    std::vector<float> state;
    state.resize(n_layer * n_embd);
    memcpy(state.data(), (float *) ggml_get_data(state_t), sizeof(float) * n_layer * n_embd);
    printf("%s: [%.8f ... %.8f]\n", name.c_str(), state.front(), state.back());
}*/

// Helpers

static inline struct ggml_tensor * rwkv_layer_norm(ggml_context * ctx, struct ggml_tensor * x, struct ggml_tensor * weight, struct ggml_tensor * bias) {
    // LayerNorm in RWKV is `x = (x - mean(x)) / sqrt(variance(x) + 1e-5) * weight + bias`
    // Looks like ggml_norm does the first part, we only need to apply weight & bias.
    return ggml_add_inplace(ctx, ggml_mul(ctx, ggml_norm(ctx, x), weight), bias);
}

// evaluate the transformer
//
//   - lctx:      rwkv context
//   - token:     new token to process
//   - n_threads: number of threads to use
//

static bool rwkv_eval_internal(struct rwkv_context & lctx,
                               const rwkv_token      token,
                               const char *          dot_path) {
    const int64_t t_start_us = ggml_time_us();

    auto & model   = lctx.model;
    auto & state = model.state;
    const auto & hparams = model.hparams;

    auto & mem_per_token = lctx.mem_per_token;
    
    ARCH_ASSERT(!!state.ctx);
    
    // Check if we need to create context and graph
    struct ggml_context * ctx = lctx.ctx;
    //struct ggml_cgraph cg = *lctx.cg;
    if (ctx == NULL) {
        const int n_embd  = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        //const int n_ctx   = hparams.n_ctx;
        //const int n_head  = hparams.n_head;
        const int n_vocab = hparams.n_vocab;
        //const int n_rot   = hparams.n_rot;
        const int rescale_every  = hparams.rescale_every;
        
        auto & buf_compute   = lctx.buf_compute;

        struct ggml_init_params params = {
            /*.mem_size   =*/ buf_compute.size,
            /*.mem_buffer =*/ buf_compute.addr,
            /*.no_alloc   =*/ false,
        };
        
        // ggml context for rwkv context
        ctx = ggml_init(params);
        lctx.ctx = ctx;
        
        // for big prompts, if BLAS is enabled, it is better to use only one thread
        // otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
        //struct ggml_cgraph cg = {};
        //lctx.cg = new ggml_cgraph();
        //cg = *lctx.cg;
        lctx.cg.n_threads = 1; //N >= 32 && ggml_cpu_has_blas() && !ggml_cpu_has_cublas() ? 1 : n_threads;
        
        // Token embedding
        //printf("eval token in %d %s\n", token, rwkv_token_to_str(&lctx, token));
        //struct ggml_tensor * id = ggml_new_i32(ctx, token);
        //strcpy(id->name, "id");
        
        struct ggml_tensor * inpL = ggml_get_rows(ctx, model.wte, state.token); //id);
        //strcpy(inpL->name, "inpL");
        
        // MARK: layer norm pre
        // inpL = ln_pre_g * norm(inpL) + ln_pre_b
        inpL = rwkv_layer_norm(ctx, inpL, model.ln_pre_g, model.ln_pre_b);
        //strcpy(inpL->name, "ln_pre");
        
        //printf("eval layers\n");

        for (int i = 0; i < n_layer; i++) {
            //struct rwkv_layer & layer = model.layers[i];
            struct rwkv_layer & layer = model.layers[i];
            
            struct rwkv_layer_state input_state  = state.input_layers[i];
            struct rwkv_layer_state & output_state = state.output_layers[i];
            
            lctx.use_buf(ctx, 0);
            
            struct ggml_tensor * inpAttn = inpL;

            // MARK: layer norm 1
            // cur = ln_1_g * norm(inpL) + ln_1_b
            struct ggml_tensor * cur = rwkv_layer_norm(ctx, inpAttn, layer.ln_1_g, layer.ln_1_b);
            //strcpy(cur->name, "ln1");

            // MARK: attention (time mixing)
            {
                // Mix current token embedding with the previous timestep to produce key, value, receptance
                // xx = state[5*i+1]
                struct ggml_tensor *& x_prev = input_state.attn;
                // xr = x * time_mix_r + xx * (1 - time_mix_r) "x_rec"
                struct ggml_tensor * x_rec = ggml_add_inplace(ctx,
                                                ggml_mul(ctx, cur, layer.attn_time_mix_r),
                                                ggml_mul(ctx, x_prev, ggml_ext_one_minus_x(ctx, layer.attn_time_mix_r)) );
                // xk = x * time_mix_k + xx * (1 - time_mix_k) "x_key"
                struct ggml_tensor * x_key = ggml_add_inplace(ctx,
                                                ggml_mul(ctx, cur, layer.attn_time_mix_k),
                                                ggml_mul(ctx, x_prev, ggml_ext_one_minus_x(ctx, layer.attn_time_mix_k)) );
                // xv = x * time_mix_v + xx * (1 - time_mix_v) "x_val"
                struct ggml_tensor * x_val = ggml_add_inplace(ctx,
                                                ggml_mul(ctx, cur, layer.attn_time_mix_v),
                                                ggml_mul(ctx, x_prev, ggml_ext_one_minus_x(ctx, layer.attn_time_mix_v)) );
                // state[5*i+1] = x.float()
                x_prev = cur;
                
                // ne0[4096, 4096] mm ne1[4096, 1] = [4096, 1] (ne0[1], ne1[1])
                // r = torch.sigmoid(xr @ rw)
                struct ggml_tensor * r = ggml_ext_sigmoid(ctx,
                                            ggml_mul_mat(ctx, layer.attn_r_w, x_rec) );
                // k = (xk @ kw).float()
                struct ggml_tensor * k = ggml_mul_mat(ctx, layer.attn_k_w, x_key);
                // v = (xv @ vw).float()
                struct ggml_tensor * v = ggml_mul_mat(ctx, layer.attn_v_w, x_val);
                
                // Linear attention
                // Using prev_num and prev_den may be unstable, so a special num and den state are used along with max state
                // aa = state[5*i+2] "num_state"
                struct ggml_tensor *& num_state = input_state.num; //ggml_view_1d(ctx, state.num, n_embd, i * n_embd * ggml_element_size(state.num));
                // bb = state[5*i+3] "den_state"
                struct ggml_tensor *& den_state = input_state.den; //ggml_view_1d(ctx, state.den, n_embd, i * n_embd * ggml_element_size(state.den));
                // pp = state[5*i+4] "max_state"
                struct ggml_tensor *& max_state = input_state.max; //ggml_view_1d(ctx, state.max, n_embd, i * n_embd * ggml_element_size(state.max));
                
                // ww = time_first + k "k_time_first"
                struct ggml_tensor * k_time_first = ggml_add(ctx, layer.attn_time_first, k);
                // p = torch.maximum(pp, ww) "max_state_k_time_first"
                struct ggml_tensor * max_state_k_time_first = ggml_ext_max(ctx, max_state, k_time_first);
                // e1 = torch.exp(pp - p) "exp_prev"
                struct ggml_tensor * exp_prev = ggml_ext_exp(ctx,
                                                    ggml_sub(ctx, max_state, max_state_k_time_first) );
                // e2 = torch.exp(ww - p) "exp_cur"
                struct ggml_tensor * exp_cur = ggml_ext_exp(ctx,
                                                    ggml_sub(ctx, k_time_first, max_state_k_time_first) );
                // a = e1 * aa + e2 * v "num" = (e^-w * prev_num + e^x_key * x_val)
                struct ggml_tensor * num = ggml_add_inplace(ctx,
                                                ggml_mul(ctx, exp_prev, num_state),
                                                ggml_mul(ctx, exp_cur, v) );
                // b = e1 * bb + e2 "den" = (e^-w * prev_den + e^x_key)
                struct ggml_tensor * den = ggml_add_inplace(ctx,
                                                ggml_mul(ctx, exp_prev, den_state),
                                                exp_cur);
                // wkv = (a / b)
                struct ggml_tensor * wkv = ggml_div(ctx, num, den);
                
                // ww = pp + time_decay, real time_decay = -exp(time_decay)
                //struct ggml_tensor * time_decay = ggml_neg(ctx, ggml_ext_exp(ctx, layer.attn_time_decay)); // layer.attn_time_decay
                struct ggml_tensor * max_state_time_decay = ggml_add(ctx, max_state, layer.attn_time_decay);
                //struct ggml_tensor * max_state_time_decay = ggml_ext_time_decay(ctx, max_state, layer.attn_time_decay);
                // p = torch.maximum(ww, k) "new_max_state"
                struct ggml_tensor * new_max_state = ggml_ext_max(ctx, max_state_time_decay, k);
                // e1 = torch.exp(ww - p) "exp_prev"
                exp_prev = ggml_ext_exp(ctx, ggml_sub(ctx, max_state_time_decay, new_max_state) );
                // e2 = torch.exp(k - p) "exp_cur"
                exp_cur = ggml_ext_exp(ctx, ggml_sub(ctx, k, new_max_state) );
                
                // state[5*i+2] = e1 * aa + e2 * v "num_state"
                struct ggml_tensor * new_num_state = ggml_add_inplace(ctx,
                                                        ggml_mul(ctx, exp_prev, num_state),
                                                        ggml_mul(ctx, exp_cur, v) );
                num_state = new_num_state;
                //num_state = ggml_cpy(ctx, new_num_state, num_state);
                //strcpy(num_state->name, "num state");
                // state[5*i+3] = e1 * bb + e2 "den_state"
                struct ggml_tensor * new_den_state = ggml_add_inplace(ctx,
                                                        ggml_mul(ctx, exp_prev, den_state),
                                                        exp_cur);
                den_state = new_den_state;
                //den_state = ggml_cpy(ctx, new_den_state, den_state);
                //strcpy(den_state->name, "den state");
                
                // state[5*i+4] = p "max_state"
                max_state = new_max_state;
                //max_state = ggml_cpy(ctx, new_max_state, max_state);
                //strcpy(max_state->name, "max state");
                
                // return (r * wkv) @ ow
                struct ggml_tensor * rwkv = ggml_mul(ctx, r, wkv);
                //strcpy(rwkv->name, "rwkv");
                cur = ggml_mul_mat(ctx, layer.attn_out_w, rwkv);
            }
            
            struct ggml_tensor * outAttn = cur;
            //strcpy(outAttn->name, "outAttn");
            
            // MARK: residual 1
            // x = x + attn(ln1(x))
            struct ggml_tensor * inpFF = ggml_add_inplace(ctx, outAttn, inpAttn);
            //strcpy(inpFF->name, "inpFF");

            lctx.use_buf(ctx, 1);
            
            // MARK: layer norm 2
            // cur = ln_2_g * norm(inpFF) + ln_2_b
            cur = rwkv_layer_norm(ctx, inpFF, layer.ln_2_g, layer.ln_2_b);
            //strcpy(cur->name, "ln2");
            
            // MARK: feed-forward network (channel mixing)
            {
                // xx = state[5*i+0] "x_prev"
                struct ggml_tensor *& x_prev = input_state.ff;
                // xr = x * time_mix_r + xx * (1 - time_mix_r) "x_rec"
                struct ggml_tensor * x_rec = ggml_add_inplace(ctx,
                                                ggml_mul(ctx, cur, layer.ff_time_mix_r),
                                                ggml_mul(ctx, x_prev, ggml_ext_one_minus_x(ctx, layer.ff_time_mix_r)) );
                // xk = x * time_mix_k + xx * (1 - time_mix_k) "x_key"
                struct ggml_tensor * x_key = ggml_add_inplace(ctx,
                                                ggml_mul(ctx, cur, layer.ff_time_mix_k),
                                                ggml_mul(ctx, x_prev, ggml_ext_one_minus_x(ctx, layer.ff_time_mix_k)) );
                // state[5*i+0] = x.float()
                x_prev = cur;
                
                // r = torch.sigmoid(xr @ rw), sigmoid used as "forget gate"
                struct ggml_tensor * r = ggml_ext_sigmoid(ctx,
                                            ggml_mul_mat(ctx, layer.ff_r_w, x_rec) );
                //strcpy(r->name, "r");
                // k = torch.square(torch.relu(xk @ kw)), relu is same as max(0, x)
                struct ggml_tensor * k = ggml_sqr(ctx,
                                            ggml_relu(ctx,
                                                ggml_mul_mat(ctx, layer.ff_k_w, x_key) ) );
                //strcpy(k->name, "k");
                // kv = k @ vw
                struct ggml_tensor * kv = ggml_mul_mat(ctx, layer.ff_v_w, k);
                //strcpy(kv->name, "kv");
                
                // return r * kv
                cur = ggml_mul(ctx, r, kv);
            }
            
            struct ggml_tensor * outFF = cur;
            //strcpy(outFF->name, "outFF");
            
            // MARK: residual 2
            // x = x + ff(ln2(x))
            inpL = ggml_add_inplace(ctx, outFF, inpFF);
            //strcpy(inpL->name, "inpL");
            
            // rescale_every, needed???
            //if (
            //    self.layers_are_rescaled
            //    and self.config.rescale_every > 0
            //    and (idx + 1) % self.config.rescale_every == 0
            //):
            //hidden_states = hidden_states / 2
            if(rescale_every > 0) {
                if(((i + 1) % rescale_every) == 0) {
                    //inpL = ggml_div(ctx, inpL, ggml_repeat(ctx, ggml_new_f32(ctx, 2), inpL));
                    inpL = ggml_scale(ctx, inpL, ggml_new_f32(ctx, 0.5f) );
                }
            }
            
            // Update states
            ggml_build_forward_expand(&lctx.cg, ggml_cpy(ctx, input_state.attn, output_state.attn));
            ggml_build_forward_expand(&lctx.cg, ggml_cpy(ctx, input_state.num,  output_state.num));
            ggml_build_forward_expand(&lctx.cg, ggml_cpy(ctx, input_state.den,  output_state.den));
            ggml_build_forward_expand(&lctx.cg, ggml_cpy(ctx, input_state.max,  output_state.max));
            ggml_build_forward_expand(&lctx.cg, ggml_cpy(ctx, input_state.ff,   output_state.ff));
        }
        
        //printf("eval output\n");

        lctx.use_buf(ctx, 0);

        // used at the end to optionally extract the embeddings
        struct ggml_tensor * embeddings = NULL;

        // layer norm out
        // x = self.layer_norm(x, self.w.ln_out)
        inpL = rwkv_layer_norm(ctx, inpL, model.ln_out_g, model.ln_out_b);
        //embeddings = inpL;
        //strcpy(inpL->name, "ln_out");

        //printf("eval lm head\n");
        
        // lm_head
        // x = self.w.head.weight @ x
        inpL = ggml_mul_mat(ctx, model.lmh_w, inpL);
        //strcpy(inpL->name, "lmh");

        lctx.use_buf(ctx, -1);

        // logits -> probs
        //inpL = ggml_soft_max(ctx, inpL);

        // Build head
        //ggml_build_forward_expand(&cg, inpL);
        ggml_build_forward_expand(&lctx.cg, ggml_cpy(ctx, inpL, state.logits));
    }

    // Debug dump dot digraph of model
    //if(dot_path != NULL) ggml_graph_dump_dot(&cg, NULL, dot_path);
    
    // Copy previous output_state to input_state
    memcpy(state.input_state->data, state.output_state->data, ggml_nbytes(state.input_state));
    
    // Set current token
    ggml_set_i32(state.token, token);
    
    // Compute the graph
    ggml_graph_compute(ctx, &lctx.cg);

//#define GGML_PERF
#ifdef GGML_PERF
    // print timing information per ggml operation (for debugging purposes)
    // requires GGML_PERF to be defined
    ggml_graph_print(&lctx.cg);
#endif

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // extract logits
    /*
    {
        auto & logits = lctx.logits;
        //logits.resize(n_vocab);
        //printf("inpL ne %lld, %lld, %lld, %lld\n", inpL->ne[0], inpL->ne[1], inpL->ne[2], inpL->ne[3]);
        memcpy(logits.data(), state.logits->data, ggml_nbytes(state.logits));
        //memcpy(logits.data(), (float *) ggml_get_data(inpL), sizeof(float)*n_vocab);
    }
     */

    // extract embeddings
    /*if (lctx.embedding.size()) {
        auto & embedding = lctx.embedding;
        //embedding.resize(n_embd);
        memcpy(embedding.data(), (float *) ggml_get_data(embeddings), sizeof(float) * n_embd);
    }*/

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx);
    }

#if 0
    printf("\n%s: used_mem = %.3f MiB, scratch -- %.3f MiB %.3f MiB\n", __func__,
            ggml_used_mem(ctx)/1024.0/1024.0,
            lctx.get_buf_max_mem(0)/1024.0/1024.0,
            lctx.get_buf_max_mem(1)/1024.0/1024.0);
#endif
    
    //ggml_free(ctx);

    // measure the performance only for the single-token evals
    lctx.t_eval_us += ggml_time_us() - t_start_us;
    lctx.n_eval++;
    
    //printf("eval complete %d\n", lctx.n_eval);

    return true;
}

//
// tokenizer
//

static size_t utf8_len(char src) {
    const size_t lookup[] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4 };
    uint8_t highbits = static_cast<uint8_t>(src) >> 4;
    return lookup[highbits];
}

struct rwkv_sp_symbol {
    using index = int;
    index prev;
    index next;
    const char * text;
    size_t n;
};

struct rwkv_sp_bigram {
    struct comparator {
        bool operator()(rwkv_sp_bigram & l, rwkv_sp_bigram & r) {
            return (l.left > r.left);
            //return (l.score < r.score) || (l.score == r.score && l.left > r.left);
        }
    };
    using queue_storage = std::vector<rwkv_sp_bigram>;
    using queue = std::priority_queue<rwkv_sp_bigram, queue_storage, comparator>;
    rwkv_sp_symbol::index left;
    rwkv_sp_symbol::index right;
    //float score;
    size_t size;
};

// original implementation:
// https://github.com/ggerganov/llama.cpp/commit/074bea2eb1f1349a0118239c4152914aecaa1be4
struct rwkv_tokenizer {
    rwkv_tokenizer(const rwkv_vocab & vocab): vocab_(vocab) {}

    void tokenize(const std::string & text, std::vector<rwkv_vocab::id> & output) {
        // split string into utf8 chars
        int index = 0;
        size_t offs = 0;
        while (offs < text.size()) {
            rwkv_sp_symbol sym;
            size_t char_len = std::min(text.size() - offs, utf8_len(text[offs]));
            sym.text = text.c_str() + offs;
            sym.n = char_len;
            offs += char_len;
            sym.prev = index - 1;
            sym.next = offs == text.size() ? -1 : index + 1;
            index++;
            symbols_.emplace_back(std::move(sym));
        }

        // seed the work queue with all possible 2-character tokens.
        for (size_t i = 1; i < symbols_.size(); ++i) {
            try_add_bigram(i - 1, i);
        }

        // keep substituting the highest frequency pairs for as long as we can.
        while (!work_queue_.empty()) {
            auto bigram = work_queue_.top();
            work_queue_.pop();

            auto & left_sym = symbols_[bigram.left];
            auto & right_sym = symbols_[bigram.right];

            // if one of the symbols already got merged, skip it.
            if (left_sym.n == 0 || right_sym.n == 0 ||
                left_sym.n + right_sym.n != bigram.size) {
                continue;
            }

            // merge the right sym into the left one
            left_sym.n += right_sym.n;
            right_sym.n = 0;

            //printf("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

            // remove the right sym from the chain
            left_sym.next = right_sym.next;
            if (right_sym.next >= 0) {
                symbols_[right_sym.next].prev = bigram.left;
            }

            // find more substitutions
            try_add_bigram(left_sym.prev, bigram.left);
            try_add_bigram(bigram.left, left_sym.next);
        }

        for (int i = 0; i != -1; i = symbols_[i].next) {
            auto & symbol = symbols_[i];
            auto token = vocab_.token_to_id.find(std::string(symbol.text, symbol.n));

            if (token == vocab_.token_to_id.end()) {
                // output any symbols that did not form tokens as bytes.
                for (int j = 0; j < (int) symbol.n; ++j) {
                    rwkv_vocab::id token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
                    output.push_back(token_id);
                }
            } else {
                output.push_back((*token).second);
            }
        }
    }

private:
    void try_add_bigram(int left, int right) {
        if (left == -1 || right == -1) {
            return;
        }

        const std::string text = std::string(symbols_[left].text, symbols_[left].n + symbols_[right].n);
        auto token = vocab_.token_to_id.find(text);

        if (token == vocab_.token_to_id.end()) {
            return;
        }

        if (static_cast<size_t>((*token).second) >= vocab_.id_to_token.size()) {
            return;
        }

        const auto &tok_score = vocab_.id_to_token[(*token).second];

        rwkv_sp_bigram bigram;
        bigram.left = left;
        bigram.right = right;
        //bigram.score = tok_score.score;
        bigram.size = text.size();
        work_queue_.push(bigram);
    }

    const rwkv_vocab & vocab_;
    std::vector<rwkv_sp_symbol> symbols_;
    rwkv_sp_bigram::queue work_queue_;
};

/*
 // From ggml, does not handle \n\n properly
 struct rwkv_tokenizer {
    rwkv_tokenizer(const rwkv_vocab & vocab): vocab_(vocab) {}
    
    void tokenize(const std::string & text, std::vector<rwkv_vocab::id> & output) {
    //std::vector<gpt_vocab::id> gpt_tokenize(const gpt_vocab & vocab, const std::string & text) {
        std::vector<std::string> words;
        
        // first split the text into words
        {
            std::string str = text;
            
            // Generate the subpattern from the special_tokens vector if it's not empty
            //
            //if (!vocab.special_tokens.empty()) {
            //    const std::regex escape(R"([\[\\\^\$\.\|\?\*\+\(\)\{\}])");
            //    std::string special_tokens_subpattern;
            //    for (const auto & token : vocab.special_tokens) {
            //        if (!special_tokens_subpattern.empty()) {
            //            special_tokens_subpattern += "|";
            //        }
            //        special_tokens_subpattern += std::regex_replace(token, escape, R"(\$&)");
            //    }
            //
            //    std::regex re(special_tokens_subpattern);
            //    std::smatch m;
            //    // Split the text by special tokens.
            //    while (std::regex_search(str, m, re)) {
            //        // Split the substrings in-between special tokens into words.
            //        split_words(m.prefix(), words);
            //        // Add matched special tokens as words.
            //        for (auto x : m) {
            //            words.push_back(x);
            //        }
            //        str = m.suffix();
            //    }
            //    // Remaining text without special tokens will be handled below.
            //}
            split_words(str, words);
        }
        
        // find the longest token that forms each word in words:
        //std::vector<rwkv_vocab::id> tokens;
        for (const auto & word : words) {
            for (int i = 0; i < word.size(); ){
                for (int j = word.size() - 1; j >= i; j--){
                    auto cand = word.substr(i, j-i+1);
                    auto it = vocab_.token_to_id.find(cand);
                    if (it != vocab_.token_to_id.end()){ // word.substr(i, j-i+1) in vocab
                        output.push_back(it->second);
                        i = j + 1;
                        break;
                    }
                    else if (j == i){ // word.substr(i, 1) has no matching
                        fprintf(stderr, "%s: unknown token '%s'\n", __func__, word.substr(i, 1).data());
                        i++;
                    }
                }
            }
        }
        
        //return tokens;
    }
    
private:
    void split_words(std::string str, std::vector<std::string>& words) {
        const std::string pattern = R"('s|'t|'re|'ve|'m|'ll|'d| ?[[:alpha:]]+| ?[[:digit:]]+| ?[^\s[:alpha:][:digit:]]+|\s+(?!\S)|\s+)";
        const std::regex re(pattern);
        std::smatch m;

        while (std::regex_search(str, m, re)) {
            for (auto x : m) {
                words.push_back(x);
            }
            str = m.suffix();
        }
    }
    
    const rwkv_vocab & vocab_;
};
 */

static std::vector<rwkv_vocab::id> rwkv_tokenize(const rwkv_vocab & vocab, const std::string & text, bool bos) {
    rwkv_tokenizer tokenizer(vocab);
    std::vector<rwkv_vocab::id> output;

    if (text.size() == 0) {
        return output;
    }

    if (bos) {
        output.push_back(rwkv_token_bos());
    }

    tokenizer.tokenize(text, output);
    return output;
}

//
// sampling
//

void rwkv_sample_softmax(struct rwkv_context * ctx, rwkv_token_data_array * candidates) {
    assert(candidates->size > 0);

    const int64_t t_start_sample_us = ggml_time_us();

    // Sort the logits in descending order
    if (!candidates->sorted) {
        std::sort(candidates->data, candidates->data + candidates->size, [](const rwkv_token_data & a, const rwkv_token_data & b) {
            return a.logit > b.logit;
        });
        candidates->sorted = true;
    }

    float max_l = candidates->data[0].logit;
    float cum_sum = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        float p = expf(candidates->data[i].logit - max_l);
        candidates->data[i].p = p;
        cum_sum += p;
    }
    for (size_t i = 0; i < candidates->size; ++i) {
        candidates->data[i].p /= cum_sum;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void rwkv_sample_top_k(struct rwkv_context * ctx, rwkv_token_data_array * candidates, int k, size_t min_keep) {
    const int64_t t_start_sample_us = ggml_time_us();

    k = std::max(k, (int) min_keep);
    k = std::min(k, (int) candidates->size);

    // Sort scores in descending order
    if (!candidates->sorted) {
        auto comp = [](const rwkv_token_data & a, const rwkv_token_data & b) {
            return a.logit > b.logit;
        };
        if (k == (int) candidates->size) {
            std::sort(candidates->data, candidates->data + candidates->size, comp);
        } else {
            std::partial_sort(candidates->data, candidates->data + k, candidates->data + candidates->size, comp);
        }
        candidates->sorted = true;
    }
    candidates->size = k;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void rwkv_sample_top_p(struct rwkv_context * ctx, rwkv_token_data_array * candidates, float p, size_t min_keep) {
    if (p >= 1.0f) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    rwkv_sample_softmax(ctx, candidates);

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;

    for (size_t i = 0; i < candidates->size; ++i) {
        cum_sum += candidates->data[i].p;

        // Check if the running sum is greater than p or if we have kept at least min_keep tokens
        if (cum_sum > p && i >= min_keep) {
            last_idx = i;
            break;
        }
    }

    // Resize the output vector to keep only the top-p tokens
    candidates->size = last_idx;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void rwkv_sample_tail_free(struct rwkv_context * ctx, rwkv_token_data_array * candidates, float z, size_t min_keep) {
    if (z >= 1.0f || candidates->size <= 2) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    rwkv_sample_softmax(nullptr, candidates);

    // Compute the first and second derivatives
    std::vector<float> first_derivatives(candidates->size - 1);
    std::vector<float> second_derivatives(candidates->size - 2);

    for (size_t i = 0; i < first_derivatives.size(); ++i) {
        first_derivatives[i] = candidates->data[i].p - candidates->data[i + 1].p;
    }
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = first_derivatives[i] - first_derivatives[i + 1];
    }

    // Calculate absolute value of second derivatives
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        second_derivatives[i] = abs(second_derivatives[i]);
    }

    // Normalize the second derivatives
    float second_derivatives_sum = std::accumulate(second_derivatives.begin(), second_derivatives.end(), 0.0f);
    for (float & value : second_derivatives) {
        value /= second_derivatives_sum;
    }

    float cum_sum = 0.0f;
    size_t last_idx = candidates->size;
    for (size_t i = 0; i < second_derivatives.size(); ++i) {
        cum_sum += second_derivatives[i];

        // Check if the running sum is greater than z or if we have kept at least min_keep tokens
        if (cum_sum > z && i >= min_keep) {
            last_idx = i;
            break;
        }
    }

    // Resize the output vector to keep only the tokens above the tail location
    candidates->size = last_idx;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}


void rwkv_sample_typical(struct rwkv_context * ctx, rwkv_token_data_array * candidates, float p, size_t min_keep) {
    // Reference implementation:
    // https://github.com/huggingface/transformers/compare/main...cimeister:typical-sampling:typical-pr
    if (p >= 1.0f) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    // Compute the softmax of logits and calculate entropy
    rwkv_sample_softmax(nullptr, candidates);

    float entropy = 0.0f;
    for (size_t i = 0; i < candidates->size; ++i) {
        entropy += -candidates->data[i].p * logf(candidates->data[i].p);
    }

    // Compute the absolute difference between negative log probability and entropy for each candidate
    std::vector<float> shifted_scores;
    for (size_t i = 0; i < candidates->size; ++i) {
        float shifted_score = fabsf(-logf(candidates->data[i].p) - entropy);
        shifted_scores.push_back(shifted_score);
    }

    // Sort tokens based on the shifted_scores and their corresponding indices
    std::vector<size_t> indices(candidates->size);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(), [&](size_t a, size_t b) {
        return shifted_scores[a] < shifted_scores[b];
    });

    // Compute the cumulative probabilities
    float cum_sum = 0.0f;
    size_t last_idx = indices.size();

    for (size_t i = 0; i < indices.size(); ++i) {
        size_t idx = indices[i];
        cum_sum += candidates->data[idx].p;

        // Check if the running sum is greater than typical or if we have kept at least min_keep tokens
        if (cum_sum > p && i >= min_keep - 1) {
            last_idx = i + 1;
            break;
        }
    }

    // Resize the output vector to keep only the locally typical tokens
    std::vector<rwkv_token_data> new_candidates;
    for (size_t i = 0; i < last_idx; ++i) {
        size_t idx = indices[i];
        new_candidates.push_back(candidates->data[idx]);
    }

    // Replace the data in candidates with the new_candidates data
    std::copy(new_candidates.begin(), new_candidates.end(), candidates->data);
    candidates->size = new_candidates.size();

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void rwkv_sample_temperature(struct rwkv_context * ctx, rwkv_token_data_array * candidates_p, float temp) {
    const int64_t t_start_sample_us = ggml_time_us();

    for (size_t i = 0; i < candidates_p->size; ++i) {
        candidates_p->data[i].logit /= temp;
    }

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void rwkv_sample_repetition_penalty(struct rwkv_context * ctx, rwkv_token_data_array * candidates, rwkv_token * last_tokens, size_t last_tokens_size, float penalty) {
    if (last_tokens_size == 0 || penalty == 1.0f) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    for (size_t i = 0; i < candidates->size; ++i) {
        auto token_iter = std::find(last_tokens, last_tokens + last_tokens_size, candidates->data[i].id);
        if (token_iter == last_tokens + last_tokens_size) {
            continue;
        }

        // The academic publication that described this technique actually just only divided, but that would cause tokens with negative logits to become more likely, which is obviously wrong.
        // This is common fix for this problem, which is to multiply by the penalty instead of dividing.
        if (candidates->data[i].logit <= 0) {
            candidates->data[i].logit *= penalty;
        } else {
            candidates->data[i].logit /= penalty;
        }
    }

    candidates->sorted = false;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}

void rwkv_sample_frequency_and_presence_penalties(struct rwkv_context * ctx, rwkv_token_data_array * candidates, rwkv_token * last_tokens_p, size_t last_tokens_size, float alpha_frequency, float alpha_presence) {
    if (last_tokens_size == 0 || (alpha_frequency == 0.0f && alpha_presence == 0.0f)) {
        return;
    }

    const int64_t t_start_sample_us = ggml_time_us();

    // Create a frequency map to count occurrences of each token in last_tokens
    std::unordered_map<rwkv_token, int> token_count;
    for (size_t i = 0; i < last_tokens_size; ++i) {
        token_count[last_tokens_p[i]]++;
    }

    // Apply frequency and presence penalties to the candidates
    for (size_t i = 0; i < candidates->size; ++i) {
        auto token_iter = token_count.find(candidates->data[i].id);
        if (token_iter == token_count.end()) {
            continue;
        }

        int count = token_iter->second;
        candidates->data[i].logit -= float(count) * alpha_frequency + float(count > 0) * alpha_presence;
    }

    candidates->sorted = false;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
}


rwkv_token rwkv_sample_token_mirostat(struct rwkv_context * ctx, rwkv_token_data_array * candidates, float tau, float eta, int m, float * mu) {
    assert(ctx);
    auto N = float(rwkv_n_vocab(ctx));
    int64_t t_start_sample_us;
    t_start_sample_us = ggml_time_us();

    rwkv_sample_softmax(nullptr, candidates);

    // Estimate s_hat using the most probable m tokens
    float s_hat = 0.0;
    float sum_ti_bi = 0.0;
    float sum_ti_sq = 0.0;
    for (size_t i = 0; i < size_t(m - 1) && i < candidates->size - 1; ++i) {
        float t_i = logf(float(i + 2) / float(i + 1));
        float b_i = logf(candidates->data[i].p / candidates->data[i + 1].p);
        sum_ti_bi += t_i * b_i;
        sum_ti_sq += t_i * t_i;
    }
    s_hat = sum_ti_bi / sum_ti_sq;

    // Compute k from the estimated s_hat and target surprise value
    float epsilon_hat = s_hat - 1;
    float k = powf((epsilon_hat * powf(2, *mu)) / (1 - powf(N, -epsilon_hat)), 1 / s_hat);

    // Sample the next word X using top-k sampling
    rwkv_sample_top_k(nullptr, candidates, int(k), 1);
    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    rwkv_token X = rwkv_sample_token(ctx, candidates);
    t_start_sample_us = ggml_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const rwkv_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
        ctx->n_sample++;
    }
    return X;
}

rwkv_token rwkv_sample_token_mirostat_v2(struct rwkv_context * ctx, rwkv_token_data_array * candidates, float tau, float eta, float * mu) {
    assert(ctx);
    int64_t t_start_sample_us;
    t_start_sample_us = ggml_time_us();

    rwkv_sample_softmax(ctx, candidates);

    // Truncate the words with surprise values greater than mu
    candidates->size = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const rwkv_token_data & candidate) {
        return -log2f(candidate.p) > *mu;
    }));

    // Normalize the probabilities of the remaining words
    rwkv_sample_softmax(ctx, candidates);

    // Sample the next word X from the remaining words
    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    rwkv_token X = rwkv_sample_token(ctx, candidates);
    t_start_sample_us = ggml_time_us();

    // Compute error as the difference between observed surprise and target surprise value
    size_t X_idx = std::distance(candidates->data, std::find_if(candidates->data, candidates->data + candidates->size, [&](const rwkv_token_data & candidate) {
        return candidate.id == X;
    }));
    float observed_surprise = -log2f(candidates->data[X_idx].p);
    float e = observed_surprise - tau;

    // Update mu using the learning rate and error
    *mu = *mu - eta * e;

    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    }
    return X;
}

rwkv_token rwkv_sample_token_greedy(struct rwkv_context * ctx, rwkv_token_data_array * candidates) {
    const int64_t t_start_sample_us = ggml_time_us();

    // Find max element
    auto max_iter = std::max_element(candidates->data, candidates->data + candidates->size, [](const rwkv_token_data & a, const rwkv_token_data & b) {
        return a.logit < b.logit;
    });

    rwkv_token result = max_iter->id;
    if (ctx) {
        ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
        ctx->n_sample++;
    }
    return result;
}

rwkv_token rwkv_sample_token(struct rwkv_context * ctx, rwkv_token_data_array * candidates) {
    assert(ctx);
    const int64_t t_start_sample_us = ggml_time_us();
    rwkv_sample_softmax(nullptr, candidates);

    std::vector<float> probs;
    probs.reserve(candidates->size);
    for (size_t i = 0; i < candidates->size; ++i) {
        probs.push_back(candidates->data[i].p);
    }

    std::discrete_distribution<> dist(probs.begin(), probs.end());
    auto & rng = ctx->rng;
    int idx = dist(rng);

    rwkv_token result = candidates->data[idx].id;

    ctx->t_sample_us += ggml_time_us() - t_start_sample_us;
    ctx->n_sample++;
    return result;
}

//
// updating
//

static void rwkv_model_update_internal(const std::string & fname_inp, const std::string & fname_out) {
    std::unique_ptr<rwkv_model_loader> model_loader(new rwkv_model_loader(fname_inp.c_str(),
                                                            /*use_mmap*/ false,
                                                            /*vocab_only*/ false));
    // Simply use the ftype of the first file
    auto ftype = model_loader->file_loaders[0]->hparams.ftype;
    rwkv_file_saver file_saver(fname_out.c_str(), model_loader->file_loaders.at(0).get(), ftype);

    size_t idx = 0;
    for (rwkv_load_tensor & tensor : model_loader->tensors_map.tensors) {
        arch_util_buffer read_data;
        read_data.resize(tensor.size);
        tensor.data = read_data.addr;
        model_loader->load_data_for(tensor);

        printf("[%4zu/%4zu] %48s - %14s, type = %4s, ",
               ++idx, model_loader->tensors_map.tensors.size(),
               tensor.name.c_str(), rwkv_format_tensor_shape(tensor.ne).c_str(),
               ggml_type_name(tensor.type));

        file_saver.write_tensor(tensor, tensor.type, tensor.data, tensor.size);
    }
}

int rwkv_model_update(
        const char * fname_inp,
        const char * fname_out) {
    try {
        rwkv_model_update_internal(fname_inp, fname_out);
        return 0;
    } catch (const std::string & err) {
        fprintf(stderr, "%s: failed to copy: %s\n", __func__, err.c_str());
        return 1;
    }
}

//
// quantization
//

static void rwkv_model_quantize_internal(const std::string & fname_inp, const std::string & fname_out, enum ggml_ftype ftype, int nthread) {
    ggml_type quantized_type;
    switch (ftype) {
        case GGML_FTYPE_MOSTLY_Q4_0: quantized_type = GGML_TYPE_Q4_0; break;
        case GGML_FTYPE_MOSTLY_Q4_1: quantized_type = GGML_TYPE_Q4_1; break;
        case GGML_FTYPE_MOSTLY_Q4_2: quantized_type = GGML_TYPE_Q4_2; break;
        case GGML_FTYPE_MOSTLY_Q5_0: quantized_type = GGML_TYPE_Q5_0; break;
        case GGML_FTYPE_MOSTLY_Q5_1: quantized_type = GGML_TYPE_Q5_1; break;
        case GGML_FTYPE_MOSTLY_Q8_0: quantized_type = GGML_TYPE_Q8_0; break;
        default: throw format("invalid output file type %d\n", ftype);
    };

    if (nthread <= 0) {
        nthread = std::thread::hardware_concurrency();
    }

    std::unique_ptr<rwkv_model_loader> model_loader(new rwkv_model_loader(fname_inp.c_str(), /*use_mmap*/ false,
                                                                            /*vocab_only*/ false));
    rwkv_file_saver file_saver(fname_out.c_str(), model_loader->file_loaders.at(0).get(), ftype);

    size_t total_size_org = 0;
    size_t total_size_new = 0;
    std::vector<int64_t> hist_all(1 << 4, 0);

    std::vector<std::thread> workers;
    std::mutex mutex;

    size_t idx = 0;
    for (rwkv_load_tensor & tensor : model_loader->tensors_map.tensors) {
        arch_util_buffer read_data;
        read_data.resize(tensor.size);
        tensor.data = read_data.addr;
        model_loader->load_data_for(tensor);

        printf("[%4zu/%4zu] %36s - %16s, type = %6s, ",
               ++idx, model_loader->tensors_map.tensors.size(),
               tensor.name.c_str(), rwkv_format_tensor_shape(tensor.ne).c_str(),
               ggml_type_name(tensor.type));

        // This used to be a regex, but <regex> has an extreme cost to compile times.
        bool quantize = tensor.name.rfind("weight") == tensor.name.size() - 6; // ends with 'weight'?

        // quantize only 2D tensors
        quantize &= (tensor.ne.size() == 2);

        // uncomment this to keep the output layer in FP16
        //if (tensor.name == "output.weight") {
        //    quantize = false;
        //}

        enum ggml_type new_type;
        void * new_data;
        size_t new_size;
        arch_util_buffer work;

        if (!quantize) {
            new_type = tensor.type;
            new_data = tensor.data;
            new_size = tensor.size;
            printf("size = %8.3f MiB\n", tensor.size/1024.0/1024.0);
        } else {
            new_type = quantized_type;
            float * f32_data;
            size_t nelements = tensor.ne.at(0) * tensor.ne.at(1);
            arch_util_buffer f32_conv_buf;
            if (tensor.type == GGML_TYPE_F32) {
                f32_data = (float *) tensor.data;
            } else if (tensor.type == GGML_TYPE_F16) {
                f32_conv_buf.resize(nelements * sizeof(float));
                f32_data = (float *) f32_conv_buf.addr;
                auto f16_data = (const ggml_fp16_t *) tensor.data;
                for (size_t i = 0; i < nelements; i++) {
                    f32_data[i] = ggml_fp16_to_fp32(f16_data[i]);
                }
            } else {
                throw format("type %s unsupported for integer quantization", ggml_type_name(tensor.type));
            }

            printf("quantizing .. ");
            fflush(stdout);

            work.resize(nelements * 4); // upper bound on size
            new_data = work.addr;
            std::vector<int64_t> hist_cur(1 << 4, 0);

            int chunk_size = 32 * 512;
            const int nchunk = (nelements + chunk_size - 1)/chunk_size;
            const int nthread_use = nthread > 1 ? std::max(1, std::min(nthread, nchunk)) : 1;
            if (nthread_use < 2) {
                new_size = ggml_quantize_chunk(new_type, f32_data, new_data, 0, nelements, hist_cur.data());
            } else {
                size_t counter = 0;
                new_size = 0;
                auto compute = [&mutex, &counter, &hist_cur, &new_size, new_type, f32_data, new_data, nelements, chunk_size] () {
                    std::vector<int64_t> local_hist;
                    size_t local_size = 0;
                    while (true) {
                        std::unique_lock<std::mutex> lock(mutex);
                        size_t first = counter; counter += chunk_size;
                        if (first >= nelements) {
                            if (!local_hist.empty()) {
                                for (int j=0; j<int(local_hist.size()); ++j) hist_cur[j] += local_hist[j];
                                new_size += local_size;
                            }
                            break;
                        }
                        lock.unlock();
                        size_t last = std::min(nelements, first + chunk_size);
                        if (local_hist.empty()) local_hist.resize(hist_cur.size(), 0);
                        local_size += ggml_quantize_chunk(new_type, f32_data, new_data, first, last - first, local_hist.data());
                    }
                };
                if (int(workers.size()) < nthread_use - 1) workers.resize(nthread_use - 1);
                for (int it = 0; it < nthread_use - 1; ++it) workers[it] = std::thread(compute);
                compute();
                for (int it = 0; it < nthread_use - 1; ++it) workers[it].join();
            }

            printf("size = %8.2f MiB -> %8.2f MiB | hist: ", tensor.size/1024.0/1024.0, new_size/1024.0/1024.0);
            for (size_t i = 0; i < hist_cur.size(); i++) {
                hist_all[i] += hist_cur[i];
            }

            for (size_t i = 0; i < hist_cur.size(); i++) {
                printf("%5.3f ", hist_cur[i] / float(nelements));
            }
            printf("\n");
        }
        total_size_org += tensor.size;
        total_size_new += new_size;
        file_saver.write_tensor(tensor, new_type, new_data, new_size);
    }

    printf("%s: model size  = %8.2f MiB\n", __func__, total_size_org/1024.0/1024.0);
    printf("%s: quant size  = %8.2f MiB\n", __func__, total_size_new/1024.0/1024.0);

    {
        int64_t sum_all = 0;
        for (size_t i = 0; i < hist_all.size(); i++) {
            sum_all += hist_all[i];
        }

        printf("%s: hist: ", __func__);
        for (size_t i = 0; i < hist_all.size(); i++) {
            printf("%5.3f ", hist_all[i] / float(sum_all));
        }
        printf("\n");
    }
}

//
// interface implementation
//

struct rwkv_context * rwkv_init_from_file(
                             const char * path_model,
            struct rwkv_context_params   params) {
    ggml_time_init();

    rwkv_context * ctx = new rwkv_context;

    if (params.seed <= 0) {
        params.seed = time(NULL);
    }

    unsigned cur_percentage = 0;
    if (params.progress_callback == NULL) {
        params.progress_callback_user_data = &cur_percentage;
        params.progress_callback = [](float progress, void * ctx) {
            unsigned * cur_percentage_p = (unsigned *) ctx;
            unsigned percentage = (unsigned) (100 * progress);
            while (percentage > *cur_percentage_p) {
                ++*cur_percentage_p;
                fprintf(stderr, ".");
                fflush(stderr);
                if (percentage >= 100) {
                    fprintf(stderr, "\n");
                }
            }
        };
    }

    ctx->rng = std::mt19937(params.seed);
    //ctx->logits_all = params.logits_all;

    ggml_type memory_type = params.f16_rwkv_state ? GGML_TYPE_F16 : GGML_TYPE_F32;

    if (!rwkv_model_load(path_model, *ctx, params.n_ctx, memory_type,
                          params.use_mmap, params.use_mlock, params.vocab_only,
                          params.progress_callback, params.progress_callback_user_data)) {
        fprintf(stderr, "%s: failed to load model\n", __func__);
        rwkv_free(ctx);
        return nullptr;
    }

    // reserve memory for context buffers
    if (!params.vocab_only) {
        if (!rwkv_state_init(ctx->model.hparams, ctx->model.state, memory_type)) {
            fprintf(stderr, "%s: rwkv_state_init() failed for rwkv state\n", __func__);
            rwkv_free(ctx);
            return nullptr;
        }

        {
            const size_t memory_size = ggml_nbytes(ctx->model.state.input_layers[0].ff) * 5 * 2 * ctx->model.hparams.n_layer;
            fprintf(stderr, "%s: rwkv state size  = %7.2f MiB\n", __func__, memory_size / 1024.0 / 1024.0);
        }

        const auto & hparams = ctx->model.hparams;

        // resized during inference
        //if (params.logits_all) {
        //    ctx->logits.reserve(hparams.n_ctx*hparams.n_vocab);
        //} else {
            ctx->logits.reserve(hparams.n_vocab);
        //}

        if (params.embedding){
            ctx->embedding.resize(hparams.n_embd);
        }

        ctx->buf_compute.resize(MEM_REQ_EVAL().at(ctx->model.type));

        ctx->buf_scratch[0].resize(MEM_REQ_SCRATCH0().at(ctx->model.type));
        ctx->buf_scratch[1].resize(MEM_REQ_SCRATCH1().at(ctx->model.type));
    }
    
    // Build graph
    //int number_of_threads = 1;
    //rwkv_build_graph(*ctx, number_of_threads);

    return ctx;
}

void rwkv_free(struct rwkv_context * ctx) {
    delete ctx;
}

int rwkv_model_quantize(
        const char * fname_inp,
        const char * fname_out,
  enum ggml_ftype   ftype,
        int          nthread) {
    try {
        rwkv_model_quantize_internal(fname_inp, fname_out, ftype, nthread);
        return 0;
    } catch (const std::string & err) {
        fprintf(stderr, "%s: failed to quantize: %s\n", __func__, err.c_str());
        return 1;
    }
}

int rwkv_apply_lora_from_file_internal(struct rwkv_context * ctx, const char * path_lora, const char * path_base_model, int n_threads) {
    fprintf(stderr, "%s: applying lora adapter from '%s' - please wait ...\n", __func__, path_lora);

    auto & model = ctx->model;

    const int64_t t_start_lora_us = ggml_time_us();

    auto fin = std::ifstream(path_lora, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, path_lora);
        return 1;
    }

    // verify magic and version
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != 'ggla') {
            fprintf(stderr, "%s: bad file magic\n", __func__);
            return 1;
        }
        uint32_t format_version;
        fin.read((char *) &format_version, sizeof(format_version));

        if (format_version != 1) {
            fprintf(stderr, "%s: unsupported file version\n", __func__ );
            return 1;
        }
    }

    int32_t lora_r;
    int32_t lora_alpha;
    fin.read((char *) &lora_r, sizeof(lora_r));
    fin.read((char *) &lora_alpha, sizeof(lora_alpha));
    float scaling = (float)lora_alpha / (float)lora_r;

    fprintf(stderr, "%s: r = %d, alpha = %d, scaling = %.2f\n", __func__, lora_r, lora_alpha, scaling);


    // create a temporary ggml context to store the lora tensors
    // todo: calculate size from biggest possible tensor
    std::vector<uint8_t> lora_buf(1024ull * 1024ull * 1024ull);
    struct ggml_init_params params;
    params.mem_size   = lora_buf.size();
    params.mem_buffer = lora_buf.data();
    params.no_alloc   = false;

    ggml_context * lora_ctx = ggml_init(params);
    std::unordered_map<std::string, struct ggml_tensor *> lora_tensors;

    // create a name -> tensor map of the model to accelerate lookups
    std::unordered_map<std::string, struct ggml_tensor*> model_tensors;
    for (auto & kv: model.tensors_by_name) {
        model_tensors.insert(kv);
    }


    // load base model
    std::unique_ptr<rwkv_model_loader> model_loader;
    ggml_context * base_ctx = NULL;
    arch_util_buffer base_buf;
    if (path_base_model) {
        fprintf(stderr, "%s: loading base model from '%s'\n", __func__, path_base_model);
        model_loader.reset(new rwkv_model_loader(path_base_model, /*use_mmap*/ true, /*vocab_only*/ false));

        size_t ctx_size, mmapped_size;
        model_loader->calc_sizes(&ctx_size, &mmapped_size);
        base_buf.resize(ctx_size);

        ggml_init_params base_params;
        base_params.mem_size   = base_buf.size;
        base_params.mem_buffer = base_buf.addr;
        base_params.no_alloc   = model_loader->use_mmap;

        base_ctx = ggml_init(base_params);

        model_loader->ggml_ctx = base_ctx;

        // maybe this should in rwkv_model_loader
        if (model_loader->use_mmap) {
            model_loader->mapping.reset(new arch_util_mmap(&model_loader->file_loaders.at(0)->file, /* prefetch */ false));
        }
    }

    // read tensors and apply
    bool warned = false;
    int n_tensors = 0;
    while (true) {
        int32_t n_dims;
        int32_t length;
        int32_t ftype;

        fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
        fin.read(reinterpret_cast<char *>(&length), sizeof(length));
        fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));
        if (fin.eof()) {
            break;
        }

        int32_t ne[2] = { 1, 1 };
        for (int i = 0; i < n_dims; ++i) {
            fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
        }

        std::string name(length, 0);
        fin.read(&name[0], length);

        // check for lora suffix and get the type of tensor
        const std::string lora_suffix = ".lora";
        size_t pos = name.rfind(lora_suffix);
        if (pos == std::string::npos) {
            fprintf(stderr, "%s: error: '%s' is not a lora tensor\n", __func__, name.c_str());
            return 1;
        }

        std::string lora_type = name.substr(pos + lora_suffix.length());
        std::string base_name = name;
        base_name.erase(pos);
        // fprintf(stderr, "%s: %s => %s (lora type %s) ", __func__, name.c_str(),base_name.c_str(), lora_type.c_str());

        if (model_tensors.find(base_name.data()) == model_tensors.end()) {
            fprintf(stderr, "%s: unknown tensor '%s' in lora adapter\n", __func__, name.data());
            return 1;
        }

        // create ggml tensor
        ggml_type wtype;
        switch (ftype) {
            case 0: wtype = GGML_TYPE_F32;  break;
            case 1: wtype = GGML_TYPE_F16;  break;
            default:
                    {
                        fprintf(stderr, "%s: invalid tensor data type '%d'\n",
                                __func__, ftype);
                        return false;
                    }
        }
        ggml_tensor* lora_tensor;
        if (n_dims == 2) {
            lora_tensor = ggml_new_tensor_2d(lora_ctx, wtype, ne[0], ne[1]);
        }
        else {
            fprintf(stderr, "%s: unsupported tensor dimension %d\n", __func__, n_dims);
            return 1;
        }

        // load tensor data
        size_t offset = fin.tellg();
        size_t tensor_data_size = ggml_nbytes(lora_tensor);
        offset = (offset + 31) & -32;
        fin.seekg(offset);
        fin.read((char*)lora_tensor->data, tensor_data_size);

        lora_tensors[name] = lora_tensor;

        // check if we have both A and B tensors and apply
        if (lora_tensors.find(base_name + ".loraA") != lora_tensors.end() &&
            lora_tensors.find(base_name + ".loraB") != lora_tensors.end()) {

            ggml_tensor * dest_t = model_tensors[base_name];
            ggml_tensor * base_t;
            if (model_loader) {
                // load from base model
                if (model_loader->tensors_map.name_to_idx.find(base_name) == model_loader->tensors_map.name_to_idx.end()) {
                    fprintf(stderr, "%s: error: tensor '%s' not found in base model\n", __func__, base_name.c_str());
                    return 1;
                }
                size_t idx = model_loader->tensors_map.name_to_idx[base_name];
                rwkv_load_tensor & lt = model_loader->tensors_map.tensors[idx];
                base_t = model_loader->get_tensor(base_name, { (uint32_t)dest_t->ne[0], (uint32_t)dest_t->ne[1] });
                lt.data = (uint8_t *) lt.ggml_tensor->data;
                model_loader->load_data_for(lt);
                lt.ggml_tensor->data = lt.data;
            }
            else {
                base_t = dest_t;
            }

            if (ggml_is_quantized(base_t->type)) {
                if (!warned) {
                    fprintf(stderr, "%s: warning: using a lora adapter with a quantized model may result in poor quality, "
                                    "use a f16 or f32 base model with --lora-base\n", __func__);
                    warned = true;
                }
            }

            ggml_tensor * loraA = lora_tensors[base_name + ".loraA"];
            ggml_tensor * loraB = lora_tensors[base_name + ".loraB"];

            if (base_t->ne[0] != loraA->ne[1] || base_t->ne[1] != loraB->ne[1]) {
                fprintf(stderr, "%s: incompatible tensor dimensions (%" PRId64 " and %" PRId64 ");"
                               " are you sure that this adapter is for this model?\n", __func__, base_t->ne[0], loraA->ne[1]);
                return 1;
            }

            // w = w + BA*s
            ggml_tensor * BA = ggml_mul_mat(lora_ctx, loraA, loraB);

            if (scaling != 1.0f) {
                ggml_tensor * scale_tensor = ggml_new_f32(lora_ctx, scaling);
                BA = ggml_scale(lora_ctx, BA, scale_tensor);
            }

            ggml_tensor * r;
            if (base_t == dest_t) {
                r = ggml_add_inplace(lora_ctx, dest_t, BA);
            }
            else {
                r = ggml_add(lora_ctx, base_t, BA);
                r = ggml_cpy(lora_ctx, r, dest_t);
            }

            struct ggml_cgraph gf = ggml_build_forward(r);
            gf.n_threads = n_threads;
            ggml_graph_compute(lora_ctx, &gf);

            // we won't need these tensors again, reset the context to save memory
            ggml_free(lora_ctx);
            lora_ctx = ggml_init(params);
            lora_tensors.clear();

            n_tensors++;
            if (n_tensors % 4 == 0)
                fprintf(stderr, ".");
        }
    }

    // TODO: this should be in a destructor, it will leak on failure
    ggml_free(lora_ctx);
    if (base_ctx) {
        ggml_free(base_ctx);
    }

    const int64_t t_lora_us = ggml_time_us() - t_start_lora_us;
    fprintf(stderr, " done (%.2f ms)\n", t_lora_us / 1000.0);

    return 0;
}

int rwkv_apply_lora_from_file(struct rwkv_context * ctx, const char * path_lora, const char * path_base_model, int n_threads) {
    try {
        return rwkv_apply_lora_from_file_internal(ctx, path_lora, path_base_model, n_threads);
    } catch (const std::string & err) {
        fprintf(stderr, "%s: failed to apply lora adapter: %s\n", __func__, err.c_str());
        return 1;
    }
}

#define RWKV_MAX_RNG_STATE 64*1024

void rwkv_set_rng_seed(struct rwkv_context * ctx, int seed) {
    if (seed <= 0) {
        seed = time(NULL);
    }
    ctx->rng.seed(seed);
}

// Returns the size of the state
size_t rwkv_get_state_size(struct rwkv_context * ctx) {
    // we don't know size of rng until we actually serialize it. so reserve more than enough memory for its serialized state.
    // for reference, std::mt19937(1337) serializes to 6701 bytes.
    const size_t s_rng_size        = sizeof(size_t);
    const size_t s_rng             = RWKV_MAX_RNG_STATE;
    const size_t s_logits_capacity = sizeof(size_t);
    const size_t s_logits_size     = sizeof(size_t);
    const size_t s_logits          = ctx->logits.capacity() * sizeof(float);
    const size_t s_embedding_size  = sizeof(size_t);
    const size_t s_embedding       = ctx->embedding.size() * sizeof(float);
    const size_t s_rwkv_state_size         = sizeof(size_t);
    const size_t s_rwkv_state_ntok         = sizeof(int);
    const size_t s_rwkv_state              = ctx->model.state.buf.size;

    const size_t s_total = (
        + s_rng_size
        + s_rng
        + s_logits_capacity
        + s_logits_size
        + s_logits
        + s_embedding_size
        + s_embedding
        + s_rwkv_state_size
        + s_rwkv_state_ntok
        + s_rwkv_state
    );

    return s_total;
}

// Copies the state to the specified destination address
size_t rwkv_copy_state_data(struct rwkv_context * ctx, uint8_t * dest) {
    uint8_t * out = dest;

    // copy rng
    {
        std::stringstream rng_ss;
        rng_ss << ctx->rng;

        const size_t rng_size = rng_ss.str().size();
        char rng_buf[RWKV_MAX_RNG_STATE];

        memset(&rng_buf[0], 0, RWKV_MAX_RNG_STATE);
        memcpy(&rng_buf[0], rng_ss.str().data(), rng_ss.str().size());

        memcpy(out, &rng_size,   sizeof(rng_size));    out += sizeof(rng_size);
        memcpy(out, &rng_buf[0], RWKV_MAX_RNG_STATE); out += RWKV_MAX_RNG_STATE;
    }

    // copy logits
    {
        const size_t logits_cap  = ctx->logits.capacity();
        const size_t logits_size = ctx->logits.size();

        memcpy(out, &logits_cap,  sizeof(logits_cap));  out += sizeof(logits_cap);
        memcpy(out, &logits_size, sizeof(logits_size)); out += sizeof(logits_size);

        if (logits_size) {
            memcpy(out, ctx->logits.data(), logits_size * sizeof(float));
        }

        out += logits_cap * sizeof(float);
    }

    // copy embeddings
    {
        const size_t embedding_size = ctx->embedding.size();

        memcpy(out, &embedding_size, sizeof(embedding_size)); out += sizeof(embedding_size);

        if (embedding_size) {
            memcpy(out, ctx->embedding.data(), embedding_size * sizeof(float));
            out += embedding_size * sizeof(float);
        }
    }

    // copy rwkv_state
    {
        const size_t rwkv_state_size = ctx->model.state.buf.size;
        //const int    rwkv_state_ntok = 5; //rwkv_get_kv_cache_token_count(ctx);

        memcpy(out, &rwkv_state_size, sizeof(rwkv_state_size)); out += sizeof(rwkv_state_size);
        //memcpy(out, &rwkv_state_ntok, sizeof(rwkv_state_ntok)); out += sizeof(rwkv_state_ntok);

        if (rwkv_state_size) {
            memcpy(out, ctx->model.state.buf.addr, rwkv_state_size); out += rwkv_state_size;
        }
    }

    const size_t written  = out - dest;
    const size_t expected = rwkv_get_state_size(ctx);

    ARCH_ASSERT(written == expected);

    return written;
}

// TODO: FIX ME! States need work
// Sets the state reading from the specified source address
size_t rwkv_set_state_data(struct rwkv_context * ctx, const uint8_t * src) {
    const uint8_t * in = src;

    // set rng
    {
        size_t rng_size;
        char   rng_buf[RWKV_MAX_RNG_STATE];

        memcpy(&rng_size,   in, sizeof(rng_size));    in += sizeof(rng_size);
        memcpy(&rng_buf[0], in, RWKV_MAX_RNG_STATE); in += RWKV_MAX_RNG_STATE;

        std::stringstream rng_ss;
        rng_ss.str(std::string(&rng_buf[0], rng_size));
        rng_ss >> ctx->rng;

        ARCH_ASSERT(rng_ss.fail() == false);
    }

    // set logits
    {
        size_t logits_cap;
        size_t logits_size;

        memcpy(&logits_cap,  in, sizeof(logits_cap));  in += sizeof(logits_cap);
        memcpy(&logits_size, in, sizeof(logits_size)); in += sizeof(logits_size);

        ARCH_ASSERT(ctx->logits.capacity() == logits_cap);

        if (logits_size) {
            ctx->logits.resize(logits_size);
            memcpy(ctx->logits.data(), in, logits_size * sizeof(float));
        }

        in += logits_cap * sizeof(float);
    }

    // set embeddings
    {
        size_t embedding_size;

        memcpy(&embedding_size, in, sizeof(embedding_size)); in += sizeof(embedding_size);

        ARCH_ASSERT(ctx->embedding.capacity() == embedding_size);

        if (embedding_size) {
            memcpy(ctx->embedding.data(), in, embedding_size * sizeof(float));
            in += embedding_size * sizeof(float);
        }
    }

    // set rwkv_state
    {
        size_t rwkv_state_size;
        //int rwkv_state_ntok;

        memcpy(&rwkv_state_size, in, sizeof(rwkv_state_size)); in += sizeof(rwkv_state_size);
        //memcpy(&rwkv_state_ntok, in, sizeof(rwkv_state_ntok)); in += sizeof(rwkv_state_ntok);

        if (rwkv_state_size) {
            ARCH_ASSERT(ctx->model.state.buf.size == rwkv_state_size);

            // remember data pointers because their value is stored in buf and overwritten by memcpy
            /*
            std::vector<void *> attn;
            std::vector<void *> num;
            std::vector<void *> den;
            std::vector<void *> max;
            std::vector<void *> ff;
            for(int i = 0; i < ctx->model.hparams.n_layer; i++) {
                attn.push_back(ctx->model.state.attn[i]->data);
                num.push_back(ctx->model.state.num[i]->data);
                den.push_back(ctx->model.state.den[i]->data);
                max.push_back(ctx->model.state.max[i]->data);
                ff.push_back(ctx->model.state.ff[i]->data);
            }
             */
            
            /*
             void * attn_data = ctx->model.state.attn->data;
            void * num_data  = ctx->model.state.num->data;
            void * den_data  = ctx->model.state.den->data;
            void * max_data  = ctx->model.state.max->data;
            void * ff_data   = ctx->model.state.ff->data;
             */
            
            memcpy(ctx->model.state.buf.addr, in, rwkv_state_size); in += rwkv_state_size;

            // restore correct data pointers
            /*
            for(int i = 0; i < ctx->model.hparams.n_layer; i++) {
                ctx->model.state.attn[i]->data = attn[i];
                ctx->model.state.num[i]->data = num[i];
                ctx->model.state.den[i]->data = den[i];
                ctx->model.state.max[i]->data = max[i];
                ctx->model.state.ff[i]->data = ff[i];
            }
             */
            
            /*
            ctx->model.state.attn->data = attn_data;
            ctx->model.state.num->data  = num_data;
            ctx->model.state.den->data  = den_data;
            ctx->model.state.max->data  = max_data;
            ctx->model.state.ff->data   = ff_data;
             */
        }

        //ctx->model.state.n = rwkv_state_ntok;
    }

    const size_t nread    = in - src;
    const size_t expected = rwkv_get_state_size(ctx);

    ARCH_ASSERT(nread == expected);

    return nread;
}

int rwkv_eval(struct rwkv_context * ctx,
                   const rwkv_token token,
              const char * dot_path) {
    if (!rwkv_eval_internal(*ctx, token, dot_path)) {
        fprintf(stderr, "%s: failed to eval\n", __func__);
        return 1;
    }
    // get a more accurate load time, upon first eval
    if (!ctx->has_evaluated_once) {
        ctx->t_load_us = ggml_time_us() - ctx->t_start_us;
        ctx->has_evaluated_once = true;
    }
    return 0;
}

int rwkv_tokenize(
        struct rwkv_context * ctx,
                  const char * text,
                 rwkv_token * tokens,
                         int   n_max_tokens,
                        bool   add_bos) {
    auto res = rwkv_tokenize(ctx->vocab, text, add_bos);

    if (n_max_tokens < (int) res.size()) {
        fprintf(stderr, "%s: too many tokens\n", __func__);
        return -((int) res.size());
    }

    for (size_t i = 0; i < res.size(); i++) {
        tokens[i] = res[i];
    }

    return res.size();
}

int rwkv_n_vocab(struct rwkv_context * ctx) {
    return ctx->vocab.id_to_token.size();
}

int rwkv_n_ctx(struct rwkv_context * ctx) {
    return ctx->model.hparams.n_ctx;
}

int rwkv_n_embd(struct rwkv_context * ctx) {
    return ctx->model.hparams.n_embd;
}

float * rwkv_get_logits(struct rwkv_context * ctx) {
    return (float *)(ctx->model.state.logits->data);
    //return ctx->logits.data();
}

float * rwkv_get_embeddings(struct rwkv_context * ctx) {
    return ctx->embedding.data();
}

const char * rwkv_token_to_str(struct rwkv_context * ctx, rwkv_token token) {
    /*if (token >= rwkv_n_vocab(ctx)) {
        return nullptr;
    }*/

    //return ctx->vocab.id_to_token[token].tok.c_str();
    return ctx->vocab.id_to_token[token].c_str();
}

rwkv_token rwkv_str_to_token(struct rwkv_context * ctx, const char * str) {
    return ctx->vocab.token_to_id[str];
}

rwkv_token rwkv_token_bos() {
    return 0;
}

rwkv_token rwkv_token_eos() {
    return 0;
}


void rwkv_print_timings(struct rwkv_context * ctx) {
    const int64_t t_end_us = ggml_time_us();

    const int32_t n_sample = std::max(1, ctx->n_sample);
    const int32_t n_eval   = std::max(1, ctx->n_eval);
    const int32_t n_p_eval = std::max(1, ctx->n_p_eval);

    fprintf(stderr, "\n");
    fprintf(stderr, "%s:        load time = %8.2f ms\n", __func__, ctx->t_load_us / 1000.0);
    fprintf(stderr, "%s:      sample time = %8.2f ms / %5d runs   (%8.2f ms per run)\n",   __func__, 1e-3 * ctx->t_sample_us, n_sample, 1e-3 * ctx->t_sample_us / n_sample);
    fprintf(stderr, "%s: prompt eval time = %8.2f ms / %5d tokens (%8.2f ms per token)\n", __func__, 1e-3 * ctx->t_p_eval_us, n_p_eval, 1e-3 * ctx->t_p_eval_us / n_p_eval);
    fprintf(stderr, "%s:        eval time = %8.2f ms / %5d runs   (%8.2f ms per run)\n",   __func__, 1e-3 * ctx->t_eval_us,   n_eval,   1e-3 * ctx->t_eval_us   / n_eval);
    fprintf(stderr, "%s:       total time = %8.2f ms\n", __func__, (t_end_us - ctx->t_start_us)/1000.0);
}

void rwkv_reset_timings(struct rwkv_context * ctx) {
    ctx->t_start_us = ggml_time_us();
    ctx->t_sample_us = ctx->n_sample = 0;
    ctx->t_eval_us   = ctx->n_eval   = 0;
    ctx->t_p_eval_us = ctx->n_p_eval = 0;
}

const char * rwkv_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "         + std::to_string(ggml_cpu_has_avx())         + " | ";
    s += "AVX2 = "        + std::to_string(ggml_cpu_has_avx2())        + " | ";
    s += "AVX512 = "      + std::to_string(ggml_cpu_has_avx512())      + " | ";
    s += "AVX512_VBMI = " + std::to_string(ggml_cpu_has_avx512_vbmi()) + " | ";
    s += "AVX512_VNNI = " + std::to_string(ggml_cpu_has_avx512_vnni()) + " | ";
    s += "FMA = "         + std::to_string(ggml_cpu_has_fma())         + " | ";
    s += "NEON = "        + std::to_string(ggml_cpu_has_neon())        + " | ";
    s += "ARM_FMA = "     + std::to_string(ggml_cpu_has_arm_fma())     + " | ";
    s += "F16C = "        + std::to_string(ggml_cpu_has_f16c())        + " | ";
    s += "FP16_VA = "     + std::to_string(ggml_cpu_has_fp16_va())     + " | ";
    s += "WASM_SIMD = "   + std::to_string(ggml_cpu_has_wasm_simd())   + " | ";
    s += "BLAS = "        + std::to_string(ggml_cpu_has_blas())        + " | ";
    s += "SSE3 = "        + std::to_string(ggml_cpu_has_sse3())        + " | ";
    s += "VSX = "         + std::to_string(ggml_cpu_has_vsx())         + " | ";

    return s.c_str();
}

// For internal test use
std::vector<std::pair<std::string, struct ggml_tensor *>>& rwkv_internal_get_tensor_map(struct rwkv_context * ctx) {
    return ctx->model.tensors_by_name;
}

size_t rwkv_load_session_file(struct rwkv_context * ctx, const char * path_session, rwkv_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out) {
    // TODO leverage mmap
    arch_util_file file(path_session, "rb");
    const uint32_t magic = file.read_u32();
    const uint32_t version = file.read_u32();

    if (!(magic == 'ggsn' && version == 0)) {
        fprintf(stderr, "%s : unknown (magic, version) for session file: %08x, %08x\n", __func__, magic, version);
        return 0;
    }

    rwkv_hparams session_hparams;
    file.read_raw(&session_hparams, sizeof(rwkv_hparams));

    // REVIEW
    if (session_hparams != ctx->model.hparams) {
        fprintf(stderr, "%s : model hparams didn't match from session file!\n", __func__);
        return 0;
    }

    const uint32_t n_token_count = file.read_u32();
    ARCH_ASSERT(n_token_capacity >= n_token_count);
    file.read_raw(tokens_out, sizeof(rwkv_token) * n_token_count);
    *n_token_count_out = n_token_count;

    const size_t n_state_size = file.size - file.tell();
    const size_t n_orig_state_size = rwkv_get_state_size(ctx);
    if (n_state_size != n_orig_state_size) {
        fprintf(stderr, "%s : failed to validate state size\n", __func__);
    }
    std::unique_ptr<uint8_t[]> state_data(new uint8_t[n_state_size]);
    file.read_raw(state_data.get(), n_state_size);
    return rwkv_set_state_data(ctx, state_data.get());
}

size_t rwkv_save_session_file(struct rwkv_context * ctx, const char * path_session, const rwkv_token * tokens, size_t n_token_count) {
    // TODO save temp & swap
    arch_util_file file(path_session, "wb");

    const size_t n_state_size = rwkv_get_state_size(ctx);
    std::unique_ptr<uint8_t[]> state_data(new uint8_t[n_state_size]);
    rwkv_copy_state_data(ctx, state_data.get());

    file.write_u32('ggsn'); // magic
    file.write_u32(0); // version
    file.write_raw(&ctx->model.hparams, sizeof(rwkv_hparams));

    file.write_u32((uint32_t) n_token_count); // REVIEW
    file.write_raw(tokens, sizeof(rwkv_token) * n_token_count);

    file.write_raw(state_data.get(), n_state_size);
    return n_state_size; // REVIEW
}

