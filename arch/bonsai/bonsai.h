#ifndef BONSAI_H
#define BONSAI_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef BONSAI_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef BONSAI_BUILD
#            define BONSAI_API __declspec(dllexport)
#        else
#            define BONSAI_API __declspec(dllimport)
#        endif
#    else
#        define BONSAI_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define BONSAI_API
#endif

#define BONSAI_FILE_VERSION 1
#define BONSAI_FILE_MAGIC 0x67676a74 // 'ggjt' in hex
#define BONSAI_FILE_MAGIC_UNVERSIONED 0x67676d6c // pre-versioned files

#define DEFAULT_SEED           0xFFFFFFFF

#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct bonsai_context;

    typedef int bonsai_token;

    typedef struct bonsai_token_data {
        bonsai_token id;  // token id
        float logit; // log-odds of the token
        float p;     // probability of the token
    } bonsai_token_data;

    typedef struct bonsai_token_data_array {
        bonsai_token_data * data;
        size_t size;
        bool sorted;
    } bonsai_token_data_array;

    typedef void (*bonsai_progress_callback)(float progress, void *ctx);

    struct bonsai_context_params {
        uint32_t seed;      // RNG seed, 0 for random
        int32_t  n_ctx;     // text context
        int32_t  n_batch;   // prompt processing batch size

        bool f16_kv;     // use fp16 for KV cache
        bool logits_all; // the bonsai_eval() call computes all logits, not just the last one
        bool vocab_only; // only load the vocabulary, no weights
        bool use_mmap;   // use mmap if possible
        bool use_mlock;  // force system to keep model in RAM
        bool embedding;  // embedding mode only

        // called with a progress value between 0 and 1, pass NULL to disable
        bonsai_progress_callback progress_callback;
        // context pointer passed to the progress callback
        void * progress_callback_user_data;
    };

    // model file types
    enum bonsai_ftype {
        BONSAI_FTYPE_ALL_F32     = 0,
        BONSAI_FTYPE_MOSTLY_F16  = 1,  // except 1d tensors
        BONSAI_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
        BONSAI_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
        BONSAI_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
        BONSAI_FTYPE_MOSTLY_Q4_2 = 5,  // except 1d tensors
        // BONSAI_FTYPE_MOSTLY_Q4_3 (6) support has been removed
        BONSAI_FTYPE_MOSTLY_Q8_0 = 7,  // except 1d tensors
        BONSAI_FTYPE_MOSTLY_Q5_0 = 8,  // except 1d tensors
        BONSAI_FTYPE_MOSTLY_Q5_1 = 9,  // except 1d tensors
    };

    BONSAI_API struct bonsai_context_params bonsai_context_default_params();

    BONSAI_API bool bonsai_mmap_supported();
    BONSAI_API bool bonsai_mlock_supported();

    // Various functions for loading a ggml llama model.
    // Allocate (almost) all memory needed for the model.
    // Return NULL on failure
    BONSAI_API struct bonsai_context * bonsai_init_from_file(
                             const char * path_model,
            struct bonsai_context_params   params);

    // Frees all allocated memory
    BONSAI_API void bonsai_free(struct bonsai_context * ctx);

    // TODO: not great API - very likely to change
    // Returns 0 on success
    // nthread - how many threads to use. If <=0, will use std::thread::hardware_concurrency(), else the number given
    BONSAI_API int bonsai_model_quantize(
            const char * fname_inp,
            const char * fname_out,
      enum bonsai_ftype   ftype,
            int          nthread);

    BONSAI_API int bonsai_model_update(
            const char * fname_inp,
            const char * fname_out);

    // Apply a LoRA adapter to a loaded model
    // path_base_model is the path to a higher quality model to use as a base for
    // the layers modified by the adapter. Can be NULL to use the current loaded model.
    // The model needs to be reloaded before applying a new adapter, otherwise the adapter
    // will be applied on top of the previous one
    // Returns 0 on success
    BONSAI_API int bonsai_apply_lora_from_file(
            struct bonsai_context * ctx,
                      const char * path_lora,
                      const char * path_base_model,
                             int   n_threads);

    // Returns the number of tokens in the KV cache
    BONSAI_API int bonsai_get_kv_cache_token_count(struct bonsai_context * ctx);

    // Shifts the KV cache effectively removing the first n tokens
    BONSAI_API void bonsai_shift_kv_cache(struct bonsai_context * ctx, int n);

    // Sets the current rng seed.
    BONSAI_API void bonsai_set_rng_seed(struct bonsai_context * ctx, int seed);

    // Returns the size in bytes of the state (rng, logits, embedding and kv_cache)
    BONSAI_API size_t bonsai_get_state_size(struct bonsai_context * ctx);

    // Copies the state to the specified destination address.
    // Destination needs to have allocated enough memory.
    // Returns the number of bytes copied
    BONSAI_API size_t bonsai_copy_state_data(struct bonsai_context * ctx, uint8_t * dest);

    // Set the state reading from the specified address
    // Returns the number of bytes read
    BONSAI_API size_t bonsai_set_state_data(struct bonsai_context * ctx, const uint8_t * src);

    // Save/load session file
    BONSAI_API size_t bonsai_load_session_file(struct bonsai_context * ctx, const char * path_session, bonsai_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out);
    BONSAI_API size_t bonsai_save_session_file(struct bonsai_context * ctx, const char * path_session, const bonsai_token * tokens, size_t n_token_count);

    // Run the llama inference to obtain the logits and probabilities for the next token.
    // tokens + n_tokens is the provided batch of new tokens to process
    // n_past is the number of tokens to use from previous eval calls
    // Returns 0 on success
    BONSAI_API int bonsai_eval(
            struct bonsai_context * ctx,
               const bonsai_token * tokens,
                             int   n_tokens,
                             int   n_past,
                             int   n_threads);

    // Convert the provided text into tokens.
    // The tokens pointer must be large enough to hold the resulting tokens.
    // Returns the number of tokens on success, no more than n_max_tokens
    // Returns a negative number on failure - the number of tokens that would have been returned
    // TODO: not sure if correct
    BONSAI_API int bonsai_tokenize(
            struct bonsai_context * ctx,
                      const char * text,
                     bonsai_token * tokens,
                             int   n_max_tokens,
                            bool   add_bos);

    BONSAI_API int bonsai_n_vocab(struct bonsai_context * ctx);
    BONSAI_API int bonsai_n_ctx  (struct bonsai_context * ctx);
    BONSAI_API int bonsai_n_embd (struct bonsai_context * ctx);

    // Token logits obtained from the last call to bonsai_eval()
    // The logits for the last token are stored in the last row
    // Can be mutated in order to change the probabilities of the next token
    // Rows: n_tokens
    // Cols: n_vocab
    BONSAI_API float * bonsai_get_logits(struct bonsai_context * ctx);

    // Get the embeddings for the input
    // shape: [n_embd] (1-dimensional)
    BONSAI_API float * bonsai_get_embeddings(struct bonsai_context * ctx);

    // Token Id -> String. Uses the vocabulary in the provided context
    BONSAI_API const char * bonsai_token_to_str(struct bonsai_context * ctx, bonsai_token token);

    // String -> Token Id. Uses the vocabulary in the provided context
    BONSAI_API bonsai_token bonsai_str_to_token(struct bonsai_context * ctx, const char * str);

    // Special tokens
    BONSAI_API bonsai_token bonsai_token_bos();
    BONSAI_API bonsai_token bonsai_token_eos();

    // Sampling functions

    /// @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
    BONSAI_API void bonsai_sample_repetition_penalty(struct bonsai_context * ctx, bonsai_token_data_array * candidates, bonsai_token * last_tokens, size_t last_tokens_size, float penalty);

    /// @details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
    BONSAI_API void bonsai_sample_frequency_and_presence_penalties(struct bonsai_context * ctx, bonsai_token_data_array * candidates, bonsai_token * last_tokens, size_t last_tokens_size, float alpha_frequency, float alpha_presence);

    /// @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
    BONSAI_API void bonsai_sample_softmax(struct bonsai_context * ctx, bonsai_token_data_array * candidates);

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    BONSAI_API void bonsai_sample_top_k(struct bonsai_context * ctx, bonsai_token_data_array * candidates, int k, size_t min_keep);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    BONSAI_API void bonsai_sample_top_p(struct bonsai_context * ctx, bonsai_token_data_array * candidates, float p, size_t min_keep);

    /// @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
    BONSAI_API void bonsai_sample_tail_free(struct bonsai_context * ctx, bonsai_token_data_array * candidates, float z, size_t min_keep);

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    BONSAI_API void bonsai_sample_typical(struct bonsai_context * ctx, bonsai_token_data_array * candidates, float p, size_t min_keep);
    BONSAI_API void bonsai_sample_temperature(struct bonsai_context * ctx, bonsai_token_data_array * candidates, float temp);

    /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `bonsai_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    BONSAI_API bonsai_token bonsai_sample_token_mirostat(struct bonsai_context * ctx, bonsai_token_data_array * candidates, float tau, float eta, int m, float * mu);

    /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `bonsai_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    BONSAI_API bonsai_token bonsai_sample_token_mirostat_v2(struct bonsai_context * ctx, bonsai_token_data_array * candidates, float tau, float eta, float * mu);

    /// @details Selects the token with the highest probability.
    BONSAI_API bonsai_token bonsai_sample_token_greedy(struct bonsai_context * ctx, bonsai_token_data_array * candidates);

    /// @details Randomly selects a token from the candidates based on their probabilities.
    BONSAI_API bonsai_token bonsai_sample_token(struct bonsai_context * ctx, bonsai_token_data_array * candidates);

    // Performance information
    BONSAI_API void bonsai_print_timings(struct bonsai_context * ctx);
    BONSAI_API void bonsai_reset_timings(struct bonsai_context * ctx);

    // Print system information
    BONSAI_API const char * bonsai_print_system_info(void);

#ifdef __cplusplus
}
#endif

// Internal API to be implemented by llama.cpp and used by tests/benchmarks only
#ifdef BONSAI_API_INTERNAL

#include <vector>
#include <string>
struct ggml_tensor;

std::vector<std::pair<std::string, struct ggml_tensor *>>& bonsai_internal_get_tensor_map(struct bonsai_context * ctx);

#endif

#endif // BONSAI_H
