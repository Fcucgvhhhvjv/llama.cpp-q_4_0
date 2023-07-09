#ifndef RWKV_H
#define RWKV_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#include "../ggml.h"

#ifdef RWKV_SHARED
#    if defined(_WIN32) && !defined(__MINGW32__)
#        ifdef RWKV_BUILD
#            define RWKV_API __declspec(dllexport)
#        else
#            define RWKV_API __declspec(dllimport)
#        endif
#    else
#        define RWKV_API __attribute__ ((visibility ("default")))
#    endif
#else
#    define RWKV_API
#endif

#define GGML_FILE_VERSION 1
#define GGJT_FILE_MAGIC 0x67676a74 // 'ggjt' in hex
#define GGML_FILE_MAGIC_UNVERSIONED 0x67676d6c // pre-versioned files

#define DEFAULT_SEED           0xFFFFFFFF

#ifdef __cplusplus
extern "C" {
#endif

    //
    // C interface
    //
    // TODO: show sample usage
    //

    struct rwkv_context;

    typedef int rwkv_token;

    typedef struct rwkv_token_data {
        rwkv_token id;  // token id
        float logit; // log-odds of the token
        float p;     // probability of the token
    } rwkv_token_data;

    typedef struct rwkv_token_data_array {
        rwkv_token_data * data;
        size_t size;
        bool sorted;
    } rwkv_token_data_array;

    typedef void (*rwkv_progress_callback)(float progress, void *ctx);

    struct rwkv_context_params {
        uint32_t seed;      // RNG seed, 0 for random
        int32_t  n_ctx;     // text context
        int32_t  n_batch;   // prompt processing batch size

        bool f16_rwkv_state;     // use fp16 for RWKV state
        //bool logits_all; // the rwkv_eval() call computes all logits, not just the last one
        bool vocab_only; // only load the vocabulary, no weights
        bool use_mmap;   // use mmap if possible
        bool use_mlock;  // force system to keep model in RAM
        bool embedding;  // embedding mode only

        // called with a progress value between 0 and 1, pass NULL to disable
        rwkv_progress_callback progress_callback;
        // context pointer passed to the progress callback
        void * progress_callback_user_data;
    };

    // model file types
    /*enum ggml_ftype {
        GGML_FTYPE_ALL_F32     = 0,
        GGML_FTYPE_MOSTLY_F16  = 1,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_0 = 2,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_1 = 3,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
        GGML_FTYPE_MOSTLY_Q4_2 = 5,  // except 1d tensors
        // GGML_FTYPE_MOSTLY_Q4_3 (6) support has been removed
        GGML_FTYPE_MOSTLY_Q8_0 = 7,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q5_0 = 8,  // except 1d tensors
        GGML_FTYPE_MOSTLY_Q5_1 = 9,  // except 1d tensors
    };*/

    RWKV_API struct rwkv_context_params rwkv_context_default_params();

    RWKV_API bool rwkv_mmap_supported();
    RWKV_API bool rwkv_mlock_supported();

    // Various functions for loading a ggml llama model.
    // Allocate (almost) all memory needed for the model.
    // Return NULL on failure
    RWKV_API struct rwkv_context * rwkv_init_from_file(
                             const char * path_model,
            struct rwkv_context_params   params);

    // Frees all allocated memory
    RWKV_API void rwkv_free(struct rwkv_context * ctx);

    // TODO: not great API - very likely to change
    // Returns 0 on success
    // nthread - how many threads to use. If <=0, will use std::thread::hardware_concurrency(), else the number given
    RWKV_API int rwkv_model_quantize(
            const char * fname_inp,
            const char * fname_out,
      enum ggml_ftype   ftype,
            int          nthread);

    RWKV_API int rwkv_model_update(
            const char * fname_inp,
            const char * fname_out);

    // Apply a LoRA adapter to a loaded model
    // path_base_model is the path to a higher quality model to use as a base for
    // the layers modified by the adapter. Can be NULL to use the current loaded model.
    // The model needs to be reloaded before applying a new adapter, otherwise the adapter
    // will be applied on top of the previous one
    // Returns 0 on success
    RWKV_API int rwkv_apply_lora_from_file(
            struct rwkv_context * ctx,
                      const char * path_lora,
                      const char * path_base_model,
                             int   n_threads);

    // Returns the number of tokens in the KV cache
    //RWKV_API int rwkv_get_kv_cache_token_count(struct rwkv_context * ctx);

    // Shifts the KV cache effectively removing the first n tokens
    //RWKV_API void rwkv_shift_kv_cache(struct rwkv_context * ctx, int n);

    // Sets the current rng seed.
    RWKV_API void rwkv_set_rng_seed(struct rwkv_context * ctx, int seed);

    // Returns the size in bytes of the state (rng, logits, embedding and kv_cache)
    RWKV_API size_t rwkv_get_state_size(struct rwkv_context * ctx);

    // Copies the state to the specified destination address.
    // Destination needs to have allocated enough memory.
    // Returns the number of bytes copied
    RWKV_API size_t rwkv_copy_state_data(struct rwkv_context * ctx, uint8_t * dest);

    // Set the state reading from the specified address
    // Returns the number of bytes read
    RWKV_API size_t rwkv_set_state_data(struct rwkv_context * ctx, const uint8_t * src);

    // Save/load session file
    RWKV_API size_t rwkv_load_session_file(struct rwkv_context * ctx, const char * path_session, rwkv_token * tokens_out, size_t n_token_capacity, size_t * n_token_count_out);
    RWKV_API size_t rwkv_save_session_file(struct rwkv_context * ctx, const char * path_session, const rwkv_token * tokens, size_t n_token_count);

    // Run the llama inference to obtain the logits and probabilities for the next token.
    // tokens + n_tokens is the provided batch of new tokens to process
    // n_past is the number of tokens to use from previous eval calls
    // Returns 0 on success
    RWKV_API int rwkv_eval(struct rwkv_context * ctx,
                            const rwkv_token     token,
                            const char         * dot_path);

    RWKV_API int rwkv_opt(struct rwkv_context * ctx,
                            const rwkv_token     token,
                            const rwkv_token     actual,
                            const char         * dot_path);

    // Convert the provided text into tokens.
    // The tokens pointer must be large enough to hold the resulting tokens.
    // Returns the number of tokens on success, no more than n_max_tokens
    // Returns a negative number on failure - the number of tokens that would have been returned
    // TODO: not sure if correct
    RWKV_API int rwkv_tokenize(
            struct rwkv_context * ctx,
                      const char * text,
                     rwkv_token * tokens,
                             int   n_max_tokens,
                            bool   add_bos);

    RWKV_API int rwkv_n_vocab(struct rwkv_context * ctx);
    RWKV_API int rwkv_n_ctx  (struct rwkv_context * ctx);
    RWKV_API int rwkv_n_embd (struct rwkv_context * ctx);

    // Token logits obtained from the last call to rwkv_eval()
    // The logits for the last token are stored in the last row
    // Can be mutated in order to change the probabilities of the next token
    // Rows: n_tokens
    // Cols: n_vocab
    RWKV_API float * rwkv_get_logits(struct rwkv_context * ctx);

    RWKV_API float rwkv_get_error_before(struct rwkv_context * ctx);
    RWKV_API float rwkv_get_error_after(struct rwkv_context * ctx);

    // Get the embeddings for the input
    // shape: [n_embd] (1-dimensional)
    RWKV_API float * rwkv_get_embeddings(struct rwkv_context * ctx);

    // Token Id -> String. Uses the vocabulary in the provided context
    RWKV_API const char * rwkv_token_to_str(struct rwkv_context * ctx, rwkv_token token);

    // String -> Token Id. Uses the vocabulary in the provided context
    RWKV_API rwkv_token rwkv_str_to_token(struct rwkv_context * ctx, const char * str);

    // Special tokens
    RWKV_API rwkv_token rwkv_token_bos();
    RWKV_API rwkv_token rwkv_token_eos();


    // Sampling functions

    /// @details Repetition penalty described in CTRL academic paper https://arxiv.org/abs/1909.05858, with negative logit fix.
    RWKV_API void rwkv_sample_repetition_penalty(struct rwkv_context * ctx, rwkv_token_data_array * candidates, rwkv_token * last_tokens, size_t last_tokens_size, float penalty);

    /// @details Frequency and presence penalties described in OpenAI API https://platform.openai.com/docs/api-reference/parameter-details.
    RWKV_API void rwkv_sample_frequency_and_presence_penalties(struct rwkv_context * ctx, rwkv_token_data_array * candidates, rwkv_token * last_tokens, size_t last_tokens_size, float alpha_frequency, float alpha_presence);

    /// @details Sorts candidate tokens by their logits in descending order and calculate probabilities based on logits.
    RWKV_API void rwkv_sample_softmax(struct rwkv_context * ctx, rwkv_token_data_array * candidates);

    /// @details Top-K sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    RWKV_API void rwkv_sample_top_k(struct rwkv_context * ctx, rwkv_token_data_array * candidates, int k, size_t min_keep);

    /// @details Nucleus sampling described in academic paper "The Curious Case of Neural Text Degeneration" https://arxiv.org/abs/1904.09751
    RWKV_API void rwkv_sample_top_p(struct rwkv_context * ctx, rwkv_token_data_array * candidates, float p, size_t min_keep);

    /// @details Tail Free Sampling described in https://www.trentonbricken.com/Tail-Free-Sampling/.
    RWKV_API void rwkv_sample_tail_free(struct rwkv_context * ctx, rwkv_token_data_array * candidates, float z, size_t min_keep);

    /// @details Locally Typical Sampling implementation described in the paper https://arxiv.org/abs/2202.00666.
    RWKV_API void rwkv_sample_typical(struct rwkv_context * ctx, rwkv_token_data_array * candidates, float p, size_t min_keep);
    RWKV_API void rwkv_sample_temperature(struct rwkv_context * ctx, rwkv_token_data_array * candidates, float temp);

    /// @details Mirostat 1.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `rwkv_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param m The number of tokens considered in the estimation of `s_hat`. This is an arbitrary value that is used to calculate `s_hat`, which in turn helps to calculate the value of `k`. In the paper, they use `m = 100`, but you can experiment with different values to see how it affects the performance of the algorithm.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    RWKV_API rwkv_token rwkv_sample_token_mirostat(struct rwkv_context * ctx, rwkv_token_data_array * candidates, float tau, float eta, int m, float * mu);

    /// @details Mirostat 2.0 algorithm described in the paper https://arxiv.org/abs/2007.14966. Uses tokens instead of words.
    /// @param candidates A vector of `rwkv_token_data` containing the candidate tokens, their probabilities (p), and log-odds (logit) for the current position in the generated text.
    /// @param tau  The target cross-entropy (or surprise) value you want to achieve for the generated text. A higher value corresponds to more surprising or less predictable text, while a lower value corresponds to less surprising or more predictable text.
    /// @param eta The learning rate used to update `mu` based on the error between the target and observed surprisal of the sampled word. A larger learning rate will cause `mu` to be updated more quickly, while a smaller learning rate will result in slower updates.
    /// @param mu Maximum cross-entropy. This value is initialized to be twice the target cross-entropy (`2 * tau`) and is updated in the algorithm based on the error between the target and observed surprisal.
    RWKV_API rwkv_token rwkv_sample_token_mirostat_v2(struct rwkv_context * ctx, rwkv_token_data_array * candidates, float tau, float eta, float * mu);

    /// @details Selects the token with the highest probability.
    RWKV_API rwkv_token rwkv_sample_token_greedy(struct rwkv_context * ctx, rwkv_token_data_array * candidates);

    /// @details Randomly selects a token from the candidates based on their probabilities.
    RWKV_API rwkv_token rwkv_sample_token(struct rwkv_context * ctx, rwkv_token_data_array * candidates);

    // Performance information
    RWKV_API void rwkv_print_timings(struct rwkv_context * ctx);
    RWKV_API void rwkv_reset_timings(struct rwkv_context * ctx);

    // Print system information
    RWKV_API const char * rwkv_print_system_info(void);

#ifdef __cplusplus
}
#endif

// Internal API to be implemented by llama.cpp and used by tests/benchmarks only
#ifdef RWKV_API_INTERNAL

#include <vector>
#include <string>
struct ggml_tensor;

std::vector<std::pair<std::string, struct ggml_tensor *>>& rwkv_internal_get_tensor_map(struct rwkv_context * ctx);

#endif

#endif // RWKV_H
