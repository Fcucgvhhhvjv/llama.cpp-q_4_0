// Defines sigaction on msys:
#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include "common-gptneox.h"
#include "gptneox.h"

#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#include <signal.h>
#endif

static console_state con_st;
static gptneox_context ** g_ctx;

static bool is_interacting = false;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    set_console_color(con_st, CONSOLE_COLOR_DEFAULT);
    printf("\n"); // this also force flush stdout.
    if (signo == SIGINT) {
        if (!is_interacting) {
            is_interacting=true;
        } else {
            gptneox_print_timings(*g_ctx);
            _exit(130);
        }
    }
}
#endif

int main(int argc, char ** argv) {
    gpt_params params;
    params.model = "models/oasst/ggml-oasst-sft-4-pythia-12b-epoch-3.5-q4_0.bin";
    
    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }
    
    // save choice to use color for later
    // (note for later: this is a slightly awkward choice)
    con_st.use_color = params.use_color;
    
#if defined (_WIN32)
    win32_console_init(params.use_color);
#endif
    
    if (params.perplexity) {
        printf("\n************\n");
        printf("%s: please use the 'perplexity' tool for perplexity calculations\n", __func__);
        printf("************\n\n");
        
        return 0;
    }
    
    if (params.embedding) {
        printf("\n************\n");
        printf("%s: please use the 'embedding' tool for embedding calculations\n", __func__);
        printf("************\n\n");
        
        return 0;
    }
    
    if (params.n_ctx > 2048) {
        fprintf(stderr, "%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
                "expect poor results\n", __func__, params.n_ctx);
    }
    
    if (params.seed <= 0) {
        params.seed = time(NULL);
    }
    
    fprintf(stderr, "%s: seed = %d\n", __func__, params.seed);
    
    std::mt19937 rng(params.seed);
    if (params.random_prompt) {
        params.prompt = gpt_random_prompt(rng);
    }
    
    gptneox_context * ctx;
    g_ctx = &ctx;
    
    // load the model
    {
        auto lparams = gptneox_context_default_params();
        
        lparams.n_ctx      = params.n_ctx;
        lparams.n_parts    = params.n_parts;
        lparams.seed       = params.seed;
        lparams.f16_kv     = params.memory_f16;
        lparams.use_mmap   = params.use_mmap;
        lparams.use_mlock  = params.use_mlock;
        
        ctx = gptneox_init_from_file(params.model.c_str(), lparams);
        
        if (ctx == NULL) {
            fprintf(stderr, "%s: error: failed to load model '%s'\n", __func__, params.model.c_str());
            return 1;
        }
    }
    
    if (!params.lora_adapter.empty()) {
        int err = gptneox_apply_lora_from_file(ctx,
                                               params.lora_adapter.c_str(),
                                               params.lora_base.empty() ? NULL : params.lora_base.c_str(),
                                               params.n_threads);
        if (err != 0) {
            fprintf(stderr, "%s: error: failed to apply lora adapter\n", __func__);
            return 1;
        }
    }
    
    // print system information
    {
        fprintf(stderr, "\n");
        fprintf(stderr, "system_info: n_threads = %d / %d | %s\n",
                params.n_threads, std::thread::hardware_concurrency(), gptneox_print_system_info());
    }
    
    // determine the maximum memory usage needed to do inference for the given n_batch and n_predict parameters
    // uncomment the "used_mem" line in llama.cpp to see the results
    if (params.mem_test) {
        {
            const std::vector<gptneox_token> tmp(params.n_batch, 0);
            gptneox_eval(ctx, tmp.data(), tmp.size(), 0, params.n_threads);
        }
        
        {
            const std::vector<gptneox_token> tmp = { 0, };
            gptneox_eval(ctx, tmp.data(), tmp.size(), params.n_predict - 1, params.n_threads);
        }
        
        gptneox_print_timings(ctx);
        gptneox_free(ctx);
        
        return 0;
    }
    
    // prefix & suffix for instruct mode
    const auto prompter_id = 50279; //"<|prompter|>";
    const auto assistant_id = 50281; //"<|endoftext|><|assistant|>";
    //const auto inp_pfx = ::gptneox_tokenize(ctx, "<|prompter|>", false); //true);
    //const auto inp_sfx = ::gptneox_tokenize(ctx, "<|endoftext|><|assistant|>", false);
    
    // Always interactive in Open-Assistant
    params.interactive = true;
    
    if (params.interactive) {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = sigint_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
#elif defined (_WIN32)
        signal(SIGINT, sigint_handler);
#endif
    }
    fprintf(stderr, "sampling: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f\n",
        params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty);
    fprintf(stderr, "generate: n_ctx = %d, n_batch = %d, n_predict = %d, n_keep = %d\n", params.n_ctx, params.n_batch, params.n_predict, params.n_keep);
    fprintf(stderr, "\n\n");
    
    // TODO: replace with ring-buffer
    std::vector<gptneox_token> last_n_tokens(params.n_ctx);
    std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);
    
    if (params.interactive) {
        fprintf(stderr, "== Running in interactive mode. ==\n"
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
               " - Press Ctrl+C to interject at any time.\n"
#endif
               " - Press Return to return control to LLaMa.\n"
               " - If you want to submit another line, end your input in '\\'.\n\n");
    }
    
    const int32_t top_k          = params.top_k;
    const float   top_p          = params.top_p;
    const float   temp           = params.temp;
    const float   repeat_penalty = params.repeat_penalty;
    
    // Chat loop
    while (true) {
        is_interacting = true;
        
        int n_past = 0;
        
        // Get input
        
        // potentially set color to indicate we are taking user input
        set_console_color(con_st, CONSOLE_COLOR_USER_INPUT);
        
#if defined (_WIN32)
        // Windows: must reactivate sigint handler after each signal
        signal(SIGINT, sigint_handler);
#endif

        if (params.instruct) {
            printf("\n> ");
        }

        std::string buffer;
        if (!params.input_prefix.empty()) {
            buffer += params.input_prefix;
            printf("%s", buffer.c_str());
        }

        std::string line;
        bool another_line = true;
        do {
#if defined(_WIN32)
            std::wstring wline;
            if (!std::getline(std::wcin, wline)) {
                // input stream is bad or EOF received
                return 0;
            }
            win32_utf8_encode(wline, line);
#else
            if (!std::getline(std::cin, line)) {
                // input stream is bad or EOF received
                return 0;
            }
#endif
            if (line.empty() || line.back() != '\\') {
                another_line = false;
            } else {
                line.pop_back(); // Remove the continue character
            }
            buffer += line;
            if (another_line) {
                buffer += '\n';
            }
        } while (another_line);
        
        is_interacting = false;
        
        // done taking input, reset color
        set_console_color(con_st, CONSOLE_COLOR_DEFAULT);
        
        // Check for input
        if (buffer.length() <= 0) {
            continue; // Restart loop for input
        }
        
        // Tokenize prompt with oasst special tokens
        auto prompt_embd = ::gptneox_tokenize(ctx, buffer, false);
        auto embd_inp = std::vector<gptneox_token>();
        embd_inp.push_back(prompter_id);
        embd_inp.insert(embd_inp.end(), prompt_embd.begin(), prompt_embd.end());
        embd_inp.push_back(0);
        embd_inp.push_back(assistant_id);
        
        // Verbose prompt
        if (params.verbose_prompt) {
            fprintf(stderr, "\n");
            fprintf(stderr, "%s: prompt: '%s'\n", __func__, buffer.c_str());
            fprintf(stderr, "%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
            for (int i = 0; i < (int) embd_inp.size(); i++) {
                fprintf(stderr, "%6d -> '%s'\n", embd_inp[i], gptneox_token_to_str(ctx, embd_inp[i]));
            }
            /*if (params.n_keep > 0) {
            fprintf(stderr, "%s: static prompt based on n_keep: '", __func__);
                for (int i = 0; i < params.n_keep; i++) {
                    fprintf(stderr, "%s", gptneox_token_to_str(ctx, embd_inp[i]));
                }
                fprintf(stderr, "'\n");
            }
             */
            fprintf(stderr, "\n");
        }
        
        // How many tokens to generate - check if theres space in context for atleast one token (or batch size tokens?)
        auto inp_size = embd_inp.size();
        auto space = params.n_ctx - inp_size;
        if(space <= 0) {
            fprintf(stderr, "%s : input too long\n", __func__);
            continue;
        }
        // Send batches to eval
        auto eval_i = 0;
        while (n_past < inp_size) {
            auto remaining = inp_size - n_past;
            int n_eval = params.n_batch < remaining ? params.n_batch : remaining;
            if (gptneox_eval(ctx, &embd_inp[eval_i], n_eval, n_past, params.n_threads)) {
                fprintf(stderr, "%s : failed to eval\n", __func__);
                return 1;
            }
            n_past += n_eval;
        }
        
        // Eval until space runs out
        auto out_count = 0;
        while (space > 0) {
            // Get token
            gptneox_token id = gptneox_sample_top_p_top_k(
                ctx,
                last_n_tokens.data(),
                params.repeat_last_n,
                top_k,
                top_p,
                temp,
                repeat_penalty);
            // Inc out count and dec space
            out_count += 1;
            space -= 1;
            // Repeat tokens update
            last_n_tokens.push_back(id);
            if (last_n_tokens.size() > params.repeat_last_n) {
                last_n_tokens.erase(last_n_tokens.begin());
            }
            // Check for eos - end early - check eos before bos in case they are the same
            if (id == gptneox_token_eos()) {
                space = 0;
                continue;
            }
            // Check for bos - skip callback if so
            if (id == gptneox_token_bos()) {
                continue;
            }
            // Convert token to string and display
            printf("%s", gptneox_token_to_str(ctx, id));
            fflush(stdout);
            // Check if we need to run another eval
            if (space > 0) {
                // Send generated token back into model for next generation
                if (gptneox_eval(ctx, &id, 1, n_past, params.n_threads)) {
                    fprintf(stderr, "%s : failed to eval\n", __func__);
                    return 1;
                }
                // Increment past count
                n_past += 1;
            }
            // Check for user interrupt
            if (is_interacting) { space = 0; }
        }
    }
    
#if defined (_WIN32)
    signal(SIGINT, SIG_DFL);
#endif

    gptneox_print_timings(ctx);
    gptneox_free(ctx);

    set_console_color(con_st, CONSOLE_COLOR_DEFAULT);

    return 0;
}
    
    
    
    
    


