"""
PROVIDER SCOUT: TOTAL WAR EDITION
Author: Senior Python Dev
Description: Mass verification of ALL possible model IDs for Google, Cohere, SambaNova, OpenRouter.
"""

import os
import sys
import time
from colorama import init, Fore, Style

# --- Установка (раскомментируй, если надо) ---
# os.system("pip install litellm colorama")

from litellm import completion 
import litellm
from keys import KEYS

init(autoreset=True)
litellm.suppress_debug_info = True
litellm.drop_params = True

# ==========================================
# 🚀 ОЧИЩЕННЫЙ СПИСОК ЦЕЛЕЙ (ТОЛЬКО ALIVE И BUSY)
# ==========================================

TARGETS = [

    # ╔══════════════════════════════════════════════════╗
    # ║  5. NEW PROVIDERS (STUBS / PLACEHOLDERS)        ║
    # ╚══════════════════════════════════════════════════╝
    
    # --- OpenAI & Anthropic ---
    {"p": "openai", "n": "GPT-4o",                       "m": "gpt-4o",                                         "k": KEYS["OPENAI"]},
    {"p": "anthropic", "n": "Claude 3.5 Sonnet",         "m": "anthropic/claude-3-5-sonnet-20240620",           "k": KEYS["ANTHROPIC"]},
    
    # --- Groq & Grok ---
    {"p": "groq", "n": "Groq Llama 3.1 70B",             "m": "groq/llama-3.1-70b-versatile",                   "k": KEYS["GROQ"]},
    {"p": "grok", "n": "Grok-1",                         "m": "xai/grok-1",                                     "k": KEYS["GROK"]},
    
    # --- Clouds & APIs ---
    {"p": "hf", "n": "HF Llama 3",                       "m": "huggingface/meta-llama/Meta-Llama-3-8B",          "k": KEYS["HF"]},
    {"p": "github", "n": "GitHub GPT-4o",                "m": "github/gpt-4o",                                  "k": KEYS["GITHUB"]},
    {"p": "together", "n": "Together Llama 3",           "m": "together_ai/meta-llama/Llama-3-70b-hf",          "k": KEYS["TOGETHER"]},
    {"p": "fireworks", "n": "Fireworks Qwen 72B",        "m": "fireworks_ai/qwen-72b",                          "k": KEYS["FIREWORKS"]},
    
    # --- specialized ---
    {"p": "mistral", "n": "Mistral Large",               "m": "mistral/mistral-large-latest",                   "k": KEYS["MISTRAL"]},
    {"p": "cerebras", "n": "Cerebras Llama 70B",         "m": "cerebras/llama3.1-70b",                          "k": KEYS["CEREBRAS"]},
    {"p": "cloudflare", "n": "CF Llama 3",               "m": "cloudflare/@cf/meta/llama-3-8b-instruct",        "k": KEYS["CLOUDFLARE"]},
    {"p": "nvidia", "n": "NVIDIA Nemotron 340B",         "m": "nvidia/nemotron-4-340b-instruct",                "k": KEYS["NVIDIA"]},
    {"p": "ai21", "n": "AI21 Jamba 1.5",                 "m": "ai21/jamba-1.5-large",                           "k": KEYS["AI21"]},
    {"p": "glhf", "n": "GLHF Llama 3",                   "m": "glhf/llama-3-70b",                               "k": KEYS["GLHF"]},

    # --- Local / Self-hosted ---
    {"p": "lms", "n": "LM Studio Local",                 "m": "openai/local-model",                             "k": "not-needed"},
       # ╔══════════════════════════════════════════════╗
    # ║  1. GOOGLE GEMINI (Native через litellm)     ║
    # ╚══════════════════════════════════════════════╝

    # --- Одобренные Gemini 2.5 / 2.0 ---
    {"p": "google", "n": "Gemini 2.5 Flash",              "m": "gemini/gemini-2.5-flash",                        "k": KEYS["GOOGLE"]},
    {"p": "google", "n": "Gemini 2.5 Flash Lite",         "m": "gemini/gemini-2.5-flash-lite",                   "k": KEYS["GOOGLE"]},
    {"p": "google", "n": "Gemini 2.5 Pro",                "m": "gemini/gemini-2.5-pro",                          "k": KEYS["GOOGLE"]},
    {"p": "google", "n": "Gemini 2.0 Flash",              "m": "gemini/gemini-2.0-flash",                        "k": KEYS["GOOGLE"]},
    {"p": "google", "n": "Gemini 2.0 Flash Lite",         "m": "gemini/gemini-2.0-flash-lite",                   "k": KEYS["GOOGLE"]},

    # --- Экспериментальные / Новые серии (3.0 / 3.1) ---
    {"p": "google", "n": "Gemini 3.0 Flash Preview",      "m": "gemini/gemini-3-flash-preview",                  "k": KEYS["GOOGLE"]},
    {"p": "google", "n": "Gemini 3.0 Pro Preview",        "m": "gemini/gemini-3-pro-preview",                    "k": KEYS["GOOGLE"]},
    {"p": "google", "n": "Gemini 3.1 Flash Image Prev",   "m": "gemini/gemini-3.1-flash-image-preview",          "k": KEYS["GOOGLE"]},
    {"p": "google", "n": "Gemini 3.1 Pro Preview",        "m": "gemini/gemini-3.1-pro-preview",                  "k": KEYS["GOOGLE"]},
    {"p": "google", "n": "Gemini 3 Pro Image Prev",       "m": "gemini/gemini-3-pro-image-preview",              "k": KEYS["GOOGLE"]},
    {"p": "google", "n": "Gemini 2.5 Flash Image",        "m": "gemini/gemini-2.5-flash-image",                  "k": KEYS["GOOGLE"]},

    # --- Локальные хосты открытых моделей от Google ---
    {"p": "google", "n": "Gemma 3 27B (Google)",          "m": "gemini/gemma-3-27b-it",                          "k": KEYS["GOOGLE"]},
    {"p": "google", "n": "Gemma 3 4B (Google)",           "m": "gemini/gemma-3-4b-it",                           "k": KEYS["GOOGLE"]},
    {"p": "google", "n": "Gemma 3 1B (Google)",           "m": "gemini/gemma-3-1b-it",                           "k": KEYS["GOOGLE"]},

    # ╔══════════════════════════════════════════════╗
    # ║  2. COHERE (Native через litellm)            ║
    # ╚══════════════════════════════════════════════╝

    {"p": "cohere", "n": "Command A 03-2025",             "m": "cohere/command-a-03-2025",                       "k": KEYS["COHERE"]},
    {"p": "cohere", "n": "Command R+ 08-2024",            "m": "cohere/command-r-plus-08-2024",                  "k": KEYS["COHERE"]},
    {"p": "cohere", "n": "Command R 08-2024",             "m": "cohere/command-r-08-2024",                       "k": KEYS["COHERE"]},
    {"p": "cohere", "n": "Command R7B 12-2024",           "m": "cohere/command-r7b-12-2024",                     "k": KEYS["COHERE"]},
    {"p": "cohere", "n": "Command Nightly",               "m": "cohere/command-nightly",                         "k": KEYS["COHERE"]},
    {"p": "cohere", "n": "Aya Expanse 32B",               "m": "cohere/c4ai-aya-expanse-32b",                    "k": KEYS["COHERE"]},
    {"p": "cohere", "n": "Aya Expanse 8B",                "m": "cohere/c4ai-aya-expanse-8b",                     "k": KEYS["COHERE"]},
    {"p": "cohere", "n": "Command A Translate",           "m": "cohere/command-a-translate-08-2025",             "k": KEYS["COHERE"]},
    {"p": "cohere", "n": "Command A Reasoning",           "m": "cohere/command-a-reasoning-08-2025",             "k": KEYS["COHERE"]},
    {"p": "cohere", "n": "Command A Vision",              "m": "cohere/command-a-vision-07-2025",                "k": KEYS["COHERE"]},
    {"p": "cohere", "n": "Aya Vision 32B",                "m": "cohere/c4ai-aya-vision-32b",                     "k": KEYS["COHERE"]},
    {"p": "cohere", "n": "Aya Vision 8B",                 "m": "cohere/c4ai-aya-vision-8b",                      "k": KEYS["COHERE"]},

    # ╔══════════════════════════════════════════════╗
    # ║  3. SAMBANOVA (OpenAI-compatible)            ║
    # ╚══════════════════════════════════════════════╝

    {"p": "samba", "n": "Samba Llama 4 Maverick",         "m": "Llama-4-Maverick-17B-128E-Instruct",             "k": KEYS["SAMBANOVA"]},
    {"p": "samba", "n": "Samba Llama 3.3 70B",            "m": "Meta-Llama-3.3-70B-Instruct",                    "k": KEYS["SAMBANOVA"]},
    {"p": "samba", "n": "Samba Llama 3.1 8B",             "m": "Meta-Llama-3.1-8B-Instruct",                     "k": KEYS["SAMBANOVA"]},
    {"p": "samba", "n": "Samba DeepSeek V3.1",            "m": "DeepSeek-V3.1",                                  "k": KEYS["SAMBANOVA"]},
    {"p": "samba", "n": "Samba DeepSeek V3",              "m": "DeepSeek-V3-0324",                               "k": KEYS["SAMBANOVA"]},
    {"p": "samba", "n": "Samba DeepSeek R1 Distill 70B",  "m": "DeepSeek-R1-Distill-Llama-70B",                  "k": KEYS["SAMBANOVA"]},
    {"p": "samba", "n": "Samba Qwen3 32B",                "m": "Qwen3-32B",                                      "k": KEYS["SAMBANOVA"]},

    # ╔══════════════════════════════════════════════════╗
    # ║  4. OPENROUTER (Бесплатные :free модели)        ║
    # ╚══════════════════════════════════════════════════╝

    # --- Умный роутер (балансировщик) ---
    {"p": "or", "n": "OR Free Auto-Router",              "m": "openrouter/openrouter/free",                      "k": KEYS["OPENROUTER"]},

    # --- Выжившие бесплатные модели ---
    {"p": "or", "n": "OR Qwen3 4B Free",                 "m": "openrouter/qwen/qwen3-4b:free",                           "k": KEYS["OPENROUTER"]},
    {"p": "or", "n": "OR Qwen3 Coder Free",              "m": "openrouter/qwen/qwen3-coder:free",                        "k": KEYS["OPENROUTER"]},
    {"p": "or", "n": "OR Llama 3.3 70B Free",            "m": "openrouter/meta-llama/llama-3.3-70b-instruct:free",       "k": KEYS["OPENROUTER"]},
    {"p": "or", "n": "OR Llama 3.2 3B Free",             "m": "openrouter/meta-llama/llama-3.2-3b-instruct:free",        "k": KEYS["OPENROUTER"]},
    {"p": "or", "n": "OR Gemma 3 27B Free",              "m": "openrouter/google/gemma-3-27b-it:free",                   "k": KEYS["OPENROUTER"]},
    {"p": "or", "n": "OR Gemma 3 12B Free",              "m": "openrouter/google/gemma-3-12b-it:free",                   "k": KEYS["OPENROUTER"]},
    {"p": "or", "n": "OR Gemma 3 4B Free",               "m": "openrouter/google/gemma-3-4b-it:free",                    "k": KEYS["OPENROUTER"]},
    {"p": "or", "n": "OR Mistral Small 3.1 Free",        "m": "openrouter/mistralai/mistral-small-3.1-24b-instruct:free", "k": KEYS["OPENROUTER"]},
    {"p": "or", "n": "OR Hermes 3 405B Free",            "m": "openrouter/nousresearch/hermes-3-llama-3.1-405b:free",    "k": KEYS["OPENROUTER"]},
    {"p": "or", "n": "OR Trinity Large Free",            "m": "openrouter/arcee-ai/trinity-large-preview:free",          "k": KEYS["OPENROUTER"]},
    {"p": "or", "n": "OR Trinity Mini Free",             "m": "openrouter/arcee-ai/trinity-mini:free",                   "k": KEYS["OPENROUTER"]},
    {"p": "or", "n": "OR Liquid 2.5 1.2B Free",          "m": "openrouter/liquid/lfm-2.5-1.2b-instruct:free",            "k": KEYS["OPENROUTER"]},
    {"p": "or", "n": "OR Dolphin Mistral Venice Free",   "m": "openrouter/cognitivecomputations/dolphin-mistral-24b-venice-edition:free", "k": KEYS["OPENROUTER"]},
    {"p": "or", "n": "OR Step 3.5 Flash Free",           "m": "openrouter/stepfun/step-3.5-flash:free",                  "k": KEYS["OPENROUTER"]},
    {"p": "or", "n": "OR Nemotron Nano 9B V2 Free",      "m": "openrouter/nvidia/nemotron-nano-9b-v2:free", "k": KEYS["OPENROUTER"]},
]

# ==========================================
# 📊 ИТОГО РАБОЧИХ: 49 моделей 
# (Из них 35 со статусом ALIVE и 14 со статусом BUSY)
# ==========================================

print(f"Total targets to test: {len(TARGETS)}")

def run_scout():
    print(f"{Fore.CYAN}╔{'═'*90}╗")
    print(f"║ {Style.BRIGHT}          PROVIDER SCOUT: TOTAL WAR EDITION (CHECKING 20+ MODELS)               {Style.RESET_ALL}{Fore.CYAN} ║")
    print(f"╚{'═'*90}╝\n")

    print(f"{'Prov':<8} | {'Model Name':<20} | {'Status':<10} | {'Latency':<8} | {'Note'}")
    print("-" * 110)

    for t in TARGETS:
        provider = t["p"]
        name = t["n"]
        model = t["m"]
        key = t["k"]

        # Config
        api_base = None
        extra_headers = None
        
        if provider == "samba":
            api_base = "https://api.sambanova.ai/v1"
            model_call = f"openai/{model}"
        elif provider == "or":
            api_base = "https://openrouter.ai/api/v1"
            extra_headers = {"HTTP-Referer": "https://test.loc"}
            model_call = model
        elif provider == "lms":
            api_base = KEYS["LM_STUDIO"]
            model_call = model
        elif provider == "grok":
            api_base = "https://api.x.ai/v1"
            model_call = f"openai/{model}"
        else:
            # Большинство новых провайдеров либо нативно в litellm, 
            # либо следуют OpenAI стандарту при указании ключа
            model_call = model

        start = time.time()
        try:
            if provider == "google":
                os.environ["GEMINI_API_KEY"] = key
                completion(model=model, messages=[{"role": "user", "content": "Hi"}], max_tokens=1)
            elif provider == "cohere":
                os.environ["COHERE_API_KEY"] = key
                completion(model=model, messages=[{"role": "user", "content": "Hi"}], max_tokens=1)
            elif provider == "anthropic":
                os.environ["ANTHROPIC_API_KEY"] = key
                completion(model=model, messages=[{"role": "user", "content": "Hi"}], max_tokens=1)
            else:
                completion(
                    model=model_call, messages=[{"role": "user", "content": "Hi"}], max_tokens=1,
                    api_key=key, api_base=api_base, extra_headers=extra_headers
                )
            
            latency = (time.time() - start) * 1000
            print(f"{provider:<8} | {name:<20} | {Fore.GREEN}ALIVE{Style.RESET_ALL}      | {int(latency)}ms    | Ready!")

        except Exception as e:
            latency = (time.time() - start) * 1000
            err = str(e)
            
            status = f"{Fore.RED}DEAD{Style.RESET_ALL}"
            note = ""

            if "404" in err or "Not Found" in err: 
                note = "Does not exist"
            elif "401" in err or "403" in err:
                note = "Bad Key / No Access"
            elif "429" in err or "Quota" in err: 
                status = f"{Fore.YELLOW}BUSY{Style.RESET_ALL}"
                note = "Rate Limited (Good!)"
            else:
                note = err[:30] + "..."

            print(f"{provider:<8} | {name:<20} | {status:<10} | {int(latency)}ms    | {note}")

    print("\n" + "="*90)
    print(" SUMMARY:")
    print(f" {Fore.GREEN}ALIVE{Style.RESET_ALL} -> Instant access.")
    print(f" {Fore.YELLOW}BUSY {Style.RESET_ALL} -> Good! Just needs Stubborn Mode.")
    print(f" {Fore.RED}DEAD {Style.RESET_ALL} -> Doesn't exist or key invalid.")

if __name__ == "__main__":
    run_scout()
