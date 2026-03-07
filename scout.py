"""
PROVIDER SCOUT: TOTAL WAR EDITION
Author: Senior Python Dev
Description: Mass verification of ALL possible model IDs for Google, Cohere, SambaNova, OpenRouter.
"""

import os
import sys
import time
import json
from colorama import init, Fore, Style

# --- Установка (раскомментируй, если надо) ---
# os.system("pip install litellm colorama")

from litellm import completion 
import litellm

init(autoreset=True)
litellm.suppress_debug_info = True
litellm.drop_params = True

# ==========================================
# 🔑 ЗОНА КЛЮЧЕЙ
# ==========================================
def load_keys(filename="scout_keys.json"):
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"{Fore.RED}Error loading keys from {filename}: {e}{Style.RESET_ALL}")
        return {}

def save_keys(keys_data, filename="scout_keys.json"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(keys_data, f, indent=4, ensure_ascii=False)
        print(f"{Fore.GREEN}Keys successfully saved to {filename}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error saving keys to {filename}: {e}{Style.RESET_ALL}")

KEYS = load_keys()

# ==========================================
# 🚀 ОЧИЩЕННЫЙ СПИСОК ЦЕЛЕЙ (ТОЛЬКО ALIVE И BUSY)
# ==========================================

PROVIDER_TO_KEY = {
    "openai": "OPENAI",
    "anthropic": "ANTHROPIC",
    "groq": "GROQ",
    "grok": "GROK",
    "hf": "HF",
    "github": "GITHUB",
    "together": "TOGETHER",
    "fireworks": "FIREWORKS",
    "mistral": "MISTRAL",
    "cerebras": "CEREBRAS",
    "cloudflare": "CLOUDFLARE",
    "nvidia": "NVIDIA",
    "ai21": "AI21",
    "glhf": "GLHF",
    "lms": "LM_STUDIO",
    "google": "GOOGLE",
    "cohere": "COHERE",
    "samba": "SAMBANOVA",
    "or": "OPENROUTER",
    "polza": "POLZA"
}

def load_models(filename="scout_models.json"):
    models = []
    try:
        with open(filename, "r", encoding="utf-8") as f:
            models_data = json.load(f)
            for t in models_data:
                prov = t.get("p")
                if prov == "lms":
                    t["k"] = "not-needed"
                else:
                    key_name = PROVIDER_TO_KEY.get(prov)
                    t["k"] = KEYS.get(key_name, "EMPTY_KEY")
                models.append(t)
    except Exception as e:
        print(f"{Fore.RED}Error loading {filename}: {e}{Style.RESET_ALL}")
    return models

def save_models(models_data, filename="scout_models.json"):
    # Очищаем поле 'k' перед сохранением, так как оно подставляется динамически
    clean_models = []
    for m in models_data:
        clean_m = {k: v for k, v in m.items() if k != 'k'}
        clean_models.append(clean_m)
        
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(clean_models, f, indent=4, ensure_ascii=False)
        print(f"{Fore.GREEN}Models successfully saved to {filename}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error saving {filename}: {e}{Style.RESET_ALL}")

def load_config(filename="scout_config.json"):
    default_config = {
        "CHECK_PAID_MODELS": False,
        "ENABLED_PROVIDERS": {}
    }
    try:
        with open(filename, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"{Fore.YELLOW}Could not load {filename}, using defaults. ({e}){Style.RESET_ALL}")
        return default_config

def save_config(config_data, filename="scout_config.json"):
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=4, ensure_ascii=False)
        print(f"{Fore.GREEN}Config successfully saved to {filename}{Style.RESET_ALL}")
    except Exception as e:
        print(f"{Fore.RED}Error saving config: {e}{Style.RESET_ALL}")

CONFIG = load_config()
raw_targets = load_models()

TARGETS = []
for t in raw_targets:
    prov = t.get("p")
    # Проверка провайдера, если он есть в конфиге (по умолчанию True)
    if CONFIG.get("ENABLED_PROVIDERS", {}).get(prov, True) == False:
        continue
    
    # Проверка платных моделей
    is_paid = t.get("paid", False)
    if is_paid and not CONFIG.get("CHECK_PAID_MODELS", False):
        continue
        
    TARGETS.append(t)

if not TARGETS:
    print(f"{Fore.YELLOW}No models matched the current config filters or file is empty.{Style.RESET_ALL}")

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
        elif provider == "polza":
            api_base = "https://api.polza.ai/v1"
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