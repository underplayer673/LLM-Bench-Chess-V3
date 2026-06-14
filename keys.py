# Shared API key storage for the LLM chess tools.
# Keep this file safe to commit. Real keys live in keys_local.json.

import json
import os
from pathlib import Path

_KEYS_LOCAL_FILE = Path(__file__).parent / "keys_local.json"
_LEGACY_KEYS_FILE = Path(__file__).parent / "scout_keys.json"

_DEFAULT_KEYS = {
    "GOOGLE": "",
    "COHERE": "",
    "OPENROUTER": "",
    "SAMBANOVA": "",
    "OPENAI": "",
    "ANTHROPIC": "",
    "GROK": "",
    "GROQ": "",
    "GITHUB": "",
    "HF": "",
    "LM_STUDIO": "http://localhost:1234/v1",
    "CEREBRAS": "",
    "MISTRAL": "",
    "CLOUDFLARE": "",
    "TOGETHER": "",
    "FIREWORKS": "",
    "NVIDIA": "",
    "AI21": "",
    "GLHF": "",
    "POLZA": "",
}


def _load_keys() -> dict:
    """Load keys_local.json, then legacy scout_keys.json, then defaults."""
    source = _KEYS_LOCAL_FILE if _KEYS_LOCAL_FILE.exists() else _LEGACY_KEYS_FILE
    if source.exists():
        try:
            with open(source, "r", encoding="utf-8") as f:
                local = json.load(f)
            merged = dict(_DEFAULT_KEYS)
            merged.update({k: v for k, v in local.items() if k in _DEFAULT_KEYS})
            return merged
        except Exception as exc:
            print(f"[keys] Warning: could not load {source}: {exc}")
    return dict(_DEFAULT_KEYS)


def save_keys(keys: dict):
    """Persist current keys to keys_local.json so they survive restarts."""
    try:
        tmp_path = _KEYS_LOCAL_FILE.with_name(f"{_KEYS_LOCAL_FILE.name}.tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(keys, f, indent=4)
        os.replace(tmp_path, _KEYS_LOCAL_FILE)
    except Exception as exc:
        print(f"[keys] Warning: could not save keys: {exc}")


def check_keys_and_prompt(keys: dict):
    """
    Prompt for missing keys at startup. Blank values are allowed; that provider
    will fail fast or be skipped by the user.
    """
    from colorama import Back, Fore, Style

    required = ["GOOGLE", "COHERE", "OPENROUTER", "SAMBANOVA", "GITHUB", "CEREBRAS"]
    missing = [k for k in required if not keys.get(k, "").strip() and keys.get(k) != "EMPTY_KEY"]
    if not missing:
        return True

    print(f"\n{Back.YELLOW}{Fore.BLACK} API KEYS NOT CONFIGURED {Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Missing keys: {', '.join(missing)}{Style.RESET_ALL}")
    print(f"Keys are stored in {_KEYS_LOCAL_FILE} (gitignored).")
    print("Enter keys now, or press Enter to skip that provider.\n")

    changed = False
    for key_name in missing:
        value = input(f"  Enter {key_name} key (blank to skip): ").strip()
        if value:
            keys[key_name] = value
            changed = True
    if changed:
        save_keys(keys)
        print(f"{Fore.GREEN}Keys saved to {_KEYS_LOCAL_FILE.name}.{Style.RESET_ALL}\n")
    return True


# Shared mutable dict. All arena files import this reference.
KEYS = _load_keys()
