# LLM Chess API Keys
# Shared across all arena_v3_API files
#
# HOW TO USE FOR GITHUB:
#   1. Keep this file with empty strings (safe to commit)
#   2. Run the app -> type 'key' -> enter your real keys
#   3. Keys are saved to keys_local.json (NOT committed to git)
#   4. keys_local.json is in .gitignore

import json
from pathlib import Path

_KEYS_LOCAL_FILE = Path(__file__).parent / "scout_keys.json"

# Default keys — empty strings, safe to upload to GitHub
# Real keys are stored in keys_local.json (gitignored)
_DEFAULT_KEYS = {
    "GOOGLE":     "",
    "COHERE":     "",
    "OPENROUTER": "",
    "SAMBANOVA":  "",
    "GITHUB":     "",
    "CEREBRAS":   "",
}

def _load_keys() -> dict:
    """Load keys from keys_local.json if it exists, otherwise use defaults."""
    if _KEYS_LOCAL_FILE.exists():
        try:
            with open(_KEYS_LOCAL_FILE, "r", encoding="utf-8") as f:
                local = json.load(f)
            merged = dict(_DEFAULT_KEYS)
            merged.update({k: v for k, v in local.items() if k in _DEFAULT_KEYS})
            return merged
        except Exception:
            pass
    return dict(_DEFAULT_KEYS)

def save_keys(keys: dict):
    """Persist current keys to keys_local.json so they survive restarts."""
    try:
        with open(_KEYS_LOCAL_FILE, "w", encoding="utf-8") as f:
            json.dump(keys, f, indent=4)
    except Exception as e:
        print(f"[keys] Warning: could not save keys — {e}")

def check_keys_and_prompt(keys: dict):
    """
    Check if required keys are set. If any are empty, prompt user interactively.
    Returns True if all required keys are now set.
    """
    from colorama import Fore, Style, Back
    required = ["GOOGLE", "COHERE", "OPENROUTER", "SAMBANOVA", "GITHUB", "CEREBRAS"]
    missing = [k for k in required if not keys.get(k, "").strip() and keys.get(k) != "EMPTY_KEY"]
    if not missing:
        return True

    print(f"\n{Back.YELLOW}{Fore.BLACK} API KEYS NOT CONFIGURED {Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Missing keys: {', '.join(missing)}{Style.RESET_ALL}")
    print(f"Keys are stored in {_KEYS_LOCAL_FILE} (gitignored).")
    print("Enter keys now, or press Enter to skip (that provider will be unavailable).\n")
    changed = False
    for k in missing:
        v = input(f"  Enter {k} key (blank to skip): ").strip()
        if v:
            keys[k] = v
            changed = True
    if changed:
        save_keys(keys)
        print(f"{Fore.GREEN}Keys saved to keys_local.json.{Style.RESET_ALL}\n")
    return True

# The shared mutable dict — all files import this reference
KEYS = _load_keys()
