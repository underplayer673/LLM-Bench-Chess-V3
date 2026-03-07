"""
LLM CHESS: PROVIDER TOURNAMENT (v32.0 — DEFINITIVE EDITION)
Author: Senior Python Dev & Netrogaty

Features:
    - 4 MODES: User | Economy | Premium (2-stage) | Terminator
    - Dynamic model selection: always picks strongest available
    - Two-stage API calls: Think first, then extract move (Premium mode)
    - In-app config: add/remove models, change keys, modes, prompts
    - Auto-failover chains per provider
    - Echo filter, accumulated failed moves, broad scan
    - PGN with model annotations
"""

import os
import sys
import time
import json
import datetime
import re
import shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple
from enum import Enum
from contextlib import contextmanager

import chess
import chess.pgn
import litellm
from litellm import completion
from colorama import init, Fore, Style, Back

init(autoreset=True)
litellm.suppress_debug_info = True
litellm.drop_params = True

@contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        try: sys.stderr = devnull
        finally: sys.stderr = old_stderr
        yield

# ==========================================
# 🔑 KEYS (imported from keys.py — shared & persistent)
# ==========================================

from keys import KEYS, save_keys

os.environ["GEMINI_API_KEY"] = KEYS["GOOGLE"]
os.environ["COHERE_API_KEY"] = KEYS["COHERE"]

# ==========================================
# 🎮 GAME MODES
# ==========================================

class GameMode(Enum):
    USER = "user"           # Just FEN + "best move?" — not recommended
    ECONOMY = "economy"     # Single smart call — good
    PREMIUM = "premium"     # Two-stage: think + extract — best (default)
    TERMINATOR = "terminator"  # Shows legal moves — not recommended
    ULTRA_TERMINATOR = "ultra-terminator"  # Premium think/extract with legal moves

CURRENT_MODE = GameMode.PREMIUM

class SelectionSubMode(Enum):
    BALANCED_MAX = "balanced"
    FULL_MAX = "fullmax"
    CLASSIC = "classic"

class OpenRouterMode(Enum):
    AUTO = "auto"
    CUSTOM = "custom"

CURRENT_SELECTION_SUBMODE = SelectionSubMode.FULL_MAX
CURRENT_OPENROUTER_MODE = OpenRouterMode.CUSTOM

MODE_DESCRIPTIONS = {
    GameMode.USER: "Basic: just FEN, ask for move. Fast but dumb. NOT recommended.",
    GameMode.ECONOMY: "Smart single call with piece lists. Good balance.",
    GameMode.PREMIUM: "Two-stage: AI thinks first, then outputs move. BEST quality. DEFAULT.",
    GameMode.TERMINATOR: "Shows legal moves to AI. Strongest but unfair. NOT recommended.",
    GameMode.ULTRA_TERMINATOR: "Premium think + extract, but with legal moves shown. Strongest practical mode.",
}

SELECTION_SUBMODE_DESCRIPTIONS = {
    SelectionSubMode.BALANCED_MAX: "Prefer strong models first, then use the rest if needed.",
    SelectionSubMode.FULL_MAX: "Prefer the strongest and newest experimental models first, then use the rest.",
    SelectionSubMode.CLASSIC: "On every move, scan the full roster from smartest to weakest and skip failures immediately.",
}

# ==========================================
# 🏢 PROVIDER CHAINS (editable in-app)
# ==========================================

PROVIDER_CHAINS = {
    "Team GitHub": {
        "provider": "github",
        "chain": [
            {"name": "GitHub GPT-4o",               "model": "github/gpt-4o",                                 "balanced": True,  "full_max": True,  "experimental": True},
            {"name": "GitHub GPT-4o Mini",          "model": "github/gpt-4o-mini",                            "balanced": True,  "full_max": False},
        ],
    },
    "Team Cerebras": {
        "provider": "cerebras",
        "chain": [
            {"name": "Cerebras Llama3.1 8B",       "model": "cerebras/llama3.1-8b",                         "balanced": True,  "full_max": True},
        ],
    },
    "Team Google": {
        "provider": "google",
        "chain": [
            {"name": "Gemini 3.1 Pro Preview",      "model": "gemini/gemini-3.1-pro-preview",                 "balanced": True,  "full_max": True,  "experimental": True},
            {"name": "Gemini 3.0 Pro Preview",      "model": "gemini/gemini-3-pro-preview",                   "balanced": True,  "full_max": True,  "experimental": True},
            {"name": "Gemini 3.0 Flash Preview",    "model": "gemini/gemini-3-flash-preview",                 "balanced": True,  "full_max": True,  "experimental": True},
            {"name": "Gemini 2.5 Pro",              "model": "gemini/gemini-2.5-pro",                         "balanced": True,  "full_max": True},
            {"name": "Gemini 2.5 Flash",            "model": "gemini/gemini-2.5-flash",                       "balanced": True,  "full_max": False},
            {"name": "Gemini 3 Pro Image Prev",     "model": "gemini/gemini-3-pro-image-preview",             "balanced": False, "full_max": True,  "experimental": True},
            {"name": "Gemini 3.1 Flash Image Prev", "model": "gemini/gemini-3.1-flash-image-preview",         "balanced": False, "full_max": True,  "experimental": True},
            {"name": "Gemini 2.5 Flash Image",      "model": "gemini/gemini-2.5-flash-image",                 "balanced": False, "full_max": False},
            {"name": "Gemini 2.5 Flash Lite",       "model": "gemini/gemini-2.5-flash-lite",                  "balanced": True,  "full_max": False},
            {"name": "Gemini 2.0 Flash",            "model": "gemini/gemini-2.0-flash",                       "balanced": False, "full_max": False},
            {"name": "Gemini 2.0 Flash Lite",       "model": "gemini/gemini-2.0-flash-lite",                  "balanced": False, "full_max": False},
            {"name": "Gemma 3 27B (Google)",        "model": "gemini/gemma-3-27b-it",                         "balanced": True,  "full_max": False},
            {"name": "Gemma 3 4B (Google)",         "model": "gemini/gemma-3-4b-it",                          "balanced": False, "full_max": False},
            {"name": "Gemma 3 1B (Google)",         "model": "gemini/gemma-3-1b-it",                          "balanced": False, "full_max": False},
        ],
    },
    "Team OpenRouter": {
        "provider": "or",
        "chain": [
            {"name": "OR Free Auto-Router",         "model": "openrouter/openrouter/free",                                         "router": True,  "balanced": True,  "full_max": True},
            {"name": "OR Hermes 3 405B Free",      "model": "openrouter/nousresearch/hermes-3-llama-3.1-405b:free",               "balanced": True,  "full_max": True,  "experimental": True},
            {"name": "OR Trinity Large Free",      "model": "openrouter/arcee-ai/trinity-large-preview:free",                     "balanced": True,  "full_max": True,  "experimental": True},
            {"name": "OR Llama 3.3 70B Free",      "model": "openrouter/meta-llama/llama-3.3-70b-instruct:free",                  "balanced": True,  "full_max": True},
            {"name": "OR Mistral Small 3.1 Free",  "model": "openrouter/mistralai/mistral-small-3.1-24b-instruct:free",          "balanced": True,  "full_max": False},
            {"name": "OR Gemma 3 27B Free",        "model": "openrouter/google/gemma-3-27b-it:free",                              "balanced": True,  "full_max": False},
            {"name": "OR Qwen3 Coder Free",        "model": "openrouter/qwen/qwen3-coder:free",                                   "balanced": True,  "full_max": False},
            {"name": "OR Trinity Mini Free",       "model": "openrouter/arcee-ai/trinity-mini:free",                              "balanced": False, "full_max": False},
            {"name": "OR Step 3.5 Flash Free",     "model": "openrouter/stepfun/step-3.5-flash:free",                             "balanced": False, "full_max": False},
            {"name": "OR Gemma 3 4B Free",         "model": "openrouter/google/gemma-3-4b-it:free",                               "balanced": False, "full_max": False},
            {"name": "OR Qwen3 4B Free",           "model": "openrouter/qwen/qwen3-4b:free",                                      "balanced": False, "full_max": False},
            {"name": "OR Llama 3.2 3B Free",       "model": "openrouter/meta-llama/llama-3.2-3b-instruct:free",                   "balanced": False, "full_max": False},
            {"name": "OR Nemotron Nano 9B V2 Free","model": "openrouter/nvidia/nemotron-nano-9b-v2:free",                         "balanced": False, "full_max": False},
            {"name": "OR Dolphin Mistral Venice",  "model": "openrouter/cognitivecomputations/dolphin-mistral-24b-venice-edition:free", "balanced": False, "full_max": False},
            {"name": "OR Liquid 2.5 1.2B Free",    "model": "openrouter/liquid/lfm-2.5-1.2b-instruct:free",                       "balanced": False, "full_max": False},
        ],
    },
}

TEAM_NAMES = list(PROVIDER_CHAINS.keys())

# ==========================================
# 🧠 PROVIDER TEAM
# ==========================================

class ProviderTeam:
    def __init__(self, team_name):
        cfg = PROVIDER_CHAINS[team_name]
        self.name = team_name
        self.provider = cfg["provider"]
        self.chain = list(cfg["chain"])
        self.current_index = 0
        self.exhausted_until: Dict[int, float] = {}
        self.switch_log: List[dict] = []
        self.total_switches = 0

    @property
    def current_model(self):
        return self.chain[min(self.current_index, len(self.chain)-1)]

    @property
    def current_model_name(self):
        return self.current_model["name"]

    def _model_elo(self, index):
        return ensure_model_stats(self.name, self.chain[index])["elo"]

    def _base_ranked_indices(self):
        indices = list(range(len(self.chain)))
        if self.provider == "or":
            router = [i for i, model in enumerate(self.chain) if model.get("router")]
            non_router = [i for i in indices if i not in router]
            if CURRENT_OPENROUTER_MODE == OpenRouterMode.AUTO:
                return router or indices
            indices = non_router or indices

        if CURRENT_SELECTION_SUBMODE == SelectionSubMode.CLASSIC:
            return indices

        if CURRENT_SELECTION_SUBMODE == SelectionSubMode.FULL_MAX:
            preferred = [i for i in indices if self.chain[i].get("full_max")]
        else:
            preferred = [i for i in indices if self.chain[i].get("balanced")]

        remaining = [i for i in indices if i not in preferred]
        preferred.sort(key=lambda i: (-self._model_elo(i), i))
        remaining.sort(key=lambda i: (-self._model_elo(i), i))
        return preferred + remaining

    def get_ranked_indices(self):
        now = time.time()
        available = []
        blocked = []
        for i in self._base_ranked_indices():
            expiry = self.exhausted_until.get(i, 0)
            if expiry and expiry >= now:
                blocked.append(i)
            else:
                self.exhausted_until.pop(i, None)
                available.append(i)
        return available + blocked

    def get_best_available(self):
        """Get the best currently available model according to the active submode."""
        now = time.time()
        for i in self.get_ranked_indices():
            if i not in self.exhausted_until or self.exhausted_until[i] < now:
                self.exhausted_until.pop(i, None)
                if i != self.current_index:
                    old = self.current_model_name
                    self.current_index = i
                    if old != self.current_model_name:
                        self.switch_log.append({"from": old, "to": self.current_model_name,
                                                "reason": "best_available",
                                                "time": datetime.datetime.now().isoformat()})
                return self.current_model
        return None

    def mark_exhausted(self, cooldown=300, reason="exhausted"):
        old = self.current_model_name
        self.exhausted_until[self.current_index] = time.time() + cooldown
        now = time.time()
        for i in self.get_ranked_indices():
            if i not in self.exhausted_until or self.exhausted_until[i] < now:
                self.current_index = i
                self.total_switches += 1
                self.switch_log.append({"from": old, "to": self.current_model_name,
                                        "reason": reason,
                                        "time": datetime.datetime.now().isoformat()})
                print(f"\n{Back.YELLOW}{Fore.BLACK} ⚡ {self.name}: {old} → {self.current_model_name} {Style.RESET_ALL}")
                return True
        print(f"\n{Back.RED}{Fore.WHITE} 💀 {self.name}: ALL EXHAUSTED {Style.RESET_ALL}")
        return False

    def to_dict(self):
        return {"name": self.name, "current_index": self.current_index,
                "exhausted_until": {str(k): v for k, v in self.exhausted_until.items()},
                "switch_log": self.switch_log, "total_switches": self.total_switches}

    @staticmethod
    def from_dict(data):
        t = ProviderTeam(data["name"])
        t.current_index = data.get("current_index", 0)
        t.exhausted_until = {int(k): v for k, v in data.get("exhausted_until", {}).items()}
        t.switch_log = data.get("switch_log", [])
        t.total_switches = data.get("total_switches", 0)
        return t

# ==========================================
# 🔌 API ROUTER
# ==========================================

def call_model_direct(provider, model_id, messages, temperature=0.2, max_tokens=500):
    args = {"messages": messages, "timeout": 90, "max_tokens": max_tokens, "temperature": temperature}
    if provider == "google":
        args["model"] = model_id
    elif provider == "cohere":
        args["model"] = model_id
    elif provider == "samba":
        args["model"] = f"openai/{model_id}"
        args["api_base"] = "https://api.sambanova.ai/v1"
        args["api_key"] = KEYS.get("SAMBANOVA", "")
    elif provider == "github":
        args["model"] = model_id
        args["api_base"] = "https://models.inference.ai.azure.com"
        args["api_key"] = KEYS.get("GITHUB", "")
    elif provider == "cerebras":
        args["model"] = model_id
        args["api_base"] = "https://api.cerebras.ai/v1"
        args["api_key"] = KEYS.get("CEREBRAS", "")
    elif provider == "or":
        args["model"] = model_id
        args["api_base"] = "https://openrouter.ai/api/v1"
        args["api_key"] = KEYS.get("OPENROUTER", "")
        args["extra_headers"] = {"HTTP-Referer": "https://chess.loc", "X-Title": "Chess"}
    else:
        raise ValueError(f"Unknown: {provider}")
    return completion(**args)

def strip_think_tags(text):
    """Remove DeepSeek's <think>...</think> reasoning blocks, keep only the actual answer."""
    if not text:
        return text
    # Remove all <think>...</think> blocks (can be multiple, may have content or be empty)
    cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Also handle unclosed <think> tags (just in case)
    cleaned = re.sub(r'<think>.*$', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    return cleaned.strip()

def api_call_with_failover(team, messages, telemetry, temperature=0.2, max_tokens=500):
    """Call API with automatic failover. Returns (content, model_name) or (None, None)."""
    # Give SambaNova (DeepSeek) more tokens for their <think> reasoning
    actual_max_tokens = max_tokens + 800 if team.provider == "samba" else max_tokens

    attempts = 0
    max_attempts = max(len(team.chain) * 2, 10)
    while attempts < max_attempts:
        active = team.get_best_available()
        if not active:
            time.sleep(Config.DELAY_NO_ACTIVE)
            active = team.get_best_available()
            if not active:
                return None, None
        try:
            with suppress_stderr():
                resp = call_model_direct(team.provider, active["model"], messages,
                                         temperature=temperature, max_tokens=actual_max_tokens)
            content = resp.choices[0].message.content
            if not content or not str(content).strip():
                raise RuntimeError("empty response")
            # Strip <think> blocks from DeepSeek responses
            if team.provider == "samba":
                content = strip_think_tags(content)
                if not content:
                    raise RuntimeError("empty response after think-strip")
            if active["name"] not in telemetry.models_used:
                telemetry.models_used.append(active["name"])
            return content, active["name"]
        except Exception as e:
            err = str(e).lower()
            telemetry.api_errors += 1
            attempts += 1
            is_limit = any(x in err for x in ["429","quota","rate limit","busy","resource_exhausted","too many","overloaded"])
            reason = "rate_limit_skip" if is_limit else "error_skip"
            cooldown = Config.COOLDOWN_RATE_LIMIT if is_limit else Config.COOLDOWN_ERROR
            team.mark_exhausted(cooldown, reason=reason)
            telemetry.model_switches += 1
            if not is_limit:
                time.sleep(Config.DELAY_API_RETRY)
    return None, None

# ==========================================
# ♟️ POSITION HELPERS
# ==========================================

def get_piece_list(board, color):
    names = {chess.KING:"King", chess.QUEEN:"Queen", chess.ROOK:"Rook",
             chess.BISHOP:"Bishop", chess.KNIGHT:"Knight", chess.PAWN:"Pawn"}
    groups = {}
    for sq in chess.SQUARES:
        p = board.piece_at(sq)
        if p and p.color == color:
            groups.setdefault(names[p.piece_type], []).append(chess.square_name(sq))
    return "\n".join(f"  {k}{'s' if len(groups[k])>1 else ''}: {', '.join(groups[k])}"
                     for k in ["King","Queen","Rook","Bishop","Knight","Pawn"] if k in groups)

def get_status(board):
    lines = []
    if board.is_check(): lines.append("⚠️ YOUR KING IS IN CHECK!")
    c = board.turn
    cast = []
    if board.has_kingside_castling_rights(c): cast.append("O-O")
    if board.has_queenside_castling_rights(c): cast.append("O-O-O")
    lines.append(f"Castling: {', '.join(cast) or 'NONE'}")
    if board.ep_square: lines.append(f"En passant: {chess.square_name(board.ep_square)}")
    return "\n".join(lines)

def get_history(board, n=10):
    if not board.move_stack: return "Game start"
    sans, tb = [], chess.Board()
    for i, m in enumerate(board.move_stack):
        s = tb.san(m)
        sans.append(f"{i//2+1}. {s}" if i%2==0 else s)
        tb.push(m)
    return " ".join(sans[-n:])

def get_opp_last(board):
    if not board.move_stack: return None
    t = board.copy(); mv = t.pop()
    return t.san(mv)

def get_legal_moves_str(board, max_show=30):
    moves = [board.san(m) for m in board.legal_moves]
    if len(moves) > max_show:
        return ", ".join(moves[:max_show]) + f"... ({len(moves)} total)"
    return ", ".join(moves)

# ==========================================
# 🎯 PROMPT BUILDERS (per mode)
# ==========================================

def build_prompt_user(board, color, failed=None):
    """USER mode: minimal, just FEN."""
    msg = f"FEN: {board.fen()}\nYou are {color}. Best move in SAN?\nMOVE:"
    if failed: msg = f"FEN: {board.fen()}\nYou are {color}. NOT these: {', '.join(failed)}.\nDifferent move?\nMOVE:"
    return [{"role": "user", "content": msg}]

def build_prompt_economy(board, color, failed=None):
    """ECONOMY mode: detailed single call."""
    opp = "Black" if color == "White" else "White"
    cb = chess.WHITE if color == "White" else chess.BLACK
    sys_msg = (f"You are a chess grandmaster as {color}. "
               "Respond with 1 sentence analysis then MOVE: <SAN> on last line. "
               "Examples: MOVE: Ba4, MOVE: Nf3, MOVE: O-O. "
               "Do NOT echo opponent's move as yours.")
    user_lines = [f"Move {board.fullmove_number}. You are {color.upper()}.",
                  "", f"YOUR pieces:", get_piece_list(board, cb),
                  "", f"OPPONENT pieces:", get_piece_list(board, not cb),
                  "", get_status(board), "", f"Game: {get_history(board)}"]
    ol = get_opp_last(board)
    if ol: user_lines.append(f"\nOpponent played: {ol}. Do NOT repeat it.")
    if failed: user_lines.append(f"\n⚠️ ILLEGAL (tried): {', '.join(failed)}. Pick DIFFERENT.")
    user_lines.append("\nMOVE:")
    return [{"role": "system", "content": sys_msg},
            {"role": "user", "content": "\n".join(user_lines)}]

def build_prompt_premium_think(board, color):
    """PREMIUM mode, Stage 1: Think about the position."""
    opp = "Black" if color == "White" else "White"
    cb = chess.WHITE if color == "White" else chess.BLACK
    sys_msg = (f"You are a chess grandmaster playing as {color}. "
               "Analyze the position and decide on the BEST move. "
               "Explain your reasoning in 2-3 sentences. "
               "At the end, state clearly which move you choose.")
    user_lines = [f"Move {board.fullmove_number}. You are {color.upper()}.",
                  "", f"YOUR pieces:", get_piece_list(board, cb),
                  "", f"OPPONENT:", get_piece_list(board, not cb),
                  "", get_status(board), "", f"Game: {get_history(board)}"]
    ol = get_opp_last(board)
    if ol: user_lines.append(f"\nOpponent just played: {ol}")
    user_lines.append("\nAnalyze and choose your best move:")
    return [{"role": "system", "content": sys_msg},
            {"role": "user", "content": "\n".join(user_lines)}]

def build_prompt_ultra_terminator_think(board, color, failed=None):
    """ULTRA TERMINATOR mode, Stage 1: Think with legal moves visible."""
    cb = chess.WHITE if color == "White" else chess.BLACK
    legal = get_legal_moves_str(board)
    sys_msg = (f"You are a chess grandmaster playing as {color}. "
               "You are given the legal moves list. Pick the strongest move from that list only. "
               "Explain your reasoning in 2-3 sentences, then clearly state the final move.")
    user_lines = [f"Move {board.fullmove_number}. You are {color.upper()}.",
                  "", f"YOUR pieces:", get_piece_list(board, cb),
                  "", f"OPPONENT:", get_piece_list(board, not cb),
                  "", get_status(board), "", f"Game: {get_history(board)}",
                  "", f"LEGAL MOVES: {legal}"]
    ol = get_opp_last(board)
    if ol: user_lines.append(f"\nOpponent just played: {ol}")
    if failed: user_lines.append(f"\nDo NOT choose these failed moves: {', '.join(failed)}")
    user_lines.append("\nAnalyze and choose the best legal move:")
    return [{"role": "system", "content": sys_msg},
            {"role": "user", "content": "\n".join(user_lines)}]

def build_prompt_premium_extract(analysis_text, color, failed=None):
    """PREMIUM mode, Stage 2: Extract just the move from analysis."""
    msg = (f"You analyzed a chess position as {color} and wrote:\n"
           f'"{analysis_text}"\n\n'
           "What move did you choose? Reply with ONLY the move in SAN notation. "
           "Nothing else. Just the move. Example: Nf3")
    if failed:
        msg += f"\n\nNOT these (illegal): {', '.join(failed)}. Pick a DIFFERENT move."
    return [{"role": "user", "content": msg}]

def build_prompt_terminator(board, color, failed=None):
    """TERMINATOR mode: shows legal moves."""
    opp = "Black" if color == "White" else "White"
    cb = chess.WHITE if color == "White" else chess.BLACK
    legal = get_legal_moves_str(board)
    sys_msg = (f"You are {color}. Pick the BEST move from the legal moves list. "
               "Reply with ONLY: MOVE: <SAN>")
    user_lines = [f"Move {board.fullmove_number}. {color.upper()} to move.",
                  "", f"YOUR pieces:", get_piece_list(board, cb),
                  "", f"OPPONENT:", get_piece_list(board, not cb),
                  "", get_status(board), "", f"Game: {get_history(board)}",
                  "", f"LEGAL MOVES: {legal}"]
    if failed: user_lines.append(f"\n⚠️ NOT these: {', '.join(failed)}")
    user_lines.append("\nMOVE:")
    return [{"role": "system", "content": sys_msg},
            {"role": "user", "content": "\n".join(user_lines)}]

# ==========================================
# 🔍 MOVE EXTRACTION v3.0
# ==========================================

NOISE = {"MOVE","SAN","UCI","FINAL","CHESS","THE","IS","MY","BEST","THINK","BLACK","WHITE",
         "BISHOP","KNIGHT","ROOK","QUEEN","KING","PAWN","RETREAT","TRADE","ATTACK","CAPTURE",
         "CHECK","THAT","THIS","WITH","FROM","WILL","WOULD","SHOULD","AFTER","SQUARE","PIECE",
         "OPPONENT","POSITION","STRONG","ANALYZE","RESPONSE","PLAY","PLAYED","I","AND","TO","IN"}

def clean_tok(val):
    if not val: return ""
    c = re.sub(r'[*`\[\]().,;!:\"]', '', val.strip())
    c = re.sub(r'^\d+[\.\s]*', '', c).strip()
    m = re.search(r'([a-zA-Z0-9+#=xO-]+)', c)
    if not m: return ""
    r = m.group(1)
    if r.upper() in NOISE: return ""
    hf = any(ch in 'abcdefgh' for ch in r.lower())
    hr = any(ch in '12345678' for ch in r)
    ic = "O-O" in r.upper() or "0-0" in r.upper()
    return r if (ic or (hf and hr)) else ""

def normalize_promotion(move_text):
    if not move_text:
        return ""
    m = re.match(r'^([a-h](?:x[a-h])?[18])=?([qrbnQRBN])([+#]?)$', move_text.strip())
    if not m:
        return ""
    prefix, piece, suffix = m.groups()
    return f"{prefix}={piece.upper()}{suffix}"

def extract_move(text, opp_last=None):
    def echo(m): return opp_last and m == opp_last

    # P0: Pawn promotion must win over broader parsers
    for promo in reversed(re.findall(r'\b([a-h](?:x[a-h])?[18]=?[qrbnQRBN][+#]?)\b', text)):
        c = normalize_promotion(promo)
        if c and not echo(c): return c

    # P1: MOVE: tag
    for t in reversed(re.findall(r'(?:MOVE|Move|move)\s*:\s*\*?\*?\s*([A-Za-z0-9+#=xO-]+)', text)):
        c = clean_tok(t)
        if c and not echo(c): return c

    # P2: Bold
    for b in reversed(re.findall(r'\*\*([A-Za-z0-9+#=xO-]+)\*\*', text)):
        c = clean_tok(b)
        if c and not echo(c): return c

    # P3: Piece moves (not bare squares)
    for p in reversed(re.findall(r'\b([KQRBN][a-h]?[1-8]?x?[a-h][1-8][+#]?|[a-h]x[a-h][1-8](?:=[QRBN])?[+#]?|O-O(?:-O)?)\b', text)):
        c = clean_tok(p)
        if c and not echo(c): return c

    # P4: Pawn from last 3 lines
    for p in reversed(re.findall(r'\b([a-h][1-8])\b', "\n".join(text.split('\n')[-3:]))):
        c = clean_tok(p)
        if c and not echo(c): return c

    # P5: UCI
    for u in reversed(re.findall(r'\b([a-h][1-8][a-h][1-8][qrbn]?)\b', text)):
        c = clean_tok(u)
        if c and not echo(c): return c

    # P6: Last words
    for w in reversed(text.split()):
        c = clean_tok(w)
        if c and 2 <= len(c) <= 7 and not echo(c): return c
    return ""

def broad_scan(content, board):
    for c in re.findall(r'\b(?:[KQRBN][a-h]?[1-8]?x?[a-h][1-8][+#]?|[a-h]x[a-h][1-8](?:=[QRBN])?[+#]?|[a-h][1-8](?:=[QRBN])?[+#]?|O-O(?:-O)?|[a-h][1-8][a-h][1-8][qrbn]?)\b', content):
        cl = c.replace("!","").replace("?","").strip()
        try: board.parse_san(cl); return cl
        except:
            try:
                mv = chess.Move.from_uci(cl)
                if mv in board.legal_moves: return cl
            except: pass
    return None

# ==========================================
# ⚙️ CONFIG
# ==========================================

class Config:
    BASE_DIR = Path("tournament_data")
    STATE_FILE = BASE_DIR / "tournament_state.json"
    STATS_FILE = BASE_DIR / "detailed_analytics.json"
    TOTAL_ROUNDS = 3
    CONTINUOUS_TOURNAMENTS = True
    MODEL_ELO_START = 1200.0
    MODEL_ELO_K = 24.0
    DELAY_BETWEEN_MOVES = 2.0
    DELAY_API_RETRY = 1.0
    DELAY_ALL_EXHAUSTED = 60.0
    DELAY_NO_ACTIVE = 15.0
    COOLDOWN_RATE_LIMIT = 340
    COOLDOWN_ERROR = 120
    COOLDOWN_MODEL_FAIL = 600

MODEL_STATS: Dict[str, dict] = {}

# ==========================================
# 🏆 TEAM ELO (separate from model ELO)
# ==========================================

TEAM_ELO: Dict[str, dict] = {}

def ensure_team_elo(team_name: str) -> dict:
    if team_name not in TEAM_ELO:
        TEAM_ELO[team_name] = {
            "name": team_name,
            "elo": Config.MODEL_ELO_START,
            "games": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
        }
    return TEAM_ELO[team_name]

def apply_team_elo_result(team_a: str, team_b: str, score_a: float):
    """Update team ELO ratings after a game. score_a: 1.0=win, 0.5=draw, 0.0=loss."""
    a = ensure_team_elo(team_a)
    b = ensure_team_elo(team_b)
    score_b = 1.0 - score_a
    exp_a = expected_score(a["elo"], b["elo"])
    exp_b = expected_score(b["elo"], a["elo"])
    a["elo"] += Config.MODEL_ELO_K * (score_a - exp_a)
    b["elo"] += Config.MODEL_ELO_K * (score_b - exp_b)
    a["games"] += 1; b["games"] += 1
    if score_a == 1.0:
        a["wins"] += 1; b["losses"] += 1
    elif score_a == 0.0:
        a["losses"] += 1; b["wins"] += 1
    else:
        a["draws"] += 1; b["draws"] += 1


def make_model_key(team_name: str, model_entry: dict) -> str:
    return f"{team_name}::{model_entry.get('name', 'Unknown')}::{model_entry.get('model', 'unknown')}"

def ensure_model_stats(team_name: str, model_entry: dict) -> dict:
    key = make_model_key(team_name, model_entry)
    if key not in MODEL_STATS:
        MODEL_STATS[key] = {
            "key": key,
            "team": team_name,
            "name": model_entry.get("name", "Unknown"),
            "model": model_entry.get("model", "unknown"),
            "elo": Config.MODEL_ELO_START,
            "games": 0,
            "wins": 0,
            "draws": 0,
            "losses": 0,
            "illegal_moves": 0,
        }
    return MODEL_STATS[key]

def expected_score(elo_a: float, elo_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((elo_b - elo_a) / 400.0))

def apply_elo_result(model_a: dict, model_b: dict, score_a: float):
    """Classic 1v1 ELO update (kept for compatibility)."""
    score_b = 1.0 - score_a
    exp_a = expected_score(model_a["elo"], model_b["elo"])
    exp_b = expected_score(model_b["elo"], model_a["elo"])
    model_a["elo"] += Config.MODEL_ELO_K * (score_a - exp_a)
    model_b["elo"] += Config.MODEL_ELO_K * (score_b - exp_b)
    model_a["games"] += 1
    model_b["games"] += 1
    if score_a == 1.0:
        model_a["wins"] += 1; model_b["losses"] += 1
    elif score_a == 0.0:
        model_a["losses"] += 1; model_b["wins"] += 1
    else:
        model_a["draws"] += 1; model_b["draws"] += 1

def apply_weighted_model_elo(side_history: list, score: float, opp_avg_elo: float):
    """
    Distribute ELO changes to ALL models that participated,
    weighted by number of moves each model made.

    Example: Model A made 10 moves, Model B made 20 moves (total 30).
    Model A gets 33% of K * (score - expected), Model B gets 67%.
    This ensures the smart early-game model gets rightful credit
    even if a weaker model took over after rate-limiting.
    """
    if not side_history:
        return
    # Count moves per model
    from collections import Counter
    move_counts = Counter(side_history)
    total_moves = len(side_history)

    for model_key, move_count in move_counts.items():
        if model_key not in MODEL_STATS:
            continue
        m = MODEL_STATS[model_key]
        weight = move_count / total_moves  # fraction of game this model played
        exp = expected_score(m["elo"], opp_avg_elo)
        m["elo"] += Config.MODEL_ELO_K * weight * (score - exp)
        m["games"] += 1
        if score == 1.0:
            m["wins"] += 1
        elif score == 0.0:
            m["losses"] += 1
        else:
            m["draws"] += 1

@dataclass
class TeamTelemetry:
    name: str
    points: float = 0.0
    wins: int = 0; draws: int = 0; losses: int = 0
    total_moves: int = 0; total_think_time: float = 0.0
    api_errors: int = 0; model_switches: int = 0; illegal_moves: int = 0
    stage1_calls: int = 0; stage2_calls: int = 0
    models_used: List[str] = field(default_factory=list)
    played_opponents: List[str] = field(default_factory=list)
    color_history: List[str] = field(default_factory=list)
    is_disqualified: bool = False
    @property
    def avg_time(self): return self.total_think_time / max(self.total_moves, 1)
    @property
    def total_attempts(self): return self.total_moves + self.illegal_moves
    @property
    def illegal_rate(self): return self.illegal_moves / max(self.total_attempts, 1)
    def to_dict(self): return asdict(self)
    @staticmethod
    def from_dict(data):
        valid = {f.name for f in TeamTelemetry.__dataclass_fields__.values()}
        return TeamTelemetry(**{k: v for k, v in data.items() if k in valid})

class MatchState(Enum):
    ACTIVE = "Active"; COMPLETED = "Completed"

# ==========================================
# ♟️ THE CHESS MATCH ENGINE
# ==========================================

class ChessMatch:
    def __init__(self, table_id, round_num, white, black, tournament_cycle=1):
        self.table_id = table_id; self.round_num = round_num
        self.white_name = white; self.black_name = black
        self.tournament_cycle = tournament_cycle
        self.board = chess.Board()
        self.state = MatchState.ACTIVE; self.forced_result = None
        self.annotations: List[str] = []
        self.white_model_history: List[str] = []
        self.black_model_history: List[str] = []
        self.pod_dir = Config.BASE_DIR / f"pod_tournament_data_{self.tournament_cycle}"
        self.pgn_dir = self.pod_dir / "pgn" / f"Round_{round_num}"
        self.pgn_dir.mkdir(parents=True, exist_ok=True)
        self.pgn_path = self.pgn_dir / f"T{table_id}_{white}_vs_{black}.pgn"

    def _register_move_model(self, team_name, model_entry):
        record = ensure_model_stats(team_name, model_entry)
        if team_name == self.white_name:
            self.white_model_history.append(record["key"])
        elif team_name == self.black_name:
            self.black_model_history.append(record["key"])

    def _primary_model_key(self, team_name):
        history = self.white_model_history if team_name == self.white_name else self.black_model_history
        if not history:
            return None
        counts = {}
        for key in history:
            counts[key] = counts.get(key, 0) + 1
        best_key = history[-1]
        best_score = (counts[best_key], len(history) - 1 - history[::-1].index(best_key))
        for key, count in counts.items():
            score = (count, len(history) - 1 - history[::-1].index(key))
            if score > best_score:
                best_key = key
                best_score = score
        return best_key

    def make_ai_move(self, teams_db, provider_teams, mode):
        color = "White" if self.board.turn == chess.WHITE else "Black"
        tn = self.white_name if self.board.turn == chess.WHITE else self.black_name
        team = provider_teams[tn]; tel = teams_db[tn]
        if tel.is_disqualified: return False

        MAX_ATTEMPTS = 5; MAX_SWITCHES = 3
        temps = [0.2, 0.3, 0.5, 0.7, 0.9]
        t0 = time.time(); opp_last = get_opp_last(self.board)
        printed = False; switches = 0

        while switches < MAX_SWITCHES:
            active = team.get_best_available()
            if not active:
                print(f"{Back.RED} 💀 {tn}: All exhausted. Wait {Config.DELAY_ALL_EXHAUSTED}s {Style.RESET_ALL}")
                time.sleep(Config.DELAY_ALL_EXHAUSTED)
                active = team.get_best_available()
                if not active: tel.is_disqualified = True; return False

            mn = active["name"]
            if not printed:
                print_board(self.board, f"{tn} [{mn}]", color, self.board.fullmove_number)
                printed = True
            print(f"{Fore.MAGENTA}🤖 {tn} | {mn} | mode={mode.value} | switch {switches+1}/{MAX_SWITCHES}{Style.RESET_ALL}")

            failed = []
            for att in range(MAX_ATTEMPTS):
                temp = temps[att]

                if mode == GameMode.PREMIUM:
                    raw = self._premium_move(team, tel, color, temp, failed, opp_last)
                elif mode == GameMode.ULTRA_TERMINATOR:
                    raw = self._ultra_terminator_move(team, tel, color, temp, failed, opp_last)
                elif mode == GameMode.TERMINATOR:
                    raw = self._single_move(team, tel, color, temp, failed, opp_last, terminator=True)
                elif mode == GameMode.USER:
                    raw = self._single_move(team, tel, color, temp, failed, opp_last, user_mode=True)
                else:  # ECONOMY
                    raw = self._single_move(team, tel, color, temp, failed, opp_last)

                if not raw:
                    print(f"{Fore.RED}❌ No move (att {att+1}){Style.RESET_ALL}")
                    continue

                print(f"{Fore.GREEN}♟️  Extracted:{Style.RESET_ALL} {raw}")
                ok, err = self._try_push(raw)
                if ok:
                    dur = time.time() - t0
                    tel.total_moves += 1; tel.total_think_time += dur
                    self.annotations.append(f"{{{team.current_model_name}}}")
                    self._register_move_model(tn, team.current_model)
                    self._save_pgn()
                    print(f"{Fore.GREEN}✅ {raw} ({dur:.1f}s){Style.RESET_ALL}")
                    return True
                else:
                    tel.illegal_moves += 1
                    # Also track illegal moves per model
                    m_key = make_model_key(tn, team.current_model)
                    if m_key in MODEL_STATS:
                        MODEL_STATS[m_key]["illegal_moves"] = MODEL_STATS[m_key].get("illegal_moves", 0) + 1
                    
                    if raw not in failed: failed.append(raw)
                    print(f"{Fore.RED}❌ Illegal ({raw}): {err}{Style.RESET_ALL}")
                    Config.BASE_DIR.mkdir(parents=True, exist_ok=True)
                    self.pod_dir.mkdir(parents=True, exist_ok=True)
                    log_line = f"[{datetime.datetime.now()}] {tn}:{team.current_model_name} | FEN: {self.board.fen()} | {raw}\n"
                    with open(Config.BASE_DIR/"hallucinations.log","a",encoding="utf-8") as f:
                        f.write(log_line)
                    with open(self.pod_dir/"hallucinations.log","a",encoding="utf-8") as f:
                        f.write(log_line)

            switches += 1
            if switches < MAX_SWITCHES:
                print(f"\n{Back.YELLOW}{Fore.BLACK} 🔄 {tn}: {mn} failed. Next model... {Style.RESET_ALL}")
                team.mark_exhausted(Config.COOLDOWN_MODEL_FAIL); tel.model_switches += 1

        tel.is_disqualified = True
        print(f"\n{Back.RED}{Fore.WHITE} ⛔ DQ: {tn} {Style.RESET_ALL}")
        return False

    def _premium_move(self, team, tel, color, temp, failed, opp_last):
        """Two-stage: Think → Extract."""
        # Stage 1: Think
        msgs1 = build_prompt_premium_think(self.board, color)
        content1, mn1 = api_call_with_failover(team, msgs1, tel, temperature=temp, max_tokens=500)
        if not content1: return None
        tel.stage1_calls += 1

        print(f"\n{Fore.CYAN}🧠 THINK [{mn1}]:{Style.RESET_ALL} {content1[:150].replace(chr(10),' ')}...")

        # Stage 2: Extract (tiny output)
        msgs2 = build_prompt_premium_extract(content1, color, failed)
        content2, mn2 = api_call_with_failover(team, msgs2, tel, temperature=0.0, max_tokens=15)
        if not content2: return None
        tel.stage2_calls += 1

        print(f"{Fore.GREEN}📤 EXTRACT [{mn2}]:{Style.RESET_ALL} '{content2.strip()}'")

        # Parse stage 2 (should be clean)
        raw = extract_move(content2, opp_last)
        if raw: return raw

        # Fallback: parse stage 1 analysis
        raw = extract_move(content1, opp_last)
        if raw: return raw

        return broad_scan(content1, self.board)

    def _ultra_terminator_move(self, team, tel, color, temp, failed, opp_last):
        """Two-stage: Premium reasoning, but with legal moves shown up front."""
        msgs1 = build_prompt_ultra_terminator_think(self.board, color, failed)
        content1, mn1 = api_call_with_failover(team, msgs1, tel, temperature=temp, max_tokens=500)
        if not content1: return None
        tel.stage1_calls += 1

        print(f"\n{Fore.CYAN}🧠 ULTRA THINK [{mn1}]:{Style.RESET_ALL} {content1[:150].replace(chr(10),' ')}...")

        msgs2 = build_prompt_premium_extract(content1, color, failed)
        content2, mn2 = api_call_with_failover(team, msgs2, tel, temperature=0.0, max_tokens=15)
        if not content2: return None
        tel.stage2_calls += 1

        print(f"{Fore.GREEN}📤 ULTRA EXTRACT [{mn2}]:{Style.RESET_ALL} '{content2.strip()}'")

        raw = extract_move(content2, opp_last)
        if raw: return raw

        raw = extract_move(content1, opp_last)
        if raw: return raw

        return broad_scan(content1, self.board)

    def _single_move(self, team, tel, color, temp, failed, opp_last,
                     terminator=False, user_mode=False):
        """Single-stage call for Economy/User/Terminator."""
        if user_mode:
            msgs = build_prompt_user(self.board, color, failed)
        elif terminator:
            msgs = build_prompt_terminator(self.board, color, failed)
        else:
            msgs = build_prompt_economy(self.board, color, failed)

        content, mn = api_call_with_failover(team, msgs, tel, temperature=temp, max_tokens=500)
        if not content: return None

        print(f"\n{Fore.YELLOW}💡 [{mn}]:{Style.RESET_ALL} {content[:150].replace(chr(10),' ')}...")

        raw = extract_move(content, opp_last)
        if not raw: raw = broad_scan(content, self.board)
        return raw

    def _try_push(self, ms):
        cl = ms.replace("!","").replace("?","").strip()
        if "/" in cl or (" " in cl and len(cl)>10): return False, "FEN/text"
        try: self.board.push_san(cl); return True, ""
        except Exception as e1:
            try: self.board.push_uci(cl); return True, ""
            except: return False, str(e1)

    def _save_pgn(self):
        g = chess.pgn.Game()
        g.headers.update({"Event": f"Provider Tournament R{self.round_num}",
                          "Site": "LLM Chess v32.0", "Date": datetime.date.today().isoformat(),
                          "White": self.white_name, "Black": self.black_name})
        if self.forced_result:
            g.headers["Result"] = self.forced_result; g.headers["Termination"] = "DQ"
        else: g.headers["Result"] = self.board.result()
        node = g
        for i, mv in enumerate(self.board.move_stack):
            node = node.add_variation(mv)
            if i < len(self.annotations): node.comment = self.annotations[i]
        with open(self.pgn_path, "w", encoding="utf-8") as f: print(g, file=f, end="\n\n")

    def to_dict(self):
        return {"table_id": self.table_id, "round_num": self.round_num,
                "tournament_cycle": getattr(self, "tournament_cycle", 1),
                "white": self.white_name, "black": self.black_name,
                "fen": self.board.fen(), "state": self.state.value, "annotations": self.annotations,
                "white_model_history": self.white_model_history,
                "black_model_history": self.black_model_history}

    @staticmethod
    def restore(data):
        m = ChessMatch(data["table_id"], data["round_num"], data["white"], data["black"], data.get("tournament_cycle", 1))
        m.state = MatchState(data["state"]); m.annotations = data.get("annotations", [])
        m.white_model_history = data.get("white_model_history", [])
        m.black_model_history = data.get("black_model_history", [])
        if m.pgn_path.exists():
            with open(m.pgn_path) as f:
                try:
                    g = chess.pgn.read_game(f)
                    if g:
                        m.board = g.board()
                        for mv in g.mainline_moves(): m.board.push(mv)
                except: pass
        return m

# ==========================================
# 🖥️ UI
# ==========================================

def draw_banner():
    os.system('cls' if os.name=='nt' else 'clear')
    print(f"{Fore.CYAN}╔{'═'*78}╗")
    print(f"║ {Style.BRIGHT}{Fore.WHITE}    ♟️  LLM CHESS: PROVIDER TOURNAMENT v32.0  ♟️                              {Fore.CYAN}║")
    print(f"╠{'═'*78}╣")
    print(f"║ {Fore.YELLOW}Mode: {CURRENT_MODE.value.upper():<10} | Select: {CURRENT_SELECTION_SUBMODE.value.upper():<8} | OR: {CURRENT_OPENROUTER_MODE.value.upper():<6} {Fore.CYAN}║")
    print(f"║ {Fore.GREEN}Game:{Style.RESET_ALL} start test top teams logs delays                                       {Fore.CYAN}║")
    print(f"║ {Fore.GREEN}Config:{Style.RESET_ALL} mode submode ormode key addmodel rmmodel reset                       {Fore.CYAN}║")
    print(f"║ {Fore.GREEN}Info:{Style.RESET_ALL} modes teams help exit                                                  {Fore.CYAN}║")
    print(f"╚{'═'*78}╝{Style.RESET_ALL}")

def print_board(board, name, color, num):
    print(f"\n   {Back.BLACK}{Fore.WHITE} Move {num}: {name} ({color}) {Style.RESET_ALL}")
    print(f"  {Back.WHITE}{Fore.BLACK} a b c d e f g h {Style.RESET_ALL}")
    for i, row in enumerate(str(board).split("\n")):
        r = 8 - i
        colored = ""
        for ch in row:
            if ch==" ": colored+=" "
            elif ch==".": colored+=f"{Fore.BLACK}.{Style.RESET_ALL}"
            elif ch.isupper(): colored+=f"{Fore.YELLOW}{Style.BRIGHT}{ch}{Style.RESET_ALL}"
            else: colored+=f"{Fore.CYAN}{ch}{Style.RESET_ALL}"
        print(f"{Back.WHITE}{Fore.BLACK} {r} {Style.RESET_ALL} {colored} {Back.WHITE}{Fore.BLACK} {r} {Style.RESET_ALL}")
    print(f"  {Back.WHITE}{Fore.BLACK} a b c d e f g h {Style.RESET_ALL}\n")

# ==========================================
# 🏆 TOURNAMENT MANAGER
# ==========================================

class TournamentManager:
    def __init__(self):
        Config.BASE_DIR.mkdir(parents=True, exist_ok=True)
        self.tournament_cycle = 1
        self.current_round = 1
        self.teams: Dict[str, TeamTelemetry] = {}
        self.provider_teams: Dict[str, ProviderTeam] = {}
        self.active_matches: List[ChessMatch] = []
        self._load_state_or_init()

    def _load_state_or_init(self):
        global CURRENT_MODE, CURRENT_SELECTION_SUBMODE, CURRENT_OPENROUTER_MODE, MODEL_STATS
        if Config.STATE_FILE.exists():
            try:
                with open(Config.STATE_FILE) as f: data = json.load(f)
                self.tournament_cycle = data.get("tournament_cycle", 1)
                self.current_round = data["current_round"]
                CURRENT_MODE = GameMode(data.get("current_mode", CURRENT_MODE.value))
                CURRENT_SELECTION_SUBMODE = SelectionSubMode(data.get("selection_submode", CURRENT_SELECTION_SUBMODE.value))
                CURRENT_OPENROUTER_MODE = OpenRouterMode(data.get("openrouter_mode", CURRENT_OPENROUTER_MODE.value))
                
                Config.DELAY_BETWEEN_MOVES = data.get("delay_between_moves", Config.DELAY_BETWEEN_MOVES)
                Config.DELAY_API_RETRY = data.get("delay_api_retry", Config.DELAY_API_RETRY)
                Config.DELAY_ALL_EXHAUSTED = data.get("delay_all_exhausted", Config.DELAY_ALL_EXHAUSTED)
                Config.DELAY_NO_ACTIVE = data.get("delay_no_active", Config.DELAY_NO_ACTIVE)
                Config.COOLDOWN_RATE_LIMIT = data.get("cooldown_rate_limit", Config.COOLDOWN_RATE_LIMIT)
                Config.COOLDOWN_ERROR = data.get("cooldown_error", Config.COOLDOWN_ERROR)
                Config.COOLDOWN_MODEL_FAIL = data.get("cooldown_model_fail", Config.COOLDOWN_MODEL_FAIL)

                MODEL_STATS = data.get("model_stats", {})
                TEAM_ELO.clear()
                TEAM_ELO.update(data.get("team_elo", {}))
                self.teams = {k: TeamTelemetry.from_dict(v) for k,v in data.get("teams",{}).items() if k in TEAM_NAMES}
                for pt in data.get("provider_teams",[]):
                    if pt["name"] in TEAM_NAMES: self.provider_teams[pt["name"]] = ProviderTeam.from_dict(pt)
                self.active_matches = [ChessMatch.restore(m) for m in data.get("active_matches",[])
                                        if m["white"] in TEAM_NAMES and m["black"] in TEAM_NAMES]
                for n in TEAM_NAMES:
                    if n not in self.teams: self.teams[n] = TeamTelemetry(name=n)
                    if n not in self.provider_teams: self.provider_teams[n] = ProviderTeam(n)
                self._sync_model_stats()
            except: self._init_fresh()
        else: self._init_fresh()

    def _init_fresh(self):
        global MODEL_STATS
        MODEL_STATS = {}
        TEAM_ELO.clear()
        self.tournament_cycle = 1
        self._reset_tournament_scores()
        self.provider_teams = {n: ProviderTeam(n) for n in TEAM_NAMES}
        self.active_matches = []
        self._sync_model_stats()

    def _reset_tournament_scores(self):
        self.teams = {n: TeamTelemetry(name=n) for n in TEAM_NAMES}

    def _sync_model_stats(self):
        for team_name, provider_team in self.provider_teams.items():
            for model_entry in provider_team.chain:
                ensure_model_stats(team_name, model_entry)
        for team_name in TEAM_NAMES:
            ensure_team_elo(team_name)

    def _save_state(self):
        with open(Config.STATE_FILE, "w") as f:
            json.dump({"tournament_cycle": self.tournament_cycle,
                        "current_round": self.current_round,
                        "current_mode": CURRENT_MODE.value,
                        "selection_submode": CURRENT_SELECTION_SUBMODE.value,
                        "openrouter_mode": CURRENT_OPENROUTER_MODE.value,
                        "continuous_tournaments": Config.CONTINUOUS_TOURNAMENTS,
                        "delay_between_moves": Config.DELAY_BETWEEN_MOVES,
                        "delay_api_retry": Config.DELAY_API_RETRY,
                        "delay_all_exhausted": Config.DELAY_ALL_EXHAUSTED,
                        "delay_no_active": Config.DELAY_NO_ACTIVE,
                        "cooldown_rate_limit": Config.COOLDOWN_RATE_LIMIT,
                        "cooldown_error": Config.COOLDOWN_ERROR,
                        "cooldown_model_fail": Config.COOLDOWN_MODEL_FAIL,
                        "model_stats": MODEL_STATS,
                        "team_elo": TEAM_ELO,
                        "teams": {k:t.to_dict() for k,t in self.teams.items()},
                        "provider_teams": [pt.to_dict() for pt in self.provider_teams.values()],
                        "active_matches": [m.to_dict() for m in self.active_matches]}, f, indent=4)
        with open(Config.STATS_FILE, "w") as f:
            stats_data = {"tournament_cycle": self.tournament_cycle,
                       "continuous_tournaments": Config.CONTINUOUS_TOURNAMENTS,
                       "team_elo": sorted(TEAM_ELO.values(), key=lambda x: x["elo"], reverse=True),
                       "teams": [t.to_dict() for t in self.teams.values()],
                       "models": sorted(MODEL_STATS.values(), key=lambda x: x["elo"], reverse=True)}
            json.dump(stats_data, f, indent=4)
        pod_dir = Config.BASE_DIR / f"pod_tournament_data_{self.tournament_cycle}"
        pod_dir.mkdir(parents=True, exist_ok=True)
        # Eagerly create/touch log files so they are visible immediately
        open(Config.BASE_DIR / "hallucinations.log", "a", encoding="utf-8").close()
        open(pod_dir / "hallucinations.log", "a", encoding="utf-8").close()
        
        with open(pod_dir / "detailed_analytics.json", "w") as f:
            json.dump(stats_data, f, indent=4)

    def _rr_pairings(self):
        return [[("Team Google","Team GitHub"),("Team Cerebras","Team OpenRouter")],
                [("Team Google","Team Cerebras"),("Team GitHub","Team OpenRouter")],
                [("Team Google","Team OpenRouter"),("Team GitHub","Team Cerebras")]]

    def _start_round(self, rn):
        if rn == 1:
            self._reset_tournament_scores()
            self.active_matches = []
        self.current_round = rn; self.active_matches = []
        pp = self._rr_pairings()
        if rn > len(pp):
            self._print_standings()
            self._final_report()
            if Config.CONTINUOUS_TOURNAMENTS:
                self.tournament_cycle += 1
                print(f"\n{Back.CYAN}{Fore.BLACK} STARTING TOURNAMENT #{self.tournament_cycle} {Style.RESET_ALL}")
                self._start_round(1)
                return
            sys.exit()
        print(f"\n{Back.GREEN}{Fore.BLACK} ═══ ROUND {rn} ═══ {Style.RESET_ALL}")
        for i,(w,b) in enumerate(pp[rn-1], 1):
            print(f"  T{i}: {w} vs {b}")
            self.active_matches.append(ChessMatch(i, rn, w, b, self.tournament_cycle))
            self.teams[w].played_opponents.append(b); self.teams[b].played_opponents.append(w)
            self.teams[w].color_history.append("W"); self.teams[b].color_history.append("B")
        self._save_state()

    def _process_results(self):
        for m in self.active_matches:
            res = m.forced_result or m.board.result()
            w, b = m.white_name, m.black_name
            # model histories used for weighted ELO (all models, proportional to moves)
            if res == "1-0":
                self.teams[w].points += 1
                self.teams[w].wins += 1
                self.teams[b].losses += 1
                apply_team_elo_result(w, b, 1.0)
                # Weighted ELO: all white models share credit by moves played
                white_opp_elo = ensure_team_elo(b)["elo"]
                black_opp_elo = ensure_team_elo(w)["elo"]
                apply_weighted_model_elo(m.white_model_history, 1.0, white_opp_elo)
                apply_weighted_model_elo(m.black_model_history, 0.0, black_opp_elo)
            elif res == "0-1":
                self.teams[b].points += 1
                self.teams[b].wins += 1
                self.teams[w].losses += 1
                apply_team_elo_result(w, b, 0.0)
                white_opp_elo = ensure_team_elo(b)["elo"]
                black_opp_elo = ensure_team_elo(w)["elo"]
                apply_weighted_model_elo(m.white_model_history, 0.0, white_opp_elo)
                apply_weighted_model_elo(m.black_model_history, 1.0, black_opp_elo)
            else:
                self.teams[w].points += 0.5
                self.teams[b].points += 0.5
                self.teams[w].draws += 1
                self.teams[b].draws += 1
                apply_team_elo_result(w, b, 0.5)
                white_opp_elo = ensure_team_elo(b)["elo"]
                black_opp_elo = ensure_team_elo(w)["elo"]
                apply_weighted_model_elo(m.white_model_history, 0.5, white_opp_elo)
                apply_weighted_model_elo(m.black_model_history, 0.5, black_opp_elo)
        self._print_standings()
        self._save_state()
        if self.current_round >= Config.TOTAL_ROUNDS:
            print(f"\n{Back.YELLOW}{Fore.BLACK} TOURNAMENT COMPLETE {Style.RESET_ALL}")
            self._final_report()
            if Config.CONTINUOUS_TOURNAMENTS:
                self.tournament_cycle += 1
                self._start_round(1)
            else:
                sys.exit()
            return
        if self.current_round < Config.TOTAL_ROUNDS:
            self._start_round(self.current_round + 1)
        else: print(f"\n{Back.YELLOW}{Fore.BLACK} 🏆 TOURNAMENT COMPLETE! 🏆{Style.RESET_ALL}"); self._final_report(); sys.exit()

    def _print_standings(self):
        st = sorted(self.teams.values(), key=lambda t:t.points, reverse=True)
        print(f"\n{Fore.YELLOW}{'═'*88}\n LEADERBOARD (R{self.current_round})\n{'═'*88}{Style.RESET_ALL}")
        print(f"{'Team':<20}|{'ELO':>7}|{'Pts':>5}|{'W-D-L':>7}|{'Succ':>5}|{'Ill':>4}|{'Rate%':>5}|{'Sw':>3}|{'S1':>3}|{'S2':>3}|{'AvgT':>5}")
        print('-'*88)
        for t in st:
            r = f"{t.wins}-{t.draws}-{t.losses}" if not t.is_disqualified else "DQ"
            team_elo_val = ensure_team_elo(t.name)["elo"]
            ill_pct = t.illegal_rate * 100
            print(f"{t.name:<20}|{team_elo_val:>7.1f}|{t.points:>5.1f}|{r:>7}|{t.total_moves:>5}|{t.illegal_moves:>4}|{ill_pct:>5.1f}%|"
                  f"{t.model_switches:>3}|{t.stage1_calls:>3}|{t.stage2_calls:>3}|{t.avg_time:>4.1f}s")

    def _final_report(self):
        print(f"\n{Back.MAGENTA}{Fore.WHITE} 📊 FINAL REPORT {Style.RESET_ALL}")
        for n in sorted(self.teams, key=lambda x:self.teams[x].points, reverse=True):
            t = self.teams[n]; pt = self.provider_teams[n]
            print(f"\n{Fore.CYAN}{'─'*50}\n {n} — {t.points}pts ({t.wins}W-{t.draws}D-{t.losses}L)\n{'─'*50}{Style.RESET_ALL}")
            print(f"  Moves: {t.total_moves} | Illegal: {t.illegal_moves} ({t.illegal_rate*100:.0f}%)")
            print(f"  API calls: S1={t.stage1_calls} S2={t.stage2_calls} | Switches: {t.model_switches}")
            print(f"  Models: {', '.join(t.models_used) or 'None'}")
            for s in pt.switch_log[-3:]: print(f"    {s['from']}→{s['to']} ({s['reason']})")

    def test_connection(self):
        print(f"\n{Back.BLUE}{Fore.WHITE} CONNECTIVITY TEST {Style.RESET_ALL}\n")
        for tn, cfg in PROVIDER_CHAINS.items():
            print(f"{Fore.CYAN}{tn} [{cfg['provider'].upper()}]:{Style.RESET_ALL}")
            for e in cfg["chain"]:
                print(f"  {e['name']:<30}...", end="", flush=True)
                try:
                    with suppress_stderr():
                        call_model_direct(cfg["provider"],e["model"],[{"role":"user","content":"OK"}],max_tokens=5)
                    print(f"{Fore.GREEN} ✅{Style.RESET_ALL}")
                except Exception as ex:
                    err = str(ex)[:40].lower()
                    print(f"{Fore.YELLOW} 🟡 BUSY{Style.RESET_ALL}" if any(x in err for x in ["429","rate","busy"]) 
                          else f"{Fore.RED} ❌ {str(ex)[:35]}{Style.RESET_ALL}")
            print()

    def _cmd_delays(self):
        print(f"\n{Fore.YELLOW}Current delays config:{Style.RESET_ALL}")
        print(f"  1. Delay between moves: {Config.DELAY_BETWEEN_MOVES}s")
        print(f"  2. Delay on API retry: {Config.DELAY_API_RETRY}s")
        print(f"  3. Wait when all models exhausted: {Config.DELAY_ALL_EXHAUSTED}s")
        print(f"  4. Wait when no active models on failover: {Config.DELAY_NO_ACTIVE}s")
        print(f"  5. Rate limit cooldown: {Config.COOLDOWN_RATE_LIMIT}s")
        print(f"  6. API error cooldown: {Config.COOLDOWN_ERROR}s")
        print(f"  7. Model fail/illegal cooldown: {Config.COOLDOWN_MODEL_FAIL}s")
        
        try:
            choice = input(f"\nSelect parameter to edit (1-7) or blank to cancel: ").strip()
            if not choice: return
            idx = int(choice)
            if idx not in range(1, 8):
                print(f"{Fore.RED}Invalid option.{Style.RESET_ALL}")
                return
            
            val = input("New value in seconds: ").strip()
            if not val: return
            val_f = float(val)
            val_i = int(val_f)
            
            if idx == 1: Config.DELAY_BETWEEN_MOVES = val_f
            elif idx == 2: Config.DELAY_API_RETRY = val_f
            elif idx == 3: Config.DELAY_ALL_EXHAUSTED = val_f
            elif idx == 4: Config.DELAY_NO_ACTIVE = val_f
            elif idx == 5: Config.COOLDOWN_RATE_LIMIT = val_i
            elif idx == 6: Config.COOLDOWN_ERROR = val_i
            elif idx == 7: Config.COOLDOWN_MODEL_FAIL = val_i
            
            self._save_state()
            print(f"{Fore.GREEN}Saved successfully.{Style.RESET_ALL}")
        except ValueError:
            print(f"{Fore.RED}Invalid number format.{Style.RESET_ALL}")

    def _cmd_mode(self):
        global CURRENT_MODE
        print(f"\n{Fore.YELLOW}Current mode: {CURRENT_MODE.value.upper()}{Style.RESET_ALL}\n")
        for m in GameMode:
            marker = "►" if m == CURRENT_MODE else " "
            print(f"  {marker} {m.value:<12} — {MODE_DESCRIPTIONS[m]}")
        choice = input(f"\nNew mode (user/economy/premium/terminator/ultra-terminator): ").strip().lower()
        for m in GameMode:
            if m.value == choice:
                CURRENT_MODE = m
                self._save_state()
                print(f"{Fore.GREEN}Mode: {m.value.upper()}{Style.RESET_ALL}")
                return
        print(f"{Fore.RED}Invalid mode.{Style.RESET_ALL}")

    def _cmd_submode(self):
        global CURRENT_SELECTION_SUBMODE
        print(f"\n{Fore.YELLOW}Current selection submode: {CURRENT_SELECTION_SUBMODE.value.upper()}{Style.RESET_ALL}\n")
        for sm in SelectionSubMode:
            marker = "►" if sm == CURRENT_SELECTION_SUBMODE else " "
            print(f"  {marker} {sm.value:<12} — {SELECTION_SUBMODE_DESCRIPTIONS[sm]}")
        choice = input("\nNew submode (balanced/fullmax/classic): ").strip().lower()
        for sm in SelectionSubMode:
            if sm.value == choice:
                CURRENT_SELECTION_SUBMODE = sm
                self._save_state()
                print(f"{Fore.GREEN}Selection submode: {sm.value.upper()}{Style.RESET_ALL}")
                return
        print(f"{Fore.RED}Invalid submode.{Style.RESET_ALL}")

    def _cmd_ormode(self):
        global CURRENT_OPENROUTER_MODE
        print(f"\n{Fore.YELLOW}Current OpenRouter mode: {CURRENT_OPENROUTER_MODE.value.upper()}{Style.RESET_ALL}\n")
        print("  ► auto   — Use OpenRouter's own free auto-router only." if CURRENT_OPENROUTER_MODE == OpenRouterMode.AUTO else "    auto   — Use OpenRouter's own free auto-router only.")
        print("  ► custom — Use our ranked OpenRouter roster and failover." if CURRENT_OPENROUTER_MODE == OpenRouterMode.CUSTOM else "    custom — Use our ranked OpenRouter roster and failover.")
        choice = input("\nOpenRouter mode (auto/custom): ").strip().lower()
        for orm in OpenRouterMode:
            if orm.value == choice:
                CURRENT_OPENROUTER_MODE = orm
                self._save_state()
                print(f"{Fore.GREEN}OpenRouter mode: {orm.value.upper()}{Style.RESET_ALL}")
                return
        print(f"{Fore.RED}Invalid OpenRouter mode.{Style.RESET_ALL}")

    def _cmd_key(self):
        """In-app API key manager - view, edit, show full key, reset to default."""
        from keys import _DEFAULT_KEYS, _KEYS_LOCAL_FILE
        while True:
            print(f"\n{Fore.YELLOW}==================== API KEY MANAGER ===================={Style.RESET_ALL}")
            local_exists = _KEYS_LOCAL_FILE.exists()
            if local_exists:
                src_label = f"{Fore.GREEN}keys_local.json (custom){Style.RESET_ALL}"
            else:
                src_label = f"{Fore.CYAN}default (keys.py){Style.RESET_ALL}"
            print(f"  Source: {src_label}\n")
            for k, v in KEYS.items():
                masked = v[:6] + "..." + v[-4:] if len(v) > 10 else "***"
                is_default = v == _DEFAULT_KEYS.get(k, "")
                tag = "" if is_default else f" {Fore.GREEN}[custom]{Style.RESET_ALL}"
                print(f"  {Fore.CYAN}{k:<12}{Style.RESET_ALL}  {masked}{tag}")
            print(f"\n{Fore.WHITE}Commands:{Style.RESET_ALL}")
            print("  GOOGLE/COHERE/OPENROUTER/SAMBANOVA  - edit that key")
            print("  show GOOGLE                         - show full key")
            print("  reset                               - reset ALL to default")
            print("  back                                - return to menu")
            cmd = input(f"\n{Fore.CYAN}keys> {Style.RESET_ALL}").strip()
            if not cmd or cmd.lower() == "back":
                break
            elif cmd.lower() == "reset":
                confirm = input("Reset ALL keys to default? (yes/no): ").strip().lower()
                if confirm == "yes":
                    KEYS.update(_DEFAULT_KEYS)
                    os.environ["GEMINI_API_KEY"] = KEYS["GOOGLE"]
                    os.environ["COHERE_API_KEY"] = KEYS["COHERE"]
                    if _KEYS_LOCAL_FILE.exists():
                        _KEYS_LOCAL_FILE.unlink()
                    print(f"{Fore.GREEN}Keys reset to default (keys_local.json removed).{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}Cancelled.{Style.RESET_ALL}")
            elif cmd.lower().startswith("show "):
                name = cmd[5:].strip().upper()
                if name in KEYS:
                    print(f"  {Fore.YELLOW}{name}:{Style.RESET_ALL} {KEYS[name]}")
                else:
                    print(f"{Fore.RED}Unknown key: {name}{Style.RESET_ALL}")
            elif cmd.upper() in KEYS:
                t = cmd.upper()
                print(f"  Current {t}: {KEYS[t][:6]}...{KEYS[t][-4:]}")
                v = input(f"  New value for {t} (blank = cancel): ").strip()
                if v:
                    KEYS[t] = v
                    if t == "GOOGLE":
                        os.environ["GEMINI_API_KEY"] = v
                    if t == "COHERE":
                        os.environ["COHERE_API_KEY"] = v
                    save_keys(KEYS)
                    print(f"{Fore.GREEN}Key {t} updated and saved to keys_local.json.{Style.RESET_ALL}")
                else:
                    print(f"{Fore.YELLOW}Cancelled.{Style.RESET_ALL}")
            else:
                print(f"{Fore.RED}Unknown. Use key name, show NAME, reset, or back.{Style.RESET_ALL}")

    def _cmd_addmodel(self):
        print(f"\n{Fore.YELLOW}Add model to team chain{Style.RESET_ALL}")
        print("Teams:", ", ".join(TEAM_NAMES))
        tn = input("Team name: ").strip()
        if tn not in PROVIDER_CHAINS: print(f"{Fore.RED}Not found.{Style.RESET_ALL}"); return
        name = input("Display name (e.g. 'Gemini Pro'): ").strip()
        model = input("Model ID (e.g. 'gemini/gemini-2.5-pro'): ").strip()
        if name and model:
            model_entry = {"name": name, "model": model}
            PROVIDER_CHAINS[tn]["chain"].append(model_entry)
            if tn in self.provider_teams: self.provider_teams[tn].chain.append(model_entry)
            ensure_model_stats(tn, model_entry)
            self._save_state()
            print(f"{Fore.GREEN}Added {name} to {tn}.{Style.RESET_ALL}")

    def _cmd_rmmodel(self):
        print(f"\n{Fore.YELLOW}Remove model from team chain{Style.RESET_ALL}")
        tn = input("Team name: ").strip()
        if tn not in PROVIDER_CHAINS: print(f"{Fore.RED}Not found.{Style.RESET_ALL}"); return
        chain = PROVIDER_CHAINS[tn]["chain"]
        for i, e in enumerate(chain): print(f"  {i}. {e['name']} ({e['model']})")
        idx = input("Index to remove: ").strip()
        try:
            idx = int(idx)
            removed = chain.pop(idx)
            if tn in self.provider_teams and idx < len(self.provider_teams[tn].chain):
                self.provider_teams[tn].chain.pop(idx)
            print(f"{Fore.GREEN}Removed {removed['name']}.{Style.RESET_ALL}")
        except: print(f"{Fore.RED}Invalid index.{Style.RESET_ALL}")

    def _cmd_teams(self):
        print(f"\n{Fore.YELLOW}═══ TEAMS ═══{Style.RESET_ALL}")
        for tn, cfg in PROVIDER_CHAINS.items():
            pt = self.provider_teams.get(tn)
            ci = pt.current_index if pt else 0
            print(f"\n{Fore.CYAN}{tn} [{cfg['provider'].upper()}]:{Style.RESET_ALL}")
            for i, e in enumerate(cfg["chain"]):
                mk = f"{Fore.GREEN}►{Style.RESET_ALL}" if i==ci else " "
                tags = []
                if e.get("router"): tags.append("router")
                if e.get("experimental"): tags.append("exp")
                if e.get("full_max"): tags.append("full")
                elif e.get("balanced"): tags.append("bal")
                ex = ""
                if pt and i in pt.exhausted_until:
                    r = int(pt.exhausted_until[i]-time.time())
                    ex = f" {Fore.RED}({r}s){Style.RESET_ALL}" if r>0 else f" {Fore.GREEN}(ok){Style.RESET_ALL}"
                label = f" [{'|'.join(tags)}]" if tags else ""
                elo = ensure_model_stats(tn, e)["elo"]
                print(f"  {mk} {i}. {e['name']:<28} {e['model']} | ELO {elo:>7.1f}{label}{ex}")

    def _print_model_elo(self):
        """Show ELO for all models (per-model)."""
        print(f"\n{Fore.YELLOW}MODEL ELO RANKING{Style.RESET_ALL}")
        print("-" * 115)
        print(f"  # | {'Team':<16} | {'Model':<28} | {'ELO':>7} | Games | W-D-L | Ill")
        print("-" * 115)
        for i, record in enumerate(sorted(MODEL_STATS.values(), key=lambda x: x["elo"], reverse=True), 1):
            wdl = f"{record['wins']}-{record['draws']}-{record['losses']}"
            ill = record.get("illegal_moves", 0)
            print(f"{i:>3}. {record['team']:<16} | {record['name']:<28} | {record['elo']:>7.1f} | {record['games']:>5} | {wdl:<7} | {ill:>3}")

    def _print_team_elo(self):
        """Show ELO for all teams."""
        print(f"\n{Fore.YELLOW}TEAM ELO RANKING{Style.RESET_ALL}")
        print("-" * 60)
        print(f"  # | {'Team':<20} | {'ELO':>7} | Games | W-D-L")
        print("-" * 60)
        for i, record in enumerate(sorted(TEAM_ELO.values(), key=lambda x: x["elo"], reverse=True), 1):
            wdl = f"{record['wins']}-{record['draws']}-{record['losses']}"
            print(f"{i:>3}. {record['name']:<20} | {record['elo']:>7.1f} | {record['games']:>5} | {wdl}")

    def _cmd_modes(self):
        print(f"\n{Fore.YELLOW}═══ GAME MODES ═══{Style.RESET_ALL}\n")
        for m in GameMode:
            marker = f"{Fore.GREEN}► ACTIVE{Style.RESET_ALL}" if m==CURRENT_MODE else ""
            print(f"  {m.value.upper():<12} {marker}")
            print(f"    {MODE_DESCRIPTIONS[m]}\n")
        print(f"{Fore.YELLOW}═══ SELECTION SUBMODES ═══{Style.RESET_ALL}\n")
        for sm in SelectionSubMode:
            marker = f"{Fore.GREEN}► ACTIVE{Style.RESET_ALL}" if sm==CURRENT_SELECTION_SUBMODE else ""
            print(f"  {sm.value.upper():<12} {marker}")
            print(f"    {SELECTION_SUBMODE_DESCRIPTIONS[sm]}\n")
        print(f"{Fore.YELLOW}═══ OPENROUTER MODES ═══{Style.RESET_ALL}\n")
        print(f"  AUTO{' ' * 8}{Fore.GREEN}► ACTIVE{Style.RESET_ALL}" if CURRENT_OPENROUTER_MODE == OpenRouterMode.AUTO else "  AUTO")
        print("    Use OpenRouter's own free router only.\n")
        print(f"  CUSTOM{' ' * 6}{Fore.GREEN}► ACTIVE{Style.RESET_ALL}" if CURRENT_OPENROUTER_MODE == OpenRouterMode.CUSTOM else "  CUSTOM")
        print("    Use the ranked OpenRouter team roster with our failover order.\n")

    def _cmd_help(self):
        print(f"""
{Fore.YELLOW}═══ COMMANDS ═══{Style.RESET_ALL}
  {Fore.GREEN}start{Style.RESET_ALL}      Start/resume tournament
  {Fore.GREEN}test{Style.RESET_ALL}       Test all model connections
  {Fore.GREEN}top{Style.RESET_ALL}        Show leaderboard
  {Fore.GREEN}elo{Style.RESET_ALL}        Show model ELO leaderboard
  {Fore.GREEN}teamelo{Style.RESET_ALL}    Show team ELO leaderboard
  {Fore.GREEN}teams{Style.RESET_ALL}      Show team rosters & status
  {Fore.GREEN}logs{Style.RESET_ALL}       Show recent illegal moves
  {Fore.GREEN}delays{Style.RESET_ALL}     Change retry and cooldown delays parameters
  {Fore.GREEN}mode{Style.RESET_ALL}       Change game mode (user/economy/premium/terminator/ultra-terminator)
  {Fore.GREEN}submode{Style.RESET_ALL}    Change model selection submode (balanced/fullmax/classic)
  {Fore.GREEN}ormode{Style.RESET_ALL}     Change OpenRouter mode (auto/custom)
  {Fore.GREEN}modes{Style.RESET_ALL}      Explain all modes
  {Fore.GREEN}key{Style.RESET_ALL}        Change API key
  {Fore.GREEN}addmodel{Style.RESET_ALL}   Add model to a team
  {Fore.GREEN}rmmodel{Style.RESET_ALL}    Remove model from a team
  {Fore.GREEN}reset{Style.RESET_ALL}      Wipe all data and restart
  {Fore.GREEN}exit{Style.RESET_ALL}       Quit
""")

    def run_main_loop(self):
        global CURRENT_MODE, CURRENT_SELECTION_SUBMODE, CURRENT_OPENROUTER_MODE
        show_menu = True
        while True:
            if show_menu:
                draw_banner()
                st = "NEW" if not self.active_matches else f"ROUND {self.current_round}"
                print(f"{Fore.GREEN}Status: T{self.tournament_cycle} {st} | Mode: {CURRENT_MODE.value.upper()} | Select: {CURRENT_SELECTION_SUBMODE.value.upper()} | OR: {CURRENT_OPENROUTER_MODE.value.upper()}{Style.RESET_ALL}")
                while True:
                    cmd = input(f"\n{Fore.CYAN}> {Style.RESET_ALL}").strip().lower()
                    if not cmd: continue
                    if cmd=="start":
                        if not self.active_matches: self._start_round(1)
                        show_menu = False; break
                    elif cmd=="test": self.test_connection()
                    elif cmd=="top": self._print_standings()
                    elif cmd=="elo": self._print_model_elo()
                    elif cmd=="teamelo": self._print_team_elo()
                    elif cmd=="teams": self._cmd_teams()
                    elif cmd=="logs":
                        p = Config.BASE_DIR/"hallucinations.log"
                        if p.exists():
                            for l in open(p,encoding="utf-8").readlines()[-15:]:
                                print(f"{Fore.RED}• {l.strip()}{Style.RESET_ALL}")
                        else: print("No errors yet.")
                    elif cmd=="mode": self._cmd_mode()
                    elif cmd=="delays": self._cmd_delays()
                    elif cmd=="submode": self._cmd_submode()
                    elif cmd=="ormode": self._cmd_ormode()
                    elif cmd=="modes": self._cmd_modes()
                    elif cmd=="key": self._cmd_key()
                    elif cmd=="addmodel": self._cmd_addmodel()
                    elif cmd=="rmmodel": self._cmd_rmmodel()
                    elif cmd=="help": self._cmd_help()
                    elif cmd=="reset":
                        if input(f"{Back.RED}Type RESET: {Style.RESET_ALL}")=="RESET":
                            if Config.BASE_DIR.exists(): shutil.rmtree(Config.BASE_DIR)
                            os.execv(sys.executable, ['python']+sys.argv)
                    elif cmd=="exit": sys.exit()
                    else: print("Type 'help' for commands.")

            if not show_menu:
                try:
                    while True:
                        active = 0
                        for match in self.active_matches:
                            if match.state == MatchState.ACTIVE:
                                active += 1
                                w,b = match.white_name, match.black_name
                                ct = w if match.board.turn==chess.WHITE else b
                                pt = self.provider_teams[ct]
                                print(f"\n{Back.BLUE}{Fore.WHITE} T{match.table_id} {Back.BLACK} {w} vs {b} [{pt.current_model_name}] {Style.RESET_ALL}")
                                if not match.make_ai_move(self.teams, self.provider_teams, CURRENT_MODE):
                                    match.state = MatchState.COMPLETED
                                    match.forced_result = "0-1" if match.board.turn==chess.WHITE else "1-0"
                                    match._save_pgn()
                                    print(f"{Back.RED}{Fore.WHITE} TERMINATED {Style.RESET_ALL}")
                                self._save_state()
                                if match.board.is_game_over():
                                    match.state = MatchState.COMPLETED
                                    print(f"{Back.GREEN}{Fore.BLACK} GAME OVER: {match.board.result()} {Style.RESET_ALL}")
                                    match._save_pgn()
                                time.sleep(Config.DELAY_BETWEEN_MOVES)
                        if active == 0: self._process_results()
                except KeyboardInterrupt:
                    print("\nSaving..."); self._save_state(); sys.exit()

if __name__ == "__main__":
    from keys import check_keys_and_prompt
    check_keys_and_prompt(KEYS)
    os.environ["GEMINI_API_KEY"] = KEYS["GOOGLE"]
    os.environ["COHERE_API_KEY"] = KEYS["COHERE"]
    TournamentManager().run_main_loop()
