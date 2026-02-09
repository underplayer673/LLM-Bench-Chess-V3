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
# 🔑 KEYS (user can change in-app)
# ==========================================

KEYS = {
    "GOOGLE":     "Paste your API",
    "COHERE":     "Paste your API",
    "OPENROUTER": "Paste your API",
    "SAMBANOVA":  "Paste your API",
}

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

CURRENT_MODE = GameMode.PREMIUM

MODE_DESCRIPTIONS = {
    GameMode.USER: "Basic: just FEN, ask for move. Fast but dumb. NOT recommended.",
    GameMode.ECONOMY: "Smart single call with piece lists. Good balance.",
    GameMode.PREMIUM: "Two-stage: AI thinks first, then outputs move. BEST quality. DEFAULT.",
    GameMode.TERMINATOR: "Shows legal moves to AI. Strongest but unfair. NOT recommended.",
}

# ==========================================
# 🏢 PROVIDER CHAINS (editable in-app)
# ==========================================

PROVIDER_CHAINS = {
    "Team Google": {
        "provider": "google",
        "chain": [
            {"name": "Gemini 2.5 Flash",        "model": "gemini/gemini-2.5-flash"},
            {"name": "Gemini 3.0 Flash Preview", "model": "gemini/gemini-3-flash-preview"},
            {"name": "Gemini 2.5 Flash Lite",    "model": "gemini/gemini-2.5-flash-lite"},
            {"name": "Gemma 3 27B",              "model": "gemini/gemma-3-27b-it"},
            {"name": "Gemma 3 4B",               "model": "gemini/gemma-3-4b-it"},
        ],
    },
    "Team SambaNova": {
        "provider": "samba",
        "chain": [
            {"name": "DeepSeek V3",              "model": "DeepSeek-V3-0324"},
            {"name": "DeepSeek R1",              "model": "DeepSeek-R1"},
            {"name": "DeepSeek R1 Distill 70B",  "model": "DeepSeek-R1-Distill-Llama-70B"},
            {"name": "Qwen3 32B",                "model": "Qwen3-32B"},
            {"name": "Llama 3.3 70B",            "model": "Meta-Llama-3.3-70B-Instruct"},
            {"name": "Llama 3.1 8B",             "model": "Meta-Llama-3.1-8B-Instruct"},
        ],
    },
    "Team Cohere": {
        "provider": "cohere",
        "chain": [
            {"name": "Command A",               "model": "cohere/command-a-03-2025"},
            {"name": "Command R+",              "model": "cohere/command-r-plus-08-2024"},
            {"name": "Command R7B",             "model": "cohere/command-r7b-12-2024"},
            {"name": "Command Nightly",         "model": "cohere/command-nightly"},
            {"name": "Aya Expanse 32B",         "model": "cohere/c4ai-aya-expanse-32b"},
        ],
    },
    "Team OpenRouter": {
        "provider": "or",
        "chain": [
            {"name": "OR Llama 3.3 70B",        "model": "openrouter/meta-llama/llama-3.3-70b-instruct:free"},
            {"name": "OR Gemma 3 27B",          "model": "openrouter/google/gemma-3-27b-it:free"},
            {"name": "Trinity Large",           "model": "openrouter/arcee-ai/trinity-large-preview:free"},
            {"name": "Trinity Mini",            "model": "openrouter/arcee-ai/trinity-mini:free"},
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

    def get_best_available(self):
        """Get strongest available model (lowest index that's not exhausted)."""
        now = time.time()
        for i in range(len(self.chain)):
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

    def mark_exhausted(self, cooldown=300):
        old = self.current_model_name
        self.exhausted_until[self.current_index] = time.time() + cooldown
        now = time.time()
        for i in range(len(self.chain)):
            if i not in self.exhausted_until or self.exhausted_until[i] < now:
                self.current_index = i
                self.total_switches += 1
                self.switch_log.append({"from": old, "to": self.current_model_name,
                                        "reason": "exhausted",
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
        args["api_key"] = KEYS["SAMBANOVA"]
    elif provider == "or":
        args["model"] = model_id
        args["api_base"] = "https://openrouter.ai/api/v1"
        args["api_key"] = KEYS["OPENROUTER"]
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
    retries = 0
    # Give SambaNova (DeepSeek) more tokens for their <think> reasoning
    actual_max_tokens = max_tokens + 800 if team.provider == "samba" else max_tokens
    
    while retries < 10:
        active = team.get_best_available()
        if not active:
            time.sleep(30)
            active = team.get_best_available()
            if not active:
                return None, None
        try:
            with suppress_stderr():
                resp = call_model_direct(team.provider, active["model"], messages,
                                         temperature=temperature, max_tokens=actual_max_tokens)
            content = resp.choices[0].message.content
            # Strip <think> blocks from DeepSeek responses
            if team.provider == "samba":
                content = strip_think_tags(content)
            if active["name"] not in telemetry.models_used:
                telemetry.models_used.append(active["name"])
            return content, active["name"]
        except Exception as e:
            err = str(e).lower()
            telemetry.api_errors += 1
            retries += 1
            is_limit = any(x in err for x in ["429","quota","rate limit","busy","resource_exhausted","too many","overloaded"])
            if is_limit:
                team.mark_exhausted(300)
                telemetry.model_switches += 1
            elif retries >= 3:
                team.mark_exhausted(180)
                telemetry.model_switches += 1
                retries = 0
            else:
                time.sleep(5)
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

def extract_move(text, opp_last=None):
    def echo(m): return opp_last and m == opp_last

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
    PGN_DIR = BASE_DIR / "pgn"
    TOTAL_ROUNDS = 3

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
    def illegal_rate(self): return self.illegal_moves / max(self.total_moves + self.illegal_moves, 1)
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
    def __init__(self, table_id, round_num, white, black):
        self.table_id = table_id; self.round_num = round_num
        self.white_name = white; self.black_name = black
        self.board = chess.Board()
        self.state = MatchState.ACTIVE; self.forced_result = None
        self.annotations: List[str] = []
        self.pgn_dir = Config.PGN_DIR / f"Round_{round_num}"
        self.pgn_dir.mkdir(parents=True, exist_ok=True)
        self.pgn_path = self.pgn_dir / f"T{table_id}_{white}_vs_{black}.pgn"

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
                print(f"{Back.RED} 💀 {tn}: All exhausted. Wait 60s {Style.RESET_ALL}")
                time.sleep(60)
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
                    self._save_pgn()
                    print(f"{Fore.GREEN}✅ {raw} ({dur:.1f}s){Style.RESET_ALL}")
                    return True
                else:
                    tel.illegal_moves += 1
                    if raw not in failed: failed.append(raw)
                    print(f"{Fore.RED}❌ Illegal ({raw}): {err}{Style.RESET_ALL}")
                    Config.BASE_DIR.mkdir(parents=True, exist_ok=True)
                    with open(Config.BASE_DIR/"hallucinations.log","a",encoding="utf-8") as f:
                        f.write(f"[{datetime.datetime.now()}] {tn}:{team.current_model_name} | FEN: {self.board.fen()} | {raw}\n")

            switches += 1
            if switches < MAX_SWITCHES:
                print(f"\n{Back.YELLOW}{Fore.BLACK} 🔄 {tn}: {mn} failed. Next model... {Style.RESET_ALL}")
                team.mark_exhausted(600); tel.model_switches += 1

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
                "white": self.white_name, "black": self.black_name,
                "fen": self.board.fen(), "state": self.state.value, "annotations": self.annotations}

    @staticmethod
    def restore(data):
        m = ChessMatch(data["table_id"], data["round_num"], data["white"], data["black"])
        m.state = MatchState(data["state"]); m.annotations = data.get("annotations", [])
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
    print(f"║ {Fore.YELLOW}Mode: {CURRENT_MODE.value.upper():<10} | 4 Teams | Round Robin | 2-Stage AI                  {Fore.CYAN}║")
    print(f"║ {Fore.GREEN}Game:{Style.RESET_ALL} start test top teams logs                                              {Fore.CYAN}║")
    print(f"║ {Fore.GREEN}Config:{Style.RESET_ALL} mode key addmodel rmmodel reset                                      {Fore.CYAN}║")
    print(f"║ {Fore.GREEN}Info:{Style.RESET_ALL} modes help exit                                                        {Fore.CYAN}║")
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
        Config.PGN_DIR.mkdir(parents=True, exist_ok=True)
        self.current_round = 1
        self.teams: Dict[str, TeamTelemetry] = {}
        self.provider_teams: Dict[str, ProviderTeam] = {}
        self.active_matches: List[ChessMatch] = []
        self._load_state_or_init()

    def _load_state_or_init(self):
        if Config.STATE_FILE.exists():
            try:
                with open(Config.STATE_FILE) as f: data = json.load(f)
                self.current_round = data["current_round"]
                self.teams = {k: TeamTelemetry.from_dict(v) for k,v in data.get("teams",{}).items() if k in TEAM_NAMES}
                for pt in data.get("provider_teams",[]):
                    if pt["name"] in TEAM_NAMES: self.provider_teams[pt["name"]] = ProviderTeam.from_dict(pt)
                self.active_matches = [ChessMatch.restore(m) for m in data.get("active_matches",[])
                                        if m["white"] in TEAM_NAMES and m["black"] in TEAM_NAMES]
                for n in TEAM_NAMES:
                    if n not in self.teams: self.teams[n] = TeamTelemetry(name=n)
                    if n not in self.provider_teams: self.provider_teams[n] = ProviderTeam(n)
            except: self._init_fresh()
        else: self._init_fresh()

    def _init_fresh(self):
        self.teams = {n: TeamTelemetry(name=n) for n in TEAM_NAMES}
        self.provider_teams = {n: ProviderTeam(n) for n in TEAM_NAMES}
        self.active_matches = []

    def _save_state(self):
        with open(Config.STATE_FILE, "w") as f:
            json.dump({"current_round": self.current_round,
                        "teams": {k:t.to_dict() for k,t in self.teams.items()},
                        "provider_teams": [pt.to_dict() for pt in self.provider_teams.values()],
                        "active_matches": [m.to_dict() for m in self.active_matches]}, f, indent=4)
        with open(Config.STATS_FILE, "w") as f:
            json.dump([t.to_dict() for t in self.teams.values()], f, indent=4)

    def _rr_pairings(self):
        return [[("Team Google","Team SambaNova"),("Team Cohere","Team OpenRouter")],
                [("Team Google","Team Cohere"),("Team SambaNova","Team OpenRouter")],
                [("Team Google","Team OpenRouter"),("Team SambaNova","Team Cohere")]]

    def _start_round(self, rn):
        if rn == 1: self._init_fresh()
        self.current_round = rn; self.active_matches = []
        pp = self._rr_pairings()
        if rn > len(pp): self._print_standings(); self._final_report(); sys.exit()
        print(f"\n{Back.GREEN}{Fore.BLACK} ═══ ROUND {rn} ═══ {Style.RESET_ALL}")
        for i,(w,b) in enumerate(pp[rn-1], 1):
            print(f"  T{i}: {w} vs {b}")
            self.active_matches.append(ChessMatch(i, rn, w, b))
            self.teams[w].played_opponents.append(b); self.teams[b].played_opponents.append(w)
            self.teams[w].color_history.append("W"); self.teams[b].color_history.append("B")
        self._save_state()

    def _process_results(self):
        for m in self.active_matches:
            res = m.forced_result or m.board.result()
            w, b = m.white_name, m.black_name
            if res == "1-0": self.teams[w].points+=1; self.teams[w].wins+=1; self.teams[b].losses+=1
            elif res == "0-1": self.teams[b].points+=1; self.teams[b].wins+=1; self.teams[w].losses+=1
            else:
                for x in [w,b]: self.teams[x].points+=0.5; self.teams[x].draws+=1
        self._print_standings(); self._save_state()
        if self.current_round < Config.TOTAL_ROUNDS: self._start_round(self.current_round+1)
        else: print(f"\n{Back.YELLOW}{Fore.BLACK} 🏆 TOURNAMENT COMPLETE! 🏆{Style.RESET_ALL}"); self._final_report(); sys.exit()

    def _print_standings(self):
        st = sorted(self.teams.values(), key=lambda t:t.points, reverse=True)
        print(f"\n{Fore.YELLOW}{'═'*75}\n LEADERBOARD (R{self.current_round})\n{'═'*75}{Style.RESET_ALL}")
        print(f"{'Team':<20}|{'Pts':>5}|{'W-D-L':>7}|{'Moves':>6}|{'Ill%':>5}|{'Sw':>4}|{'S1':>4}|{'S2':>4}|{'AvgT':>6}")
        print('-'*75)
        for t in st:
            r = f"{t.wins}-{t.draws}-{t.losses}" if not t.is_disqualified else "DQ"
            print(f"{t.name:<20}|{t.points:>5.1f}|{r:>7}|{t.total_moves:>6}|{t.illegal_rate*100:>4.0f}%|"
                  f"{t.model_switches:>4}|{t.stage1_calls:>4}|{t.stage2_calls:>4}|{t.avg_time:>5.1f}s")

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

    def _cmd_mode(self):
        global CURRENT_MODE
        print(f"\n{Fore.YELLOW}Current mode: {CURRENT_MODE.value.upper()}{Style.RESET_ALL}\n")
        for m in GameMode:
            marker = "►" if m == CURRENT_MODE else " "
            print(f"  {marker} {m.value:<12} — {MODE_DESCRIPTIONS[m]}")
        choice = input(f"\nNew mode (user/economy/premium/terminator): ").strip().lower()
        for m in GameMode:
            if m.value == choice: CURRENT_MODE = m; print(f"{Fore.GREEN}Mode: {m.value.upper()}{Style.RESET_ALL}"); return
        print(f"{Fore.RED}Invalid mode.{Style.RESET_ALL}")

    def _cmd_key(self):
        print(f"\n{Fore.YELLOW}API Keys:{Style.RESET_ALL}")
        for k,v in KEYS.items(): print(f"  {k}: {v[:8]}...{v[-4:]}")
        t = input("Which key? (GOOGLE/COHERE/OPENROUTER/SAMBANOVA): ").upper()
        if t in KEYS:
            v = input(f"New key for {t}: ").strip()
            if v:
                KEYS[t] = v
                if t=="GOOGLE": os.environ["GEMINI_API_KEY"]=v
                if t=="COHERE": os.environ["COHERE_API_KEY"]=v
                print(f"{Fore.GREEN}Updated.{Style.RESET_ALL}")
        else: print(f"{Fore.RED}Invalid.{Style.RESET_ALL}")

    def _cmd_addmodel(self):
        print(f"\n{Fore.YELLOW}Add model to team chain{Style.RESET_ALL}")
        print("Teams:", ", ".join(TEAM_NAMES))
        tn = input("Team name: ").strip()
        if tn not in PROVIDER_CHAINS: print(f"{Fore.RED}Not found.{Style.RESET_ALL}"); return
        name = input("Display name (e.g. 'Gemini Pro'): ").strip()
        model = input("Model ID (e.g. 'gemini/gemini-2.5-pro'): ").strip()
        if name and model:
            PROVIDER_CHAINS[tn]["chain"].append({"name": name, "model": model})
            if tn in self.provider_teams: self.provider_teams[tn].chain.append({"name":name,"model":model})
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
                ex = ""
                if pt and i in pt.exhausted_until:
                    r = int(pt.exhausted_until[i]-time.time())
                    ex = f" {Fore.RED}({r}s){Style.RESET_ALL}" if r>0 else f" {Fore.GREEN}(ok){Style.RESET_ALL}"
                print(f"  {mk} {i}. {e['name']:<28} {e['model']}{ex}")

    def _cmd_modes(self):
        print(f"\n{Fore.YELLOW}═══ GAME MODES ═══{Style.RESET_ALL}\n")
        for m in GameMode:
            marker = f"{Fore.GREEN}► ACTIVE{Style.RESET_ALL}" if m==CURRENT_MODE else ""
            print(f"  {m.value.upper():<12} {marker}")
            print(f"    {MODE_DESCRIPTIONS[m]}\n")

    def _cmd_help(self):
        print(f"""
{Fore.YELLOW}═══ COMMANDS ═══{Style.RESET_ALL}
  {Fore.GREEN}start{Style.RESET_ALL}      Start/resume tournament
  {Fore.GREEN}test{Style.RESET_ALL}       Test all model connections
  {Fore.GREEN}top{Style.RESET_ALL}        Show leaderboard
  {Fore.GREEN}teams{Style.RESET_ALL}      Show team rosters & status
  {Fore.GREEN}logs{Style.RESET_ALL}       Show recent illegal moves
  {Fore.GREEN}mode{Style.RESET_ALL}       Change game mode (user/economy/premium/terminator)
  {Fore.GREEN}modes{Style.RESET_ALL}      Explain all modes
  {Fore.GREEN}key{Style.RESET_ALL}        Change API key
  {Fore.GREEN}addmodel{Style.RESET_ALL}   Add model to a team
  {Fore.GREEN}rmmodel{Style.RESET_ALL}    Remove model from a team
  {Fore.GREEN}reset{Style.RESET_ALL}      Wipe all data and restart
  {Fore.GREEN}exit{Style.RESET_ALL}       Quit
""")

    def run_main_loop(self):
        global CURRENT_MODE
        show_menu = True
        while True:
            if show_menu:
                draw_banner()
                st = "NEW" if not self.active_matches else f"ROUND {self.current_round}"
                print(f"{Fore.GREEN}Status: {st} | Mode: {CURRENT_MODE.value.upper()}{Style.RESET_ALL}")
                while True:
                    cmd = input(f"\n{Fore.CYAN}> {Style.RESET_ALL}").strip().lower()
                    if not cmd: continue
                    if cmd=="start":
                        if not self.active_matches: self._start_round(1)
                        show_menu = False; break
                    elif cmd=="test": self.test_connection()
                    elif cmd=="top": self._print_standings()
                    elif cmd=="teams": self._cmd_teams()
                    elif cmd=="logs":
                        p = Config.BASE_DIR/"hallucinations.log"
                        if p.exists():
                            for l in open(p,encoding="utf-8").readlines()[-15:]:
                                print(f"{Fore.RED}• {l.strip()}{Style.RESET_ALL}")
                        else: print("No errors yet.")
                    elif cmd=="mode": self._cmd_mode()
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
                                time.sleep(2)
                        if active == 0: self._process_results()
                except KeyboardInterrupt:
                    print("\nSaving..."); self._save_state(); sys.exit()

if __name__ == "__main__":
    TournamentManager().run_main_loop()
