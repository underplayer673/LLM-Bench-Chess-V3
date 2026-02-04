"""
LLM Chess Tournament Engine (Swiss System Edition)
Author: Senior Python Dev
Version: 3.0.0
Description: 3-Round Swiss System Tournament with automatic pairings, scoring, and leaderboard.
"""

import os
import sys
import time
import json
import logging
import datetime
import random
import shutil
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Tuple, Set
from enum import Enum

import chess
import chess.pgn
import pyperclip
from colorama import init, Fore, Style, Back

# --- INITIALIZATION ---
init(autoreset=True)

# --- CONFIGURATION ---
class Config:
    TOTAL_ROUNDS = 3
    BASE_DIR = Path("tournament_data")
    LOGS_DIR = BASE_DIR / "logs"
    STATE_FILE = BASE_DIR / "tournament_state.json"
    STATS_FILE = BASE_DIR / "final_standings.json"
    HALLUCINATION_LOG = LOGS_DIR / "hallucinations.log"
    
   # --- CONFIGURATION ---
class Config:
    TOTAL_ROUNDS = 3  # Количество туров
    BASE_DIR = Path("tournament_data")
    LOGS_DIR = BASE_DIR / "logs"
    STATE_FILE = BASE_DIR / "tournament_state.json"
    STATS_FILE = BASE_DIR / "final_standings.json"
    HALLUCINATION_LOG = LOGS_DIR / "hallucinations.log"
    
    PARTICIPANTS = [
        "Gemini 3 Pro", "Kimi K2",
        "Claude Opus 4.5", "GPT-5.2",
        "GLM-4.7", "Grok 4.1 Fast",
        "DeepSeek V3.2", "Qwen3 Max Thinking"
    ]

# --- UTILS ---
class Logger:
    @staticmethod
    def setup():
        Config.LOGS_DIR.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def log_hallucination(model: str, move: str, fen: str, error_type: str, round_num: int, table_id: int):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        entry = (f"[{timestamp}] R{round_num}|T{table_id} | Model: {model} | "
                 f"Bad Move: '{move}' | Error: {error_type} | FEN: {fen}\n")
        
        with open(Config.HALLUCINATION_LOG, "a", encoding="utf-8") as f:
            f.write(entry)

# --- GLOBAL PLAYER TRACKING ---

@dataclass
class GlobalPlayerStats:
    name: str
    points: float = 0.0
    games_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    total_time: float = 0.0
    total_errors: int = 0
    played_opponents: List[str] = field(default_factory=list)
    color_history: List[str] = field(default_factory=list) # 'W', 'B'

    def to_dict(self):
        return asdict(self)
    
    @staticmethod
    def from_dict(data):
        return GlobalPlayerStats(**data)

# --- MATCH STRUCTURES ---

class MatchState(Enum):
    ACTIVE = "Active"
    COMPLETED = "Completed"

@dataclass
class MatchPlayer:
    name: str
    color: str
    time_spent: float = 0.0
    errors: int = 0

class ChessMatch:
    def __init__(self, table_id: int, round_num: int, white_name: str, black_name: str):
        self.table_id = table_id
        self.round_num = round_num
        self.white = MatchPlayer(white_name, "White")
        self.black = MatchPlayer(black_name, "Black")
        self.board = chess.Board()
        self.state = MatchState.ACTIVE
        
        # Папка для раунда
        self.pgn_dir = Config.BASE_DIR / "pgn" / f"Round_{round_num}"
        self.pgn_dir.mkdir(parents=True, exist_ok=True)
        self.pgn_path = self.pgn_dir / f"T{table_id}_{white_name}_vs_{black_name}.pgn"
        
        self._ensure_pgn_init()

    def _ensure_pgn_init(self):
        if not self.pgn_path.exists():
            self._save_pgn()

    @property
    def current_player(self) -> MatchPlayer:
        return self.white if self.board.turn == chess.WHITE else self.black

    def get_prompt_text(self) -> str:
        color_ru = "Белые" if self.board.turn == chess.WHITE else "Черные"
        history = []
        try:
            if len(self.board.move_stack) > 0:
                history = [m.uci() for m in self.board.move_stack[-3:]]
        except: pass

        context = f"Предыдущие ходы: {', '.join(history)}. " if history else "Начало партии. "
        return (f"CONTEXT: {context}\nFEN: {self.board.fen()}\n"
                f"TASK: Твой ход {color_ru}. Оцени позицию и напиши ТОЛЬКО ход в формате SAN.")

    def process_move(self, move_str: str, duration: float) -> Tuple[bool, str]:
        player = self.current_player
        try:
            move = self.board.parse_san(move_str)
        except ValueError:
            player.errors += 1
            Logger.log_hallucination(player.name, move_str, self.board.fen(), "Syntax Error", self.round_num, self.table_id)
            return False, "Syntax Error"

        if move not in self.board.legal_moves:
            player.errors += 1
            Logger.log_hallucination(player.name, move_str, self.board.fen(), "Illegal Move", self.round_num, self.table_id)
            return False, "Illegal Move"

        self.board.push(move)
        player.time_spent += duration
        self._check_game_over()
        self._save_pgn()
        return True, "OK"

    def undo(self):
        if len(self.board.move_stack) > 0:
            self.board.pop()
            self.state = MatchState.ACTIVE
            self._save_pgn()

    def _check_game_over(self):
        if self.board.is_game_over():
            self.state = MatchState.COMPLETED

    def _save_pgn(self):
        game = chess.pgn.Game()
        game.headers["Event"] = f"LLM Swiss Tournament R{self.round_num}"
        game.headers["Site"] = f"Table {self.table_id}"
        game.headers["Date"] = datetime.datetime.now().strftime("%Y.%m.%d")
        game.headers["White"] = self.white.name
        game.headers["Black"] = self.black.name
        game.headers["Result"] = self.board.result()
        if self.board.root().fen() != chess.STARTING_FEN: game.setup(self.board.root())
        
        node = game
        for move in self.board.move_stack: node = node.add_variation(move)
        with open(self.pgn_path, "w", encoding="utf-8") as f: print(game, file=f, end="\n\n")

    def to_dict(self):
        return {
            "table_id": self.table_id,
            "round_num": self.round_num,
            "white": asdict(self.white),
            "black": asdict(self.black),
            "fen": self.board.fen(),
            "state": self.state.value,
            "moves": [m.uci() for m in self.board.move_stack]
        }

    @staticmethod
    def restore(data):
        m = ChessMatch(data['table_id'], data['round_num'], data['white']['name'], data['black']['name'])
        m.white.time_spent = data['white']['time_spent']
        m.white.errors = data['white']['errors']
        m.black.time_spent = data['black']['time_spent']
        m.black.errors = data['black']['errors']
        m.state = MatchState(data['state'])
        
        # Logic to restore moves from PGN or JSON
        moves_restored = False
        if m.pgn_path.exists():
            try:
                with open(m.pgn_path) as f: 
                    game = chess.pgn.read_game(f)
                    if game and list(game.mainline_moves()):
                        m.board = game.board()
                        for mv in game.mainline_moves(): m.board.push(mv)
                        moves_restored = True
            except: pass
        
        if not moves_restored and "moves" in data:
            m.board.reset()
            for uci in data["moves"]: m.board.push(chess.Move.from_uci(uci))
        
        return m

# --- TOURNAMENT ENGINE ---

class TournamentManager:
    def __init__(self):
        Logger.setup()
        self.current_round = 1
        self.players: Dict[str, GlobalPlayerStats] = {}
        self.active_matches: List[ChessMatch] = []
        self.is_finished = False
        
        self._load_state_or_init()

    def _load_state_or_init(self):
        if Config.STATE_FILE.exists():
            try:
                with open(Config.STATE_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                self.current_round = data.get("current_round", 1)
                self.is_finished = data.get("is_finished", False)
                self.players = {name: GlobalPlayerStats.from_dict(p_data) for name, p_data in data["players"].items()}
                self.active_matches = [ChessMatch.restore(m) for m in data["active_matches"]]
                return
            except Exception as e:
                print(f"{Fore.RED}Save file corrupted ({e}). Starting fresh.{Style.RESET_ALL}")

        # Fresh Start
        for name in Config.PARTICIPANTS:
            self.players[name] = GlobalPlayerStats(name=name)
        self._start_round(1)

    def _start_round(self, round_num):
        self.current_round = round_num
        self.active_matches = []
        pairings = self._generate_swiss_pairings(round_num)
        
        for i, (p1_name, p2_name) in enumerate(pairings, 1):
            # p1 is White, p2 is Black
            self.active_matches.append(ChessMatch(i, round_num, p1_name, p2_name))
            
            # Update history
            self.players[p1_name].played_opponents.append(p2_name)
            self.players[p1_name].color_history.append('W')
            self.players[p2_name].played_opponents.append(p1_name)
            self.players[p2_name].color_history.append('B')
        
        self._save_state()

    def _generate_swiss_pairings(self, round_num):
        # 1. Сортировка по очкам
        sorted_players = sorted(self.players.values(), key=lambda p: p.points, reverse=True)
        names_pool = [p.name for p in sorted_players]
        pairs = []

        if round_num == 1:
            # Fixed pairings for Round 1 (as requested originally)
            # Assuming PARTICIPANTS list is ordered: 0vs1, 2vs3...
            for i in range(0, len(Config.PARTICIPANTS), 2):
                pairs.append((Config.PARTICIPANTS[i], Config.PARTICIPANTS[i+1]))
            return pairs

        # Swiss Logic
        while len(names_pool) >= 2:
            p1_name = names_pool.pop(0)
            p1_obj = self.players[p1_name]
            
            # Search for best opponent
            chosen_opp = None
            for i, cand_name in enumerate(names_pool):
                if cand_name not in p1_obj.played_opponents:
                    chosen_opp = names_pool.pop(i)
                    break
            
            if not chosen_opp:
                chosen_opp = names_pool.pop(0) # Fallback: repeat opponent if no choice
            
            # Color balancing
            opp_obj = self.players[chosen_opp]
            p1_w = p1_obj.color_history.count('W')
            opp_w = opp_obj.color_history.count('W')
            
            if p1_w > opp_w:
                pairs.append((chosen_opp, p1_name))
            else:
                pairs.append((p1_name, chosen_opp))
                
        return pairs

    def _process_round_results(self):
        """Called when all games in round are done"""
        print(f"\n{Fore.GREEN}Processing results for Round {self.current_round}...{Style.RESET_ALL}")
        
        for m in self.active_matches:
            w_name = m.white.name
            b_name = m.black.name
            res = m.board.result()
            
            # Update Time & Errors
            self.players[w_name].total_time += m.white.time_spent
            self.players[w_name].total_errors += m.white.errors
            self.players[b_name].total_time += m.black.time_spent
            self.players[b_name].total_errors += m.black.errors
            
            self.players[w_name].games_played += 1
            self.players[b_name].games_played += 1

            # Update Points
            if res == "1-0":
                self.players[w_name].points += 1.0
                self.players[w_name].wins += 1
                self.players[b_name].losses += 1
            elif res == "0-1":
                self.players[b_name].points += 1.0
                self.players[b_name].wins += 1
                self.players[w_name].losses += 1
            elif res == "1/2-1/2":
                self.players[w_name].points += 0.5
                self.players[b_name].points += 0.5
                self.players[w_name].draws += 1
                self.players[b_name].draws += 1

        if self.current_round < Config.TOTAL_ROUNDS:
            self._start_round(self.current_round + 1)
        else:
            self.is_finished = True
            self._save_state()
            self._export_final_standings()

    def _save_state(self):
        data = {
            "current_round": self.current_round,
            "is_finished": self.is_finished,
            "players": {name: p.to_dict() for name, p in self.players.items()},
            "active_matches": [m.to_dict() for m in self.active_matches]
        }
        with open(Config.STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def _export_final_standings(self):
        sorted_ranking = sorted(self.players.values(), key=lambda p: p.points, reverse=True)
        
        report = []
        for rank, p in enumerate(sorted_ranking, 1):
            report.append({
                "Rank": rank,
                "Model": p.name,
                "Points": p.points,
                "Record": f"+{p.wins}={p.draws}-{p.losses}",
                "TotalTime": round(p.total_time, 2),
                "Errors": p.total_errors
            })
        
        with open(Config.STATS_FILE, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=4)

    # --- UI LOOP ---
    def run(self):
        while True:
            self._clear()
            self._print_header()
            
            if self.is_finished:
                self._print_final_standings()
                input(f"\n{Fore.GREEN}Tournament Completed. Press Enter to exit.{Style.RESET_ALL}")
                sys.exit()

            # Check if round is done
            all_done = all(m.state == MatchState.COMPLETED for m in self.active_matches)
            
            if all_done:
                print(f"\n{Fore.GREEN}!!! ROUND {self.current_round} COMPLETED !!!{Style.RESET_ALL}")
                cmd = input("Start next round / finish? (y/n): ").lower()
                if cmd == 'y':
                    self._process_round_results()
                    continue
                else:
                    sys.exit()

            # Show active tables
            self._print_matches()
            
            print(f"\n{Fore.CYAN}Controls:{Style.RESET_ALL} [1-{len(self.active_matches)}] Open Table | [x] Exit")
            choice = input(f"{Style.BRIGHT}> {Style.RESET_ALL}").strip()
            
            if choice == 'x':
                self._save_state()
                sys.exit()
            elif choice.isdigit() and 1 <= int(choice) <= len(self.active_matches):
                self._run_table_view(int(choice) - 1)

    def _run_table_view(self, idx):
        # ... Similar table view logic as before ...
        current_idx = idx
        while True:
            m = self.active_matches[current_idx]
            self._clear()
            
            # Header
            print(f"{Back.BLUE}{Fore.WHITE} ROUND {self.current_round} | TABLE {m.table_id} {Style.RESET_ALL}")
            print(f"{Fore.WHITE}White: {m.white.name} ({self.players[m.white.name].points} pts)")
            print(f"{Fore.BLUE}Black: {m.black.name} ({self.players[m.black.name].points} pts)")
            print("-" * 40)
            
            if m.state == MatchState.COMPLETED:
                 print(f"{Fore.RED}GAME OVER: {m.board.result()}{Style.RESET_ALL}")
            else:
                p = m.current_player
                color_fmt = Fore.WHITE if p.color == "White" else Fore.BLUE
                print(f"To Move: {color_fmt}{Style.BRIGHT}{p.color.upper()}{Style.RESET_ALL}")
                print(f"FEN: {Style.DIM}{m.board.fen()}{Style.RESET_ALL}")
            
            print(f"\n{Fore.CYAN}Nav:{Style.RESET_ALL} [n]ext | [p]rev | [b]ack | [u]ndo")
            
            if m.state == MatchState.ACTIVE:
                prompt = m.get_prompt_text()
                pyperclip.copy(prompt)
                print(f"{Fore.MAGENTA}[ PROMPT COPIED ]{Style.RESET_ALL}")
                
                start = time.time()
                move = input(f"{Fore.YELLOW}Move > {Style.RESET_ALL}").strip()
                dur = time.time() - start
                
                if move == 'b': break
                if move == 'n': 
                    current_idx = (current_idx + 1) % len(self.active_matches)
                    continue
                if move == 'p':
                    current_idx = (current_idx - 1) % len(self.active_matches)
                    continue
                if move == 'u':
                    m.undo()
                    self._save_state()
                    continue

                success, msg = m.process_move(move, dur)
                if not success:
                    print(f"{Fore.RED}ERROR: {msg}{Style.RESET_ALL}")
                    input("Enter to retry...")
                else:
                    self._save_state()
            else:
                cmd = input("cmd > ").strip()
                if cmd == 'b': break
                if cmd == 'n': current_idx = (current_idx + 1) % len(self.active_matches)
                if cmd == 'p': current_idx = (current_idx - 1) % len(self.active_matches)

    def _clear(self): os.system('cls' if os.name == 'nt' else 'clear')

    def _print_header(self):
        print(f"{Style.BRIGHT}=== LLM SWISS TOURNAMENT (Round {self.current_round}/{Config.TOTAL_ROUNDS}) ==={Style.RESET_ALL}")

    def _print_matches(self):
        print(f"{'ID':<3} {'White':<20} {'Black':<20} {'Status':<10} {'Result'}")
        print("-" * 65)
        for m in self.active_matches:
            st = "Done" if m.state == MatchState.COMPLETED else "Active"
            col = Fore.RED if st == "Done" else Fore.GREEN
            res = m.board.result() if st == "Done" else "*"
            print(f"{m.table_id:<3} {m.white.name:<20} {m.black.name:<20} {col}{st:<10}{Style.RESET_ALL} {res}")

    def _print_final_standings(self):
        sorted_p = sorted(self.players.values(), key=lambda p: p.points, reverse=True)
        print(f"\n{Back.YELLOW}{Fore.BLACK} FINAL STANDINGS {Style.RESET_ALL}")
        print(f"{'#':<3} {'Model':<20} {'Pts':<5} {'W-D-L':<10} {'Errs':<5}")
        print("-" * 50)
        for i, p in enumerate(sorted_p, 1):
            rec = f"{p.wins}-{p.draws}-{p.losses}"
            medal = "🥇" if i==1 else "🥈" if i==2 else "🥉" if i==3 else "  "
            print(f"{i:<3} {medal} {p.name:<18} {p.points:<5} {rec:<10} {p.total_errors:<5}")

# --- ENTRY ---
if __name__ == "__main__":
    try:
        app = TournamentManager()
        app.run()
    except KeyboardInterrupt:
        print("\nSaved & Exiting.")
        sys.exit()