import os
import sys
import shutil
import math
import io
import re
from pathlib import Path

try:
    import chess
    import chess.pgn
    import chess.engine
except ImportError:
    print("Не найден модуль `chess`. Пожалуйста, установите его: pip install chess")
    sys.exit(1)

try:
    from colorama import init, Fore, Style, Back
    init(autoreset=True)
except ImportError:
    print("Не найден модуль `colorama`. Пожалуйста, установите его: pip install colorama")
    sys.exit(1)

# --- SETTINGS ---
MIN_MOVES = 26  # Пропускать партии короче этого кол-ва ПОЛУХОДОВ (10 полных ходов)
ANALYSIS_DEPTH = 14 # Глубина Stockfish (выше - точнее, но медленнее)
CPL_CAP = 600   # Максимальная потеря за 1 ход (чтобы зевок не убивал всю статистику)
# ----------------

def get_accuracy(cpl):
    """
    Более мягкая конвертация CPL в %. 
    0 CPL = 100%
    50 CPL = ~77%
    100 CPL = ~60%
    200 CPL = ~36%
    """
    return 100.0 * math.exp(-0.005 * cpl)

def clean_pgn_content(text):
    """Удаляет проблемные комментарии и лишние скобки, которые ломают парсер."""
    # Удаляем всё внутри фигурных скобок {...}
    text = re.sub(r'\{.*?\}', '', text, flags=re.DOTALL)
    # Удаляем остаточные скобки, если они были вложенными
    text = text.replace('{', '').replace('}', '')
    return text

def analyze_game(pgn_file_path, engine):
    with open(pgn_file_path, 'r', encoding='utf-8') as f:
        raw_content = f.read()
    
    clean_content = clean_pgn_content(raw_content)
    game = chess.pgn.read_game(io.StringIO(clean_content))
    
    if not game:
        return None
    
    board = game.board()
    white_cpl_sum = 0
    black_cpl_sum = 0
    white_moves = 0
    black_moves = 0
    
    limit = chess.engine.Limit(depth=ANALYSIS_DEPTH)
    
    # Оценка стартовой позиции
    info = engine.analyse(board, limit)
    prev_eval = info["score"].pov(chess.WHITE).score(mate_score=2000)
    
    for move in game.mainline_moves():
        turn = board.turn
        board.push(move)
        
        info = engine.analyse(board, limit)
        curr_eval = info["score"].pov(chess.WHITE).score(mate_score=2000)
        
        if turn == chess.WHITE:
            # Ход белых: потеря = было - стало
            loss = prev_eval - curr_eval
            loss = max(0, min(CPL_CAP, loss))
            white_cpl_sum += loss
            white_moves += 1
        else:
            # Ход черных: потеря = стало - было (т.к. eval от лица белых)
            loss = curr_eval - prev_eval
            loss = max(0, min(CPL_CAP, loss))
            black_cpl_sum += loss
            black_moves += 1
            
        prev_eval = curr_eval
        
    white_acpl = white_cpl_sum / white_moves if white_moves > 0 else 0
    black_acpl = black_cpl_sum / black_moves if black_moves > 0 else 0
    
    white_acc = get_accuracy(white_acpl)
    black_acc = get_accuracy(black_acpl)
    
    return {
        "file": pgn_file_path,
        "white": game.headers.get("White", "Unknown"),
        "black": game.headers.get("Black", "Unknown"),
        "result": game.headers.get("Result", "*"),
        "white_acc": white_acc,
        "black_acc": black_acc,
        "white_acpl": white_acpl,
        "black_acpl": black_acpl,
        "total_acc": white_acc + black_acc,
        "moves": white_moves + black_moves
    }

def find_stockfish():
    # 1. Проверка корня и PATH
    for name in ["stockfish.exe", "stockfish", "stockfish-windows-x86-64-avx2.exe"]:
        if os.path.exists(name) and os.path.isfile(name):
            return os.path.abspath(name)
        path = shutil.which(name)
        if path:
            return path
            
    # 2. Проверка в папке 'stockfish'
    sf_dir = Path("stockfish")
    if sf_dir.exists() and sf_dir.is_dir():
        for f in sf_dir.glob("*.exe"):
            return str(f.absolute())
        for f in sf_dir.iterdir():
            if f.is_file() and "stockfish" in f.name.lower():
                return str(f.absolute())
    return None

def main():
    print(f"\n{Back.BLUE}{Fore.WHITE} ♔ LLM CHESS PGN ANALYZER v2.0 ♔ {Style.RESET_ALL}\n")
    
    engine_path = find_stockfish()
    if not engine_path:
        print(f"{Fore.RED}❌ Stockfish не найден!{Style.RESET_ALL}")
        return
        
    print(f"{Fore.GREEN}✓ Stockfish найден: {engine_path}{Style.RESET_ALL}")
    
    all_pgns = []
    for folder in Path(".").glob("tournament_data*"):
        if folder.is_dir():
            all_pgns.extend(list(folder.rglob("*.pgn")))
            
    if not all_pgns:
        print(f"{Fore.YELLOW}PGN файлы не найдены.{Style.RESET_ALL}")
        return
        
    print(f"{Fore.CYAN}Найдено PGN файлов: {len(all_pgns)}. Начинаю глубокий анализ...{Style.RESET_ALL}")
    
    try:
        engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    except Exception as e:
        print(f"{Fore.RED}❌ Ошибка запуска Stockfish: {e}{Style.RESET_ALL}")
        return

    analyzed_games = []
    skipped_short = 0
    
    for i, p in enumerate(all_pgns, 1):
        print(f"{Fore.MAGENTA}⏳ [{i}/{len(all_pgns)}] Анализ: {p.name[:40]}...{Style.RESET_ALL}", end="\r")
        try:
            # Предварительная проверка на длину без запуска движка
            with open(p, 'r', encoding='utf-8') as f:
                content = clean_pgn_content(f.read())
                g_peek = chess.pgn.read_game(io.StringIO(content))
                if g_peek:
                    m_count = sum(1 for _ in g_peek.mainline_moves())
                    if m_count < MIN_MOVES:
                        skipped_short += 1
                        continue
            
            res = analyze_game(p, engine)
            if res:
                analyzed_games.append(res)
        except Exception:
            pass
            
    engine.quit()
    print("\n" + " " * 70 + "\r", end="")
    
    if skipped_short:
        print(f"{Fore.YELLOW}Пропущено коротких партий (< {MIN_MOVES} полуходов): {skipped_short}{Style.RESET_ALL}")
    
    if not analyzed_games:
        print(f"{Fore.RED}❌ Нет данных для отображения.{Style.RESET_ALL}")
        return
        
    # Сортировка по суммарной точности
    analyzed_games.sort(key=lambda x: x["total_acc"], reverse=True)
    best = analyzed_games[0]
    
    print(f"\n{Back.GREEN}{Fore.BLACK} 🏆 РЕЗУЛЬТАТЫ (ТОП-1 ПАРТИЯ) 🏆 {Style.RESET_ALL}\n")
    print(f"📁 Файл:   {Fore.LIGHTBLACK_EX}{best['file']}{Style.RESET_ALL}")
    print(f"🤍 Белые:  {Fore.WHITE}{best['white']:<20} {Fore.GREEN}{best['white_acc']:.1f}% {Fore.BLACK}(ACPL: {best['white_acpl']:.0f}){Style.RESET_ALL}")
    print(f"🖤 Черные: {Fore.WHITE}{best['black']:<20} {Fore.GREEN}{best['black_acc']:.1f}% {Fore.BLACK}(ACPL: {best['black_acpl']:.0f}){Style.RESET_ALL}")
    print(f"📊 Итог:   {Fore.YELLOW}{best['result']} ({best['moves']} полуходов){Style.RESET_ALL}")
    print(f"🔥 Общая точность: {Fore.CYAN}{best['total_acc']/2:.1f}%{Style.RESET_ALL}")

if __name__ == '__main__':
    main()
