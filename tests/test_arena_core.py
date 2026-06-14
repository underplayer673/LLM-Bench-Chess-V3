import importlib.util
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("chess")
pytest.importorskip("litellm")
pytest.importorskip("colorama")


ROOT = Path(__file__).resolve().parents[1]
ARENA_PATH = ROOT / "python arena_v3_API_new_versu.py"
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def arena():
    spec = importlib.util.spec_from_file_location("arena_new", ARENA_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_extract_move_prefers_uci_over_bare_square(arena):
    assert arena.extract_move("I choose e2e4") == "e2e4"


def test_extract_move_normalizes_promotion(arena):
    assert arena.extract_move("MOVE: e8q") == "e8=Q"
    assert arena.extract_move("MOVE: exd8n+") == "exd8=N+"


def test_provider_team_roundtrip_keeps_custom_chain(arena):
    team = arena.ProviderTeam("Team GitHub")
    team.chain.append({"name": "Custom Model", "model": "github/custom"})
    restored = arena.ProviderTeam.from_dict(team.to_dict())

    assert restored.chain[-1]["name"] == "Custom Model"
    assert restored.chain[-1]["model"] == "github/custom"


def test_chess_match_restore_from_moves_without_pgn(arena, tmp_path, monkeypatch):
    monkeypatch.setattr(arena.Config, "BASE_DIR", tmp_path)
    match = arena.ChessMatch(1, 1, "Team Google", "Team GitHub", 1)
    match.board.push_san("e4")
    data = match.to_dict()
    match.pgn_path.unlink(missing_ok=True)

    restored = arena.ChessMatch.restore(data)

    assert restored.board.fen() == match.board.fen()


def test_chess_match_restore_from_fen_without_pgn_or_moves(arena, tmp_path, monkeypatch):
    monkeypatch.setattr(arena.Config, "BASE_DIR", tmp_path)
    match = arena.ChessMatch(1, 1, "Team Google", "Team GitHub", 1)
    match.board.push_san("d4")
    data = match.to_dict()
    data.pop("moves")
    match.pgn_path.unlink(missing_ok=True)

    restored = arena.ChessMatch.restore(data)

    assert restored.board.fen() == match.board.fen()


def test_atomic_write_json(arena, tmp_path):
    path = tmp_path / "state.json"
    arena.atomic_write_json(path, {"ok": True})

    assert json.loads(path.read_text(encoding="utf-8")) == {"ok": True}
    assert not (tmp_path / "state.json.tmp").exists()
