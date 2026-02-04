# 🏆 LLM Chess Arena v3.0

**LLM Chess Arena** is a professional CLI-based tournament manager designed to benchmark Large Language Models (LLMs) through chess. It utilizes a **Swiss-system pairing algorithm** and is specifically optimized for a **Human-in-the-loop** workflow, allowing you to test top-tier models via their web interfaces without needing expensive API keys.

![Version](https://img.shields.io/badge/version-3.0.0-blue)
![Python](https://img.shields.io/badge/python-3.8%2B-green)
![License](https://img.shields.io/badge/license-MIT-orange)

## 🌟 1. Overview

While most AI benchmarks focus on text or code, chess is a perfect proxy for testing **spatial reasoning**, **sustained attention**, and **logical consistency**. 

This engine allows researchers to observe:
* How well models "see" the board without specialized chess engines.
* The frequency of **"Spatial Hallucinations"** (attempting illegal moves).
* How "Thinking" models utilize their processing time in complex strategic positions.

## 🚀 2. Key Features

*   **🏆 Swiss System Integration:** Automatically generates fair pairings based on current standings (Winners vs. Winners). No early eliminations—every model plays every round.
*   **🧠 Hallucination Tracker:** Every illegal move or syntax error is logged in `hallucinations.log` with a timestamp and FEN, providing a dataset for AI reliability research.
*   **📋 Turbo Workflow:** The script automatically copies the current FEN and formatted prompt to your clipboard (`pyperclip`). Just press `Ctrl+V` in your AI chat.
*   **💾 Persistent State:** Tournament data is saved after every single move. You can close the terminal and resume the tournament later exactly where you left off.
*   **📊 Professional Data Export:** 
    *   **PGN:** Full game histories compatible with Chess.com, Lichess, and Fritz.
    *   **JSON:** Final standings including "Time Spent" and "Error Rate" metrics.

## 🛠 3. Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/llm-chess-arena.git
   cd llm-chess-arena
2. Install dependencies:
   ```bash
   pip install python-chess pyperclip colorama
3. Run the manager:
    ```bash   
   python arena.py
📖 4. Tournament Workflow
Setup: Add your model names to the PARTICIPANTS list inside arena.py.
The Pipeline:
Select a Table ID (1-4).
The prompt is auto-copied. Paste it into your AI's chat (GPT, Claude, Gemini, etc.).
Type the AI's response (e.g., Nf3) back into the terminal.
Validation: The engine checks move legality instantly. If the AI "hallucinates," the error is logged, and you can request a retry.
Final Standings: After 3 rounds, the engine generates a final_standings.json leaderboard.
📜 5. The Story Behind the Project
This project was born from a casual chess match against my grandmother. I was using a "state-of-the-art" AI to assist, and it unexpectedly hung its Queen for no reason. This inspired me to build a system that objectively measures whether these "superintelligences" can actually follow rigid rules under pressure.
