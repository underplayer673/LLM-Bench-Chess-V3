# LLM Chess Arena API

Локальная CLI-арена для шахматных турниров между LLM-провайдерами. Проект проверяет, как модели держат правила шахмат, выбирают легальные ходы, переживают rate limit и меняются по failover-цепочке.

## Состав

- `python arena_v3_API_new_versu.py` - основная API-арена: партии, failover, ELO команд и моделей, continuous tournaments.
- `python arena_v3_API.py` - старая API-версия.
- `python arena_v3.py` - ручная human-in-the-loop версия.
- `keys.py` - безопасная загрузка и сохранение API-ключей в `keys_local.json`.
- `scout.py` - проверка доступности моделей из `scout_models.json`.
- `analysis.py` - анализ PGN через Stockfish.
- `stockfish/` - локальный Stockfish и его исходники/документация.
- `tournament_data*/` - результаты запусков, PGN, JSON-аналитика и hallucination logs.

## Установка

```bash
python -m pip install -r requirements.txt
```

Для тестов:

```bash
python -m pip install -r requirements-dev.txt
python -m pytest
```

## Запуск

```bash
python "python arena_v3_API_new_versu.py"
```

При первом запуске программа спросит API-ключи. Они сохраняются в `keys_local.json`, который должен оставаться локальным и не попадать в репозиторий.

## Команды Внутри Арены

- `start` - начать или продолжить турнир.
- `test` - проверить подключение к моделям.
- `top` - таблица текущего турнира.
- `elo` - ELO моделей.
- `teamelo` - ELO команд.
- `teams` - цепочки моделей и cooldown.
- `logs` - последние нелегальные ходы.
- `mode`, `submode`, `ormode` - режим игры и выбора моделей.
- `key` - редактировать ключи.
- `delays` - изменить задержки и cooldown.
- `reset` - удалить `tournament_data` и начать заново.

## Безопасность

- Не храните реальные ключи в `keys.py`.
- Не коммитьте `keys_local.json` и `scout_keys.json`.
- Не публикуйте `tournament_data*`, если в логах могут быть приватные prompt/ответы или внутренние имена моделей.

## Ограничения

- Это исследовательский CLI-инструмент, не production-сервис.
- Результаты зависят от доступности API, лимитов провайдеров и текущих версий моделей.
- Для строгого бенчмарка фиксируйте дату, список моделей, режим, prompt, seed/температуру и полный state.
