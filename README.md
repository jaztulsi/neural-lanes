# Neural Lanes

Two-player bowling: you vs. a self-learning AI opponent. PyGame handles
rendering and simplified 2D physics; the AI is a Dueling Double-DQN
(PyTorch) that learns online — after every single throw — and persists its
weights and replay buffer to `checkpoints/agent.pt`, so it keeps improving
across sessions.

## Run

```sh
python -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python main.py
```

## Controls

| Key | Action |
|-----|--------|
| ← / → | aim angle |
| Z / X | spin (hook left / right) |
| SPACE (hold + release) | power meter, release to throw |
| ENTER | start match / confirm |
| R (menu) | reset the AI's learning (asks to confirm) |
| ESC | back to menu / quit (progress auto-saves) |

Tip: straight balls top out around 8-9 pins — strikes need a hooked ball
into the 1-3 (or 1-2) pocket, just like real bowling. The AI has to
discover this too.

## Lane oil

Every match rolls a random **oil pattern** (7.5–13.5 m, shown as a sheen on
the lane). The ball barely hooks while it's on the oil and only grips the
dry backend — short oil means big early hooks, long oil plays nearly
straight. The aim preview accounts for it, so you have to find a new line
every match; oil length is part of the AI's state, so it must adapt too.

## Extras

- Strike-streak banners (STRIKE! / DOUBLE! / TURKEY!), split detection
  (SPLIT!, with a special 7-10 callout), confetti, ball trail, screen flash.
- Synthesized sound effects (ball rumble, pin crash scaled by pins hit,
  strike fanfare, gutter thud) — generated with numpy at startup, no asset
  files (`sounds.py`).
- Career stats on the menu: your win-loss record vs. the AI and both high
  scores, persisted with the AI checkpoint.

## The AI

- **State** (14 dims): 10 pin bits, frame, ball-in-frame, score gap vs. you,
  oil pattern length.
- **Actions**: 9 angles × 5 powers × 3 spins = 135 discrete throws.
- **Reward**: pins knocked, +5 strike, +3 spare, −2 gutter.
- Epsilon-greedy exploration decays with total career throws; the on-screen
  "AI BRAIN" panel shows epsilon, TD-loss trend, and reward trend live.

## Layout

- `main.py` — game loop, match flow, PyGame setup
- `physics.py` — headless ball/pin simulation
- `game_state.py` — 10-frame bowling scoring
- `ai_agent.py` — DQN, replay buffer, training, persistence
- `human_controller.py` — keyboard input
- `ui.py` — all rendering
- `sounds.py` — numpy-synthesized sound effects
- `checkpoints/` — saved model + replay buffer (gitignored)
