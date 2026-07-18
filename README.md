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

## The AI

- **State** (13 dims): 10 pin bits, frame, ball-in-frame, score gap vs. you.
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
- `checkpoints/` — saved model + replay buffer (gitignored)
