"""PyTorch Dueling DQN opponent with replay buffer and disk persistence.

No pygame imports here: the agent can be trained/evaluated headlessly.
"""
from __future__ import annotations

import math
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from physics import ThrowParams

# --- Action space: 9 angles x 5 powers x 3 spins = 135 discrete actions ---
ANGLES: np.ndarray = np.linspace(-2.7, 2.7, 9)
POWERS: np.ndarray = np.linspace(5.5, 8.5, 5)
SPINS: tuple[float, ...] = (-0.7, 0.0, 0.7)
N_ACTIONS = len(ANGLES) * len(POWERS) * len(SPINS)

STATE_DIM = 14  # 10 pin bits + frame + ball + score gap + oil length

GAMMA = 0.95
LR = 3e-4
BATCH_SIZE = 64
BUFFER_CAPACITY = 20_000
MIN_BUFFER = 96
TAU = 0.005              # soft target-network update rate
GRAD_UPDATES_PER_THROW = 4
EPS_START = 0.9
EPS_END = 0.06
EPS_DECAY = 500.0        # throws; e-folding scale for epsilon decay

CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_PATH = CHECKPOINT_DIR / "agent.pt"

Transition = tuple[np.ndarray, int, float, np.ndarray, bool]


def encode_state(
    pins_standing: list[bool],
    frame_idx: int,
    ball_idx: int,
    score_gap: int,
    oil_length: float,
) -> np.ndarray:
    """State vector: pin bits, normalized frame/ball, clipped score gap, oil."""
    vec = np.zeros(STATE_DIM, dtype=np.float32)
    vec[:10] = [1.0 if p else 0.0 for p in pins_standing]
    vec[10] = frame_idx / 9.0
    vec[11] = ball_idx / 2.0
    vec[12] = float(np.clip(score_gap / 60.0, -1.0, 1.0))
    vec[13] = oil_length / 18.29  # lane length; the oil pattern this match
    return vec


def decode_action(index: int) -> ThrowParams:
    n_spin = len(SPINS)
    n_power = len(POWERS)
    spin = SPINS[index % n_spin]
    power = float(POWERS[(index // n_spin) % n_power])
    angle = float(ANGLES[index // (n_spin * n_power)])
    return ThrowParams(angle_deg=angle, speed=power, spin=spin)


class DuelingDQN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(STATE_DIM, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.value = nn.Linear(128, 1)
        self.advantage = nn.Linear(128, N_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        v = self.value(h)
        a = self.advantage(h)
        return v + a - a.mean(dim=1, keepdim=True)


class ReplayBuffer:
    def __init__(self, capacity: int = BUFFER_CAPACITY) -> None:
        self.data: deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self.data.append(t)

    def __len__(self) -> int:
        return len(self.data)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, ...]:
        batch = random.sample(self.data, batch_size)
        states = torch.from_numpy(np.stack([t[0] for t in batch]))
        actions = torch.tensor([t[1] for t in batch], dtype=torch.long)
        rewards = torch.tensor([t[2] for t in batch], dtype=torch.float32)
        next_states = torch.from_numpy(np.stack([t[3] for t in batch]))
        dones = torch.tensor([t[4] for t in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones


class BowlingAgent:
    """Online-learning DQN opponent, persisted across sessions."""

    def __init__(self, checkpoint_path: Path = CHECKPOINT_PATH) -> None:
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cpu")  # tiny net; CPU is plenty
        self.policy = DuelingDQN().to(self.device)
        self.target = DuelingDQN().to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR)
        self.buffer = ReplayBuffer()
        self.total_throws = 0
        self.games_played = 0
        self.loss_history: deque[float] = deque(maxlen=300)
        self.reward_history: deque[float] = deque(maxlen=300)
        # career match record (human vs AI), persisted with the checkpoint
        self.stats = {"you_wins": 0, "ai_wins": 0, "draws": 0,
                      "you_high": 0, "ai_high": 0}
        if checkpoint_path.exists():
            try:
                self.load()
            except Exception as e:
                print(f"checkpoint incompatible ({e}); starting fresh")
                self.buffer = ReplayBuffer()

    # --- policy ---

    @property
    def epsilon(self) -> float:
        return EPS_END + (EPS_START - EPS_END) * math.exp(
            -self.total_throws / EPS_DECAY
        )

    def select_action(self, state: np.ndarray, greedy: bool = False) -> int:
        if not greedy and random.random() < self.epsilon:
            return random.randrange(N_ACTIONS)
        with torch.no_grad():
            q = self.policy(torch.from_numpy(state).unsqueeze(0))
        return int(q.argmax(dim=1).item())

    # --- learning ---

    def observe(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> float | None:
        """Store one throw's transition and run gradient updates.

        Returns the mean TD loss for this update round (None if the buffer
        is still warming up).
        """
        self.buffer.push((state, action, reward, next_state, done))
        self.total_throws += 1
        self.reward_history.append(reward)
        losses = [
            loss for _ in range(GRAD_UPDATES_PER_THROW)
            if (loss := self.train_step()) is not None
        ]
        if not losses:
            return None
        mean_loss = float(np.mean(losses))
        self.loss_history.append(mean_loss)
        return mean_loss

    def train_step(self) -> float | None:
        if len(self.buffer) < MIN_BUFFER:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample(BATCH_SIZE)
        q = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN: policy net picks the action, target net values it.
            next_actions = self.policy(next_states).argmax(dim=1, keepdim=True)
            next_q = self.target(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + GAMMA * (1.0 - dones) * next_q
        loss = nn.functional.smooth_l1_loss(q, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy.parameters(), 5.0)
        self.optimizer.step()
        self._soft_update()
        return float(loss.item())

    def _soft_update(self) -> None:
        with torch.no_grad():
            for tp, pp in zip(self.target.parameters(), self.policy.parameters()):
                tp.mul_(1.0 - TAU).add_(pp, alpha=TAU)

    # --- persistence ---

    def save(self) -> None:
        CHECKPOINT_DIR.mkdir(exist_ok=True)
        torch.save(
            {
                "policy": self.policy.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "buffer": list(self.buffer.data),
                "total_throws": self.total_throws,
                "games_played": self.games_played,
                "loss_history": list(self.loss_history),
                "reward_history": list(self.reward_history),
                "stats": self.stats,
            },
            self.checkpoint_path,
        )

    def load(self) -> None:
        ckpt = torch.load(
            self.checkpoint_path, map_location=self.device, weights_only=False
        )
        self.policy.load_state_dict(ckpt["policy"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.buffer.data.extend(ckpt["buffer"])
        self.total_throws = ckpt["total_throws"]
        self.games_played = ckpt["games_played"]
        self.loss_history.extend(ckpt["loss_history"])
        self.reward_history.extend(ckpt["reward_history"])
        self.stats.update(ckpt.get("stats", {}))

    def reset_learning(self) -> None:
        """Wipe all learned progress (fresh network, empty buffer, no file)."""
        self.policy = DuelingDQN().to(self.device)
        self.target = DuelingDQN().to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=LR)
        self.buffer = ReplayBuffer()
        self.total_throws = 0
        self.games_played = 0
        self.loss_history.clear()
        self.reward_history.clear()
        self.checkpoint_path.unlink(missing_ok=True)


def throw_reward(knocked: int, is_strike: bool, is_spare: bool, gutter: bool) -> float:
    """Reward for one throw: pins + strike/spare bonus, gutter penalty."""
    reward = float(knocked)
    if is_strike:
        reward += 5.0
    elif is_spare:
        reward += 3.0
    if gutter and knocked == 0:
        reward -= 2.0
    return reward
