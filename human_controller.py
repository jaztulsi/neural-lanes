"""Keyboard input handling for the human player's throw."""
from __future__ import annotations

import pygame

from physics import (
    MAX_ANGLE_DEG,
    MAX_SPEED,
    MAX_SPIN,
    MIN_ANGLE_DEG,
    MIN_SPEED,
    ThrowParams,
)

ANGLE_RATE = 3.0   # degrees per second while holding left/right
SPIN_RATE = 1.2    # spin units per second while holding Z/X
POWER_RATE = 1.1   # power meter sweeps 0 -> 1 in ~0.9s, then back down


class HumanController:
    """Aim with arrows, spin with Z/X, hold SPACE to charge, release to throw."""

    def __init__(self) -> None:
        self.angle_deg = 0.0
        self.spin = 0.0
        self.power = 0.0          # 0..1 meter position
        self.charging = False
        self._power_dir = 1.0
        self._released: ThrowParams | None = None

    def reset_throw(self) -> None:
        self.power = 0.0
        self.charging = False
        self._power_dir = 1.0
        self._released = None

    def handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
            if not self.charging:
                self.charging = True
                self.power = 0.0
                self._power_dir = 1.0
        elif event.type == pygame.KEYUP and event.key == pygame.K_SPACE:
            if self.charging:
                self.charging = False
                self._released = ThrowParams(
                    angle_deg=self.angle_deg,
                    speed=MIN_SPEED + self.power * (MAX_SPEED - MIN_SPEED),
                    spin=self.spin,
                )

    def update(self, dt: float) -> None:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            self.angle_deg = max(MIN_ANGLE_DEG, self.angle_deg - ANGLE_RATE * dt)
        if keys[pygame.K_RIGHT]:
            self.angle_deg = min(MAX_ANGLE_DEG, self.angle_deg + ANGLE_RATE * dt)
        if keys[pygame.K_z]:
            self.spin = max(-MAX_SPIN, self.spin - SPIN_RATE * dt)
        if keys[pygame.K_x]:
            self.spin = min(MAX_SPIN, self.spin + SPIN_RATE * dt)
        if self.charging:
            self.power += self._power_dir * POWER_RATE * dt
            if self.power >= 1.0:
                self.power, self._power_dir = 1.0, -1.0
            elif self.power <= 0.0:
                self.power, self._power_dir = 0.0, 1.0

    def take_throw(self) -> ThrowParams | None:
        """Returns the throw once SPACE has been released, else None."""
        params, self._released = self._released, None
        return params
