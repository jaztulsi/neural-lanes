"""Synthesized sound effects — no asset files, everything generated with numpy.

If the mixer can't initialize (no audio device), every play call is a no-op.
"""
from __future__ import annotations

import numpy as np
import pygame

RATE = 22050


def _sound(samples: np.ndarray, volume: float = 0.5) -> pygame.mixer.Sound:
    arr = (np.clip(samples, -1.0, 1.0) * 32767 * 0.8).astype(np.int16)
    channels = pygame.mixer.get_init()[2]  # mixer may force stereo
    if channels > 1:
        arr = np.repeat(arr[:, None], channels, axis=1)
    snd = pygame.sndarray.make_sound(np.ascontiguousarray(arr))
    snd.set_volume(volume)
    return snd


def _tone(freq: float, dur: float, decay: float = 6.0) -> np.ndarray:
    t = np.arange(int(RATE * dur)) / RATE
    return np.sin(2 * np.pi * freq * t) * np.exp(-decay * t)


def _noise(dur: float, decay: float = 8.0) -> np.ndarray:
    t = np.arange(int(RATE * dur)) / RATE
    return np.random.uniform(-1, 1, len(t)) * np.exp(-decay * t)


def _mix(*parts: np.ndarray) -> np.ndarray:
    out = np.zeros(max(len(p) for p in parts))
    for p in parts:
        out[:len(p)] += p
    return out


class Sounds:
    def __init__(self) -> None:
        self.enabled = True
        try:
            pygame.mixer.init(frequency=RATE, size=-16, channels=1)
            self.ok = True
        except pygame.error:
            self.ok = False
            return
        rng = np.random.default_rng(0)
        # Ball roll: low rumble (smoothed noise), fading in then out.
        rumble = np.convolve(rng.uniform(-1, 1, RATE), np.ones(64) / 64, "same")
        env = np.minimum(np.linspace(0, 4, RATE), np.linspace(2, 0, RATE))
        self._roll = _sound(rumble * np.clip(env, 0, 1), 0.35)
        # Pin crash: bright noise burst with a woody thump underneath.
        self._crash = _sound(_mix(_noise(0.5, 7), 0.6 * _tone(180, 0.3, 12)), 0.6)
        # Strike fanfare: rising major arpeggio.
        notes = [_tone(f, 0.42, 5) for f in (523, 659, 784, 1047)]
        fan = np.zeros(int(RATE * 0.75))
        for i, n in enumerate(notes):
            j = int(RATE * 0.11 * i)
            fan[j:j + len(n)] += n[:len(fan) - j]
        self._strike = _sound(fan, 0.5)
        # Spare: two quick notes.
        sp = np.zeros(int(RATE * 0.4))
        for i, n in enumerate((_tone(587, 0.3, 7), _tone(880, 0.3, 7))):
            j = int(RATE * 0.09 * i)
            sp[j:j + len(n)] += n[:len(sp) - j]
        self._spare = _sound(sp, 0.5)
        # Gutter: sad low thud.
        self._gutter = _sound(_tone(110, 0.45, 5) + _tone(82, 0.45, 5), 0.4)

    @property
    def on(self) -> bool:
        return self.ok and self.enabled

    def toggle(self) -> bool:
        """Flip mute; returns True if sound is now on."""
        self.enabled = not self.enabled
        return self.on

    def roll(self) -> None:
        if self.on:
            self._roll.play()

    def crash(self, pins: int) -> None:
        if self.on and pins > 0:
            self._crash.set_volume(min(1.0, 0.25 + pins * 0.08))
            self._crash.play()

    def strike(self) -> None:
        if self.on:
            self._strike.play()

    def spare(self) -> None:
        if self.on:
            self._spare.play()

    def gutter(self) -> None:
        if self.on:
            self._gutter.play()
