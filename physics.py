"""Simplified 2D (top-down) physics for the bowling lane.

World coordinates are metres: x is lateral (0 = lane centre, +x = right),
y runs from the foul line (0) toward the pin deck. Rendering applies its
own perspective projection; nothing in here touches pygame, so the module
can be simulated headlessly for AI training and unit tests.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# --- Lane geometry (roughly regulation, metres) ---
LANE_LENGTH = 18.29          # foul line to head pin
LANE_HALF_WIDTH = 0.5268
PIT_END = LANE_LENGTH + 1.15  # back of pin deck; things past this are gone

BALL_RADIUS = 0.108
PIN_RADIUS = 0.057
BALL_MASS = 6.35
PIN_MASS = 1.55

BALL_FRICTION = 0.30         # m/s^2 deceleration on the lane
PIN_FRICTION = 3.2           # pins scrub off speed quickly
RESTITUTION = 0.42
HOOK_ACCEL = 0.55            # lateral accel per unit of spin (dry lane)
OIL_HOOK_MULT = 0.15         # hook barely grips while the ball is on the oil
MIN_OIL, MAX_OIL = 7.5, 13.5  # per-match oil pattern length range (metres)

KNOCK_DISPLACEMENT = 0.075   # a pin displaced this far counts as down
SUBSTEP = 1.0 / 240.0
MAX_SIM_TIME = 9.0
REST_SPEED = 0.05

# Standard 10-pin rack. Pin 1 is the head pin; rows go back-left to right.
_PIN_SPACING = 0.3048
_ROW_DY = _PIN_SPACING * math.sqrt(3.0) / 2.0
PIN_SPOTS: tuple[tuple[float, float], ...] = (
    (0.0, LANE_LENGTH),                                          # 1
    (-_PIN_SPACING / 2, LANE_LENGTH + _ROW_DY),                  # 2
    (_PIN_SPACING / 2, LANE_LENGTH + _ROW_DY),                   # 3
    (-_PIN_SPACING, LANE_LENGTH + 2 * _ROW_DY),                  # 4
    (0.0, LANE_LENGTH + 2 * _ROW_DY),                            # 5
    (_PIN_SPACING, LANE_LENGTH + 2 * _ROW_DY),                   # 6
    (-1.5 * _PIN_SPACING, LANE_LENGTH + 3 * _ROW_DY),            # 7
    (-_PIN_SPACING / 2, LANE_LENGTH + 3 * _ROW_DY),              # 8
    (_PIN_SPACING / 2, LANE_LENGTH + 3 * _ROW_DY),               # 9
    (1.5 * _PIN_SPACING, LANE_LENGTH + 3 * _ROW_DY),             # 10
)

# Throw parameter ranges shared by the human controls and the AI action grid.
MIN_ANGLE_DEG = -4.0
MAX_ANGLE_DEG = 4.0
MIN_SPEED = 4.5
MAX_SPEED = 9.0
MAX_SPIN = 1.0


@dataclass
class ThrowParams:
    angle_deg: float
    speed: float
    spin: float  # -1..1, positive curves right


@dataclass
class Body:
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    radius: float = PIN_RADIUS
    mass: float = PIN_MASS
    active: bool = True

    @property
    def speed(self) -> float:
        return math.hypot(self.vx, self.vy)


@dataclass
class Pin:
    index: int
    body: Body
    home: tuple[float, float]
    down: bool = False


@dataclass
class ThrowOutcome:
    knocked: list[int] = field(default_factory=list)
    gutter: bool = False


class LaneSimulation:
    """Simulates one throw against the currently standing pins."""

    def __init__(self, standing: list[bool], oil_length: float = 0.0) -> None:
        self.pins: list[Pin] = [
            Pin(i, Body(x, y), (x, y))
            for i, (x, y) in enumerate(PIN_SPOTS)
            if standing[i]
        ]
        self.ball: Body | None = None
        self.oil_length = oil_length  # 0 = dry lane, full hook everywhere
        self.in_gutter = False
        self.time = 0.0
        self.done = False

    def throw(self, params: ThrowParams) -> None:
        a = math.radians(params.angle_deg)
        self.ball = Body(
            x=0.0, y=0.0,
            vx=math.sin(a) * params.speed,
            vy=math.cos(a) * params.speed,
            radius=BALL_RADIUS, mass=BALL_MASS,
        )
        self._spin = params.spin

    def step(self, dt: float) -> bool:
        """Advance the sim by dt seconds. Returns True once settled."""
        if self.done:
            return True
        remaining = dt
        while remaining > 0.0 and not self.done:
            h = min(SUBSTEP, remaining)
            self._substep(h)
            remaining -= h
        return self.done

    def outcome(self) -> ThrowOutcome:
        return ThrowOutcome(
            knocked=[p.index for p in self.pins if p.down],
            gutter=self.in_gutter,
        )

    # --- internals ---

    def _substep(self, dt: float) -> None:
        self.time += dt
        ball = self.ball

        if ball is not None and ball.active:
            if not self.in_gutter:
                grip = OIL_HOOK_MULT if ball.y < self.oil_length else 1.0
                ball.vx += self._spin * HOOK_ACCEL * grip * dt
            _apply_friction(ball, BALL_FRICTION, dt)
            ball.x += ball.vx * dt
            ball.y += ball.vy * dt

            if not self.in_gutter and abs(ball.x) > LANE_HALF_WIDTH:
                # Ball drops into the gutter and rides it straight.
                self.in_gutter = True
                ball.x = math.copysign(LANE_HALF_WIDTH + BALL_RADIUS * 0.6, ball.x)
                ball.vx = 0.0
            if ball.y > PIT_END or ball.speed < REST_SPEED:
                ball.active = False

        for pin in self.pins:
            b = pin.body
            if not b.active:
                continue
            _apply_friction(b, PIN_FRICTION, dt)
            b.x += b.vx * dt
            b.y += b.vy * dt
            if not pin.down and _dist(b.x, b.y, *pin.home) > KNOCK_DISPLACEMENT:
                pin.down = True
            # Off the deck or into the pit: gone (and certainly down).
            if b.y > PIT_END or b.y < LANE_LENGTH - 1.0 or abs(b.x) > LANE_HALF_WIDTH + 0.25:
                pin.down = True
                b.active = False

        # Collisions: ball-pin (never from the gutter), then pin-pin.
        if ball is not None and ball.active and not self.in_gutter:
            for pin in self.pins:
                if pin.body.active:
                    _collide(ball, pin.body)
        for i in range(len(self.pins)):
            bi = self.pins[i].body
            if not bi.active:
                continue
            for j in range(i + 1, len(self.pins)):
                bj = self.pins[j].body
                if bj.active:
                    _collide(bi, bj)

        self._check_done()

    def _check_done(self) -> None:
        if self.time > MAX_SIM_TIME:
            self.done = True
            return
        ball_settled = self.ball is None or not self.ball.active
        pins_settled = all(
            not p.body.active or p.body.speed < REST_SPEED for p in self.pins
        )
        if ball_settled and pins_settled:
            self.done = True


def _apply_friction(b: Body, decel: float, dt: float) -> None:
    s = b.speed
    if s <= 0.0:
        return
    drop = decel * dt
    scale = max(0.0, s - drop) / s
    b.vx *= scale
    b.vy *= scale


def _dist(x1: float, y1: float, x2: float, y2: float) -> float:
    return math.hypot(x2 - x1, y2 - y1)


def _collide(a: Body, b: Body) -> None:
    """Impulse resolution for two circles, with positional correction."""
    dx = b.x - a.x
    dy = b.y - a.y
    d = math.hypot(dx, dy)
    min_d = a.radius + b.radius
    if d >= min_d or d == 0.0:
        return
    nx, ny = dx / d, dy / d
    # Separate the overlap proportionally to inverse mass.
    inv_a, inv_b = 1.0 / a.mass, 1.0 / b.mass
    total_inv = inv_a + inv_b
    push = (min_d - d) / total_inv
    a.x -= nx * push * inv_a
    a.y -= ny * push * inv_a
    b.x += nx * push * inv_b
    b.y += ny * push * inv_b

    rel = (b.vx - a.vx) * nx + (b.vy - a.vy) * ny
    if rel > 0.0:
        return  # already separating
    j = -(1.0 + RESTITUTION) * rel / total_inv
    a.vx -= j * nx * inv_a
    a.vy -= j * ny * inv_a
    b.vx += j * nx * inv_b
    b.vy += j * ny * inv_b
