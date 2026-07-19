"""All pygame rendering: lane, pins, scoreboard, HUD, menus."""
from __future__ import annotations

import math

import pygame

from ai_agent import BowlingAgent
from game_state import BowlingGame
from human_controller import HumanController
from physics import (
    BALL_RADIUS,
    HOOK_ACCEL,
    LANE_HALF_WIDTH,
    OIL_HOOK_MULT,
    PIT_END,
    LaneSimulation,
    MAX_ANGLE_DEG,
)

SCREEN_W, SCREEN_H = 960, 720

# Lane projection: world y=0 (foul line) at the bottom, pins near the top.
NEAR_Y, FAR_Y = 672, 225
NEAR_HW, FAR_HW = 235, 92
CX = SCREEN_W // 2

COL_BG = (18, 18, 26)
COL_LANE = (176, 138, 92)
COL_LANE_OIL = (191, 156, 112)  # subtle sheen where the oil pattern lies
COL_LANE_EDGE = (120, 90, 56)
COL_GUTTER = (52, 52, 62)
COL_PIN = (245, 245, 240)
COL_PIN_DOWN = (140, 135, 128)
COL_PIN_STRIPE = (200, 40, 50)
COL_BALL = (40, 90, 200)
COL_TEXT = (230, 230, 235)
COL_DIM = (140, 140, 155)
COL_ACCENT = (255, 200, 60)
COL_GRID = (70, 70, 85)
COL_HUMAN = (90, 180, 255)
COL_AI = (255, 120, 120)
COL_LOSS = (255, 140, 90)
COL_REWARD = (120, 220, 140)


def project(x: float, y: float) -> tuple[float, float, float]:
    """World (x, y) -> (screen_x, screen_y, scale) with a fake perspective."""
    t = max(0.0, min(1.0, y / PIT_END)) ** 0.88
    sy = NEAR_Y + (FAR_Y - NEAR_Y) * t
    half_w = NEAR_HW + (FAR_HW - NEAR_HW) * t
    sx = CX + (x / LANE_HALF_WIDTH) * half_w
    return sx, sy, half_w / NEAR_HW


class Renderer:
    def __init__(self, screen: pygame.Surface) -> None:
        self.screen = screen
        self.font = pygame.font.SysFont("menlo", 15)
        self.font_small = pygame.font.SysFont("menlo", 12)
        self.font_big = pygame.font.SysFont("menlo", 34, bold=True)
        self.font_mid = pygame.font.SysFont("menlo", 20, bold=True)

    # --- text helpers ---

    def text(self, s: str, x: int, y: int, color=COL_TEXT, font=None,
             center: bool = False) -> None:
        surf = (font or self.font).render(s, True, color)
        rect = surf.get_rect(center=(x, y)) if center else surf.get_rect(topleft=(x, y))
        self.screen.blit(surf, rect)

    # --- lane & entities ---

    def draw_lane(self, oil_length: float = 0.0) -> None:
        lane_edge = LANE_HALF_WIDTH
        gutter_edge = LANE_HALF_WIDTH * 1.28
        pts = [project(-gutter_edge, 0), project(gutter_edge, 0),
               project(gutter_edge, PIT_END), project(-gutter_edge, PIT_END)]
        pygame.draw.polygon(self.screen, COL_GUTTER, [(p[0], p[1]) for p in pts])
        pts = [project(-lane_edge, 0), project(lane_edge, 0),
               project(lane_edge, PIT_END), project(-lane_edge, PIT_END)]
        pygame.draw.polygon(self.screen, COL_LANE, [(p[0], p[1]) for p in pts])
        if oil_length > 0.0:
            pts = [project(-lane_edge, 0), project(lane_edge, 0),
                   project(lane_edge, oil_length), project(-lane_edge, oil_length)]
            pygame.draw.polygon(self.screen, COL_LANE_OIL, [(p[0], p[1]) for p in pts])
            a, b = project(-lane_edge, oil_length), project(lane_edge, oil_length)
            pygame.draw.line(self.screen, COL_LANE_EDGE, (a[0], a[1]), (b[0], b[1]), 1)
            self.text(f"oil {oil_length:.1f}m", int(b[0]) + 8, int(b[1]) - 6,
                      COL_DIM, self.font_small)
        # Board seams for depth.
        for i in range(1, 8):
            x = -lane_edge + i * (2 * lane_edge / 8)
            a, b = project(x, 0), project(x, PIT_END)
            pygame.draw.line(self.screen, COL_LANE_EDGE, (a[0], a[1]), (b[0], b[1]), 1)
        # Aiming arrows.
        for i, x in enumerate((-0.35, -0.175, 0.0, 0.175, 0.35)):
            ax, ay, s = project(x, 4.5 + abs(i - 2) * 0.5)
            pygame.draw.polygon(self.screen, COL_LANE_EDGE,
                                [(ax, ay - 8 * s), (ax - 5 * s, ay + 4 * s),
                                 (ax + 5 * s, ay + 4 * s)])
        # Pit shadow.
        a, b = project(-gutter_edge, PIT_END), project(gutter_edge, PIT_END)
        pygame.draw.rect(self.screen, (10, 10, 14),
                         (a[0], a[1] - 26, b[0] - a[0], 26))

    def draw_sim(self, sim: LaneSimulation) -> None:
        # Far-to-near so nearer objects draw on top.
        for pin in sorted(sim.pins, key=lambda p: -p.body.y):
            if not pin.body.active:
                continue
            x, y, s = project(pin.body.x, pin.body.y)
            self._draw_pin(x, y, s, pin.down)
        if sim.ball is not None and sim.ball.active:
            self.draw_ball(sim.ball.x, sim.ball.y)

    def _draw_pin(self, x: float, y: float, s: float, down: bool) -> None:
        color = COL_PIN_DOWN if down else COL_PIN
        w, h = 11 * s, 26 * s
        if down:
            body = pygame.Rect(0, 0, h, w)
            body.center = (x, y)
            pygame.draw.ellipse(self.screen, color, body)
        else:
            body = pygame.Rect(0, 0, w * 2, h)
            body.midbottom = (x, y + 4 * s)
            pygame.draw.ellipse(self.screen, color, body)
            pygame.draw.ellipse(self.screen, (60, 60, 70), body, 1)
            pygame.draw.rect(self.screen, COL_PIN_STRIPE,
                             (body.x + 1, body.y + h * 0.30, body.w - 2, max(1, 3 * s)))

    def draw_ball(self, wx: float, wy: float) -> None:
        x, y, s = project(wx, wy)
        r = max(3, BALL_RADIUS / LANE_HALF_WIDTH * NEAR_HW * s)
        pygame.draw.circle(self.screen, COL_BALL, (x, y), r)
        pygame.draw.circle(self.screen, (120, 160, 240), (x - r * 0.3, y - r * 0.3),
                           max(1, r * 0.3))

    def draw_aim(self, ctrl: HumanController, oil_length: float = 0.0) -> None:
        """Dotted preview line plus angle/spin/power readouts."""
        a = math.radians(ctrl.angle_deg)
        v = 7.0  # preview at mid power; friction ignored
        steps = 24
        for i in range(1, steps):
            d = i / steps * (PIT_END * 0.97)
            t_dry = max(0.0, (d - oil_length) / v)
            t_oil = min(d, oil_length) / v
            hook = 0.5 * ctrl.spin * HOOK_ACCEL * (
                OIL_HOOK_MULT * t_oil ** 2 + t_dry * (t_dry + 2 * OIL_HOOK_MULT * t_oil))
            wx = math.sin(a) * d + hook
            if abs(wx) > LANE_HALF_WIDTH:
                break
            x, y, s = project(wx, d)
            if i % 2 == 0:
                pygame.draw.circle(self.screen, COL_ACCENT, (x, y), max(1, 2.5 * s))
        self.draw_ball(0.0, 0.0)

        # Left-side control panel.
        px, py = 18, 250
        self.text("YOUR THROW", px, py, COL_HUMAN, self.font)
        self.text(f"angle {ctrl.angle_deg:+.1f}°  (←/→)", px, py + 26)
        spin_lbl = "left" if ctrl.spin < -0.05 else "right" if ctrl.spin > 0.05 else "none"
        self.text(f"spin  {ctrl.spin:+.2f} {spin_lbl}  (Z/X)", px, py + 48)
        self.text("hold SPACE = power", px, py + 70, COL_DIM)
        # Power meter.
        mx, my, mw, mh = px, py + 95, 160, 16
        pygame.draw.rect(self.screen, COL_GRID, (mx, my, mw, mh), 1)
        fill = int((mw - 2) * ctrl.power)
        c = (int(80 + 175 * ctrl.power), int(220 - 120 * ctrl.power), 60)
        pygame.draw.rect(self.screen, c, (mx + 1, my + 1, fill, mh - 2))

    def draw_trail(self, trail: list[tuple[float, float]]) -> None:
        for i, (wx, wy) in enumerate(trail):
            x, y, s = project(wx, wy)
            fade = (i + 1) / len(trail)
            c = (int(40 * fade), int(90 * fade), int(200 * fade))
            pygame.draw.circle(self.screen, c, (x, y),
                               max(1, 5 * s * fade),
                               )

    def draw_particles(self, particles: list[list]) -> None:
        for x, y, _vx, _vy, life, color in particles:
            if life > 0:
                pygame.draw.circle(self.screen, color, (x, y), max(1, int(3 * life)))

    def draw_flash(self, strength: float) -> None:
        v = int(110 * strength)
        self.screen.fill((v, v, v // 2), special_flags=pygame.BLEND_RGB_ADD)

    # --- scoreboard ---

    def draw_scoreboard(self, games: list[BowlingGame], names: list[str],
                        active: int | None) -> None:
        top, row_h = 12, 64
        name_w, cell_w = 64, 84
        x0 = (SCREEN_W - name_w - 10 * cell_w) // 2
        for row, (game, name) in enumerate(zip(games, names)):
            y = top + row * (row_h + 8)
            color = COL_HUMAN if row == 0 else COL_AI
            if active == row:
                pygame.draw.rect(self.screen, color,
                                 (x0 - 4, y - 3, name_w + 10 * cell_w + 8, row_h + 6), 2)
            self.text(name, x0 + name_w // 2, y + row_h // 2, color,
                      self.font, center=True)
            views = game.frame_views()
            for f in range(10):
                cx = x0 + name_w + f * cell_w
                pygame.draw.rect(self.screen, COL_GRID, (cx, y, cell_w, row_h), 1)
                view = views[f] if f < len(views) else None
                # Roll mark boxes along the top edge.
                n_boxes = 3 if f == 9 else 2
                bw = 20
                for b in range(n_boxes):
                    bx = cx + cell_w - (n_boxes - b) * bw
                    pygame.draw.rect(self.screen, COL_GRID, (bx, y, bw, 20), 1)
                    if view and b < len(view.marks):
                        self.text(view.marks[b], bx + bw // 2, y + 10,
                                  COL_TEXT, self.font, center=True)
                if view and view.cumulative is not None:
                    self.text(str(view.cumulative), cx + cell_w // 2, y + 44,
                              COL_TEXT, self.font_mid, center=True)
                self.text(str(f + 1), cx + 4, y + 4, COL_DIM, self.font_small)

    # --- AI HUD ---

    def draw_ai_panel(self, agent: BowlingAgent, last_action: str | None) -> None:
        px, pw = SCREEN_W - 232, 218
        py = 250
        pygame.draw.rect(self.screen, (26, 26, 38), (px, py, pw, 240))
        pygame.draw.rect(self.screen, COL_GRID, (px, py, pw, 240), 1)
        self.text("AI BRAIN", px + 10, py + 8, COL_AI)
        eps = agent.epsilon
        mode = "exploring" if eps > 0.35 else "mixed" if eps > 0.12 else "exploiting"
        self.text(f"epsilon {eps:.3f} · {mode}", px + 10, py + 32, font=self.font_small)
        # Epsilon bar: how much of its behaviour is random.
        pygame.draw.rect(self.screen, COL_GRID, (px + 10, py + 52, pw - 20, 8), 1)
        pygame.draw.rect(self.screen, COL_ACCENT,
                         (px + 11, py + 53, int((pw - 22) * eps), 6))
        self.text(f"throws trained  {agent.total_throws}", px + 10, py + 68)
        self.text(f"games played    {agent.games_played}", px + 10, py + 88)
        self._sparkline("loss (TD error)", list(agent.loss_history),
                        px + 10, py + 112, pw - 20, 46, COL_LOSS)
        self._sparkline("reward / throw", list(agent.reward_history),
                        px + 10, py + 174, pw - 20, 46, COL_REWARD)
        if last_action:
            self.text(last_action, px + 10, py + 226, COL_DIM, self.font_small)

    def _sparkline(self, label: str, data: list[float], x: int, y: int,
                   w: int, h: int, color) -> None:
        self.text(label, x, y, COL_DIM, self.font_small)
        box = pygame.Rect(x, y + 14, w, h - 14)
        pygame.draw.rect(self.screen, COL_GRID, box, 1)
        if len(data) < 2:
            self.text("warming up...", box.centerx, box.centery, COL_DIM,
                      self.font_small, center=True)
            return
        lo, hi = min(data), max(data)
        span = (hi - lo) or 1.0
        pts = [
            (box.x + 2 + i * (box.w - 4) / (len(data) - 1),
             box.bottom - 2 - (v - lo) / span * (box.h - 4))
            for i, v in enumerate(data)
        ]
        pygame.draw.lines(self.screen, color, False, pts, 1)
        label_val = self.font_small.render(f"{data[-1]:.2f}", True, color)
        self.screen.blit(label_val, label_val.get_rect(topright=(box.right, y)))

    # --- overlays ---

    def draw_banner(self, message: str, color=COL_ACCENT) -> None:
        self.text(message, CX, 195, color, self.font_big, center=True)

    def draw_status(self, message: str) -> None:
        self.text(message, CX, SCREEN_H - 12, COL_DIM, self.font, center=True)

    def draw_menu(self, agent: BowlingAgent, confirm_reset: bool) -> None:
        self.screen.fill(COL_BG)
        self.text("NEURAL LANES", CX, 150, COL_ACCENT, self.font_big, center=True)
        self.text("you vs. a self-learning AI bowler", CX, 195, COL_DIM,
                  self.font, center=True)
        if confirm_reset:
            self.text("Really wipe the AI's learning?", CX, 300, COL_AI,
                      self.font_mid, center=True)
            self.text("Y — yes, wipe it      N — cancel", CX, 340, COL_TEXT,
                      self.font, center=True)
        else:
            for i, line in enumerate((
                "ENTER  —  start match",
                "A      —  watch AI vs AI",
                "R      —  reset AI learning",
                "ESC    —  quit (progress auto-saves)",
            )):
                self.text(line, CX - 130, 280 + i * 30)
        scores = agent.stats.get("ai_scores", [])
        if len(scores) >= 2:
            self._sparkline("AI learning curve (score per game)",
                            [float(s) for s in scores],
                            CX - 130, 396, 260, 64, COL_AI)
        self.text(
            f"AI: {agent.total_throws} throws trained | "
            f"{agent.games_played} games | epsilon {agent.epsilon:.3f}",
            CX, 470, COL_DIM, self.font, center=True)
        st = agent.stats
        self.text(
            f"career: you {st['you_wins']} — {st['ai_wins']} AI "
            f"({st['draws']} draws) | high scores: you {st['you_high']} · "
            f"AI {st['ai_high']}",
            CX, 500, COL_DIM, self.font, center=True)
        self.text("aim ←/→   spin Z/X   power: hold+release SPACE   "
                  "TAB fast-forward   M mute",
                  CX, 545, COL_DIM, self.font, center=True)
        self.text("each match rolls a random oil pattern — the ball only hooks "
                  "past the oil", CX, 570, COL_DIM, self.font_small, center=True)

    def draw_game_over(self, games: list[BowlingGame], names: list[str]) -> None:
        h, a = games[0].score(), games[1].score()
        if h == a:
            msg, col = "DRAW", COL_ACCENT
        else:
            w = 0 if h > a else 1
            msg = "YOU WIN!" if names[w] == "YOU" else f"{names[w]} WINS"
            col = COL_HUMAN if w == 0 else COL_AI
        self.draw_banner(f"{msg}  {h} - {a}", col)
        self.draw_status("ENTER — back to menu")
