"""Entry point: pygame loop and match flow (human vs. learning AI)."""
from __future__ import annotations

import sys

import pygame

from ai_agent import BowlingAgent, decode_action, encode_state, throw_reward
from game_state import BowlingGame
from human_controller import HumanController
from physics import LaneSimulation, ThrowParams
from ui import COL_ACCENT, COL_AI, COL_BG, COL_HUMAN, Renderer, SCREEN_W, SCREEN_H

FPS = 60
TIME_SCALE = 1.35        # sim seconds per real second (keeps throws snappy)
AI_THINK_TIME = 0.9
POST_ROLL_TIME = 1.5
SAVE_EVERY_THROWS = 20

HUMAN, AI = 0, 1


class App:
    def __init__(self) -> None:
        pygame.init()
        pygame.display.set_caption("Neural Lanes — you vs. a learning AI")
        self.screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
        self.clock = pygame.time.Clock()
        self.renderer = Renderer(self.screen)
        self.agent = BowlingAgent()
        self.ctrl = HumanController()
        self.scene = "menu"          # menu | playing | game_over
        self.confirm_reset = False
        self.last_ai_action: str | None = None
        self._new_match()

    # --- match setup / flow ---

    def _new_match(self) -> None:
        self.games = [BowlingGame(), BowlingGame()]
        self.names = ["YOU", "AI"]
        self.pins: list[list[bool]] = [[True] * 10, [True] * 10]
        self.current = HUMAN
        self.sim = LaneSimulation(self.pins[HUMAN])
        self.phase = "aim"           # aim | ai_think | rolling | post
        self.phase_timer = 0.0
        self.banner: tuple[str, tuple[int, int, int]] | None = None
        self.banner_timer = 0.0
        self.pending_ai: tuple | None = None  # (state, action, fresh, standing)
        self.ctrl.reset_throw()
        self._enter_turn()

    def _enter_turn(self) -> None:
        """Prepare the current player's next roll (rack fresh pins if due)."""
        game = self.games[self.current]
        if game.needs_fresh_pins():
            self.pins[self.current] = [True] * 10
        self.sim = LaneSimulation(self.pins[self.current])
        if self.current == HUMAN:
            self.phase = "aim"
            self.ctrl.reset_throw()
        else:
            self.phase = "ai_think"
            self.phase_timer = 0.0

    def _launch(self, params: ThrowParams) -> None:
        self.sim.throw(params)
        self.phase = "rolling"

    def _ai_throw(self) -> None:
        game = self.games[AI]
        frame, ball = game.frame_ball()
        gap = self.games[AI].score() - self.games[HUMAN].score()
        state = encode_state(self.pins[AI], frame, ball, gap)
        action = self.agent.select_action(state)
        params = decode_action(action)
        fresh = game.needs_fresh_pins()
        standing = sum(self.pins[AI])
        self.pending_ai = (state, action, fresh, standing)
        spin_lbl = "L" if params.spin < 0 else "R" if params.spin > 0 else "·"
        self.last_ai_action = (
            f"threw {params.angle_deg:+.1f}° pow {params.speed:.1f} spin {spin_lbl}"
        )
        self._launch(params)

    def _resolve_roll(self) -> None:
        outcome = self.sim.outcome()
        knocked = len(outcome.knocked)
        game = self.games[self.current]
        frame_before = game.frame_ball()[0]
        was_fresh = game.needs_fresh_pins()
        for i in outcome.knocked:
            self.pins[self.current][i] = False
        game.roll(knocked)

        cleared = sum(self.pins[self.current]) == 0
        is_strike = was_fresh and knocked == 10
        is_spare = not was_fresh and cleared
        if is_strike:
            self._set_banner("STRIKE!", COL_ACCENT)
        elif is_spare:
            self._set_banner("SPARE!", COL_ACCENT)
        elif outcome.gutter and knocked == 0:
            self._set_banner("gutter...", (150, 150, 160))
        else:
            self._set_banner(f"{knocked} pin{'s' if knocked != 1 else ''}",
                             COL_HUMAN if self.current == HUMAN else COL_AI)

        if self.current == AI and self.pending_ai is not None:
            self._ai_learn(knocked, is_strike, is_spare, outcome.gutter)

        self.phase = "post"
        self.phase_timer = 0.0
        self._frame_done = game.is_complete() or game.frame_ball()[0] != frame_before

    def _ai_learn(self, knocked: int, is_strike: bool, is_spare: bool,
                  gutter: bool) -> None:
        state, action, _fresh, _standing = self.pending_ai
        self.pending_ai = None
        reward = throw_reward(knocked, is_strike, is_spare, gutter)
        game = self.games[AI]
        if game.is_complete():
            done = True
            next_state = state  # unused: done masks the bootstrap term
        else:
            frame, ball = game.frame_ball()
            done = ball == 0  # frame ended -> episode boundary for TD target
            next_pins = [True] * 10 if game.needs_fresh_pins() else self.pins[AI]
            gap = self.games[AI].score() - self.games[HUMAN].score()
            next_state = encode_state(next_pins, frame, ball, gap)
        self.agent.observe(state, action, reward, next_state, done)
        if self.agent.total_throws % SAVE_EVERY_THROWS == 0:
            self.agent.save()

    def _advance_after_post(self) -> None:
        if all(g.is_complete() for g in self.games):
            self.scene = "game_over"
            self.agent.games_played += 1
            self.agent.save()
            return
        if self._frame_done:
            other = 1 - self.current
            if not self.games[other].is_complete():
                self.current = other
        self._enter_turn()

    def _set_banner(self, msg: str, color) -> None:
        self.banner = (msg, color)
        self.banner_timer = 0.0

    # --- event handling ---

    def _handle_event(self, event: pygame.event.Event) -> None:
        if event.type == pygame.QUIT:
            self._quit()
        if event.type != pygame.KEYDOWN and self.scene != "playing":
            return
        if self.scene == "menu":
            if self.confirm_reset:
                if event.key == pygame.K_y:
                    self.agent.reset_learning()
                    self.confirm_reset = False
                elif event.key in (pygame.K_n, pygame.K_ESCAPE):
                    self.confirm_reset = False
            elif event.key == pygame.K_RETURN:
                self._new_match()
                self.scene = "playing"
            elif event.key == pygame.K_r:
                self.confirm_reset = True
            elif event.key == pygame.K_ESCAPE:
                self._quit()
        elif self.scene == "playing":
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.agent.save()
                self.scene = "menu"
            elif self.phase == "aim":
                self.ctrl.handle_event(event)
        elif self.scene == "game_over":
            if event.key in (pygame.K_RETURN, pygame.K_ESCAPE):
                self.scene = "menu"

    def _quit(self) -> None:
        self.agent.save()
        pygame.quit()
        sys.exit(0)

    # --- per-frame update ---

    def _update(self, dt: float) -> None:
        if self.scene != "playing":
            return
        self.banner_timer += dt
        if self.phase == "aim":
            self.ctrl.update(dt)
            params = self.ctrl.take_throw()
            if params is not None:
                self._launch(params)
        elif self.phase == "ai_think":
            self.phase_timer += dt
            if self.phase_timer >= AI_THINK_TIME:
                self._ai_throw()
        elif self.phase == "rolling":
            if self.sim.step(dt * TIME_SCALE):
                self._resolve_roll()
        elif self.phase == "post":
            self.phase_timer += dt
            if self.phase_timer >= POST_ROLL_TIME:
                self._advance_after_post()

    # --- drawing ---

    def _draw(self) -> None:
        r = self.renderer
        if self.scene == "menu":
            r.draw_menu(self.agent, self.confirm_reset)
            pygame.display.flip()
            return
        self.screen.fill(COL_BG)
        r.draw_lane()
        r.draw_sim(self.sim)
        active = None if self.scene == "game_over" else self.current
        r.draw_scoreboard(self.games, self.names, active)
        r.draw_ai_panel(self.agent, self.last_ai_action)
        if self.scene == "game_over":
            r.draw_game_over(self.games, self.names)
        else:
            if self.phase == "aim":
                r.draw_aim(self.ctrl)
                r.draw_status("←/→ aim   Z/X spin   hold SPACE for power, "
                              "release to throw   ESC menu")
            elif self.phase == "ai_think":
                r.draw_status("AI is thinking...")
            if self.banner and self.banner_timer < 1.6:
                r.draw_banner(*self.banner)
        pygame.display.flip()

    # --- main loop ---

    def run(self) -> None:
        while True:
            dt = self.clock.tick(FPS) / 1000.0
            for event in pygame.event.get():
                self._handle_event(event)
            self._update(min(dt, 0.05))
            self._draw()


if __name__ == "__main__":
    App().run()
