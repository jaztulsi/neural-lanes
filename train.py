"""Headless self-play training: the AI bowls full games as fast as possible.

Runs the same physics and learning loop as the live game, minus rendering,
so hundreds of games take minutes instead of hours. Progress persists to the
usual checkpoint — train here, then play against the improved AI in main.py.

    python train.py --games 200
"""
from __future__ import annotations

import argparse
import random
import time

from ai_agent import BowlingAgent, decode_action, encode_state, throw_reward
from game_state import BowlingGame
from physics import MAX_OIL, MIN_OIL, LaneSimulation

SIM_DT = 0.05  # coarse steps are fine headless; physics substeps internally


def play_game(agent: BowlingAgent, oil_len: float, learn: bool = True,
              greedy: bool = False) -> int:
    """One full 10-frame game of self-play; the agent learns every throw."""
    game = BowlingGame()
    pins = [True] * 10
    while not game.is_complete():
        frame, ball = game.frame_ball()
        fresh = game.needs_fresh_pins()
        if fresh:
            pins = [True] * 10
        # ponytail: score gap is 0 headless (no opponent); fine for throw choice
        state = encode_state(pins, frame, ball, 0, oil_len)
        action = agent.select_action(state, greedy=greedy)
        sim = LaneSimulation(pins, oil_len)
        sim.throw(decode_action(action))
        while not sim.step(SIM_DT):
            pass
        outcome = sim.outcome()
        knocked = len(outcome.knocked)
        for i in outcome.knocked:
            pins[i] = False
        game.roll(knocked)

        cleared = sum(pins) == 0
        is_strike = fresh and knocked == 10
        is_spare = not fresh and cleared
        reward = throw_reward(knocked, is_strike, is_spare, outcome.gutter)
        if game.is_complete():
            done, next_state = True, state  # done masks the bootstrap term
        else:
            nf, nb = game.frame_ball()
            done = nb == 0  # frame ended -> episode boundary for TD target
            next_pins = [True] * 10 if game.needs_fresh_pins() else pins
            next_state = encode_state(next_pins, nf, nb, 0, oil_len)
        if learn:
            agent.observe(state, action, reward, next_state, done)
    return game.score()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", type=int, default=100,
                        help="number of games to train (default 100)")
    parser.add_argument("--save-every", type=int, default=10,
                        help="checkpoint every N games (default 10)")
    parser.add_argument("--eval", type=int, metavar="N",
                        help="benchmark: N greedy games, no exploration, "
                             "no learning, nothing saved")
    args = parser.parse_args()

    agent = BowlingAgent()
    print(f"resuming at {agent.total_throws} throws / "
          f"{agent.games_played} games, epsilon {agent.epsilon:.3f}")
    if args.eval:
        scores = [play_game(agent, random.uniform(MIN_OIL, MAX_OIL),
                            learn=False, greedy=True)
                  for _ in range(args.eval)]
        avg = sum(scores) / len(scores)
        print(f"greedy policy over {args.eval} games: "
              f"avg {avg:.1f}  min {min(scores)}  max {max(scores)}")
        return
    start = time.time()
    recent: list[int] = []
    for g in range(1, args.games + 1):
        oil_len = random.uniform(MIN_OIL, MAX_OIL)
        score = play_game(agent, oil_len)
        agent.games_played += 1
        agent.stats["ai_high"] = max(agent.stats["ai_high"], score)
        agent.stats["ai_scores"].append(score)
        del agent.stats["ai_scores"][:-200]
        recent.append(score)
        if g % args.save_every == 0:
            agent.save()
            avg = sum(recent) / len(recent)
            rate = g / (time.time() - start)
            print(f"game {g}/{args.games}  avg score {avg:.1f}  "
                  f"best {max(recent)}  epsilon {agent.epsilon:.3f}  "
                  f"({rate:.1f} games/s)")
            recent.clear()
    agent.save()
    print(f"done: {agent.total_throws} total throws, "
          f"career high {agent.stats['ai_high']}")


if __name__ == "__main__":
    main()
