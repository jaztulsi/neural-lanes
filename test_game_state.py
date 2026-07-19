"""Self-checks for scoring, frame flow, and split detection.

Run directly (`python test_game_state.py`) or via pytest.
"""
from game_state import BowlingGame, is_split


def game_of(*rolls: int) -> BowlingGame:
    g = BowlingGame()
    for r in rolls:
        g.roll(r)
    return g


def test_perfect_game():
    g = game_of(*[10] * 12)
    assert g.score() == 300
    assert g.is_complete()


def test_all_spares():
    g = game_of(*[5] * 21)
    assert g.score() == 150
    assert g.is_complete()


def test_gutter_game():
    g = game_of(*[0] * 20)
    assert g.score() == 0
    assert g.is_complete()


def test_open_frames():
    assert game_of(3, 4, 2, 5).score() == 14


def test_strike_bonus():
    assert game_of(10, 3, 4).score() == 17 + 7  # frame1 = 10+3+4, frame2 = 7


def test_spare_bonus():
    assert game_of(6, 4, 5, 2).score() == 15 + 7


def test_tenth_frame_no_bonus_roll():
    g = game_of(*[0] * 18, 3, 4)
    assert g.is_complete()
    assert g.score() == 7


def test_tenth_frame_spare_gets_one_more():
    g = game_of(*[0] * 18, 6, 4)
    assert not g.is_complete()
    g.roll(7)
    assert g.is_complete()
    assert g.score() == 17


def test_frame_ball_progression():
    g = BowlingGame()
    assert g.frame_ball() == (0, 0)
    g.roll(10)                       # strike skips ball 2
    assert g.frame_ball() == (1, 0)
    g.roll(4)
    assert g.frame_ball() == (1, 1)


def test_needs_fresh_pins():
    g = game_of(3)
    assert not g.needs_fresh_pins()
    g.roll(5)
    assert g.needs_fresh_pins()
    g = game_of(*[10] * 10)          # tenth frame after a strike: fresh rack
    assert g.needs_fresh_pins()


def test_splits():
    def rack(*up: int) -> list[bool]:
        return [i + 1 in up for i in range(10)]

    assert is_split(rack(7, 10))          # the famous one
    assert is_split(rack(4, 6))
    assert not is_split(rack(1, 2, 3))    # headpin standing: never a split
    assert not is_split(rack(2, 4, 5))    # connected group
    assert not is_split(rack(10))         # single pin


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"ok  {fn.__name__}")
    print(f"{len(fns)} checks passed")
