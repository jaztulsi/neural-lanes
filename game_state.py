"""Standard 10-frame bowling scoring. Pure logic, no pygame."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FrameView:
    """Display-ready view of one frame on the scoreboard."""
    marks: list[str]
    cumulative: int | None  # None while bonuses are still pending


@dataclass
class BowlingGame:
    """Tracks one player's rolls and computes standard bowling scores."""

    rolls: list[int] = field(default_factory=list)

    def roll(self, pins: int) -> None:
        if self.is_complete():
            raise ValueError("game is already complete")
        self.rolls.append(pins)

    # --- structure ---

    def _frames(self) -> list[list[int]]:
        """Partition the flat roll list into frames (last frame gets 1-3 rolls)."""
        frames: list[list[int]] = []
        i = 0
        for f in range(9):
            if i >= len(self.rolls):
                break
            if self.rolls[i] == 10:
                frames.append([self.rolls[i]])
                i += 1
            else:
                frames.append(self.rolls[i:i + 2])
                i += 2
        if i < len(self.rolls) or len(frames) == 9:
            frames.append(self.rolls[i:i + 3])
        return frames

    def is_complete(self) -> bool:
        frames = self._frames()
        if len(frames) < 10:
            return False
        tenth = frames[9]
        if len(tenth) == 3:
            return True
        if len(tenth) == 2:
            return tenth[0] != 10 and sum(tenth[:2]) != 10
        return False

    def frame_ball(self) -> tuple[int, int]:
        """(frame_index 0-9, ball_index 0-2) for the NEXT roll."""
        if self.is_complete():
            raise ValueError("game is complete")
        frames = self._frames()
        if not frames:
            return 0, 0
        last = frames[-1]
        idx = len(frames) - 1
        if idx < 9:
            if len(last) == 2 or last == [10]:
                return idx + 1, 0
            return idx, 1
        return 9, len(last)

    def needs_fresh_pins(self) -> bool:
        """True if the next roll happens on a full rack of 10 pins."""
        frame, ball = self.frame_ball()
        if ball == 0:
            return True
        if frame < 9:
            return False
        tenth = self._frames()[9] if len(self._frames()) == 10 else []
        if ball == 1:
            return tenth[0] == 10
        # ball == 2: fresh after a double, or after a spare
        return (tenth[0] == 10 and tenth[1] == 10) or \
               (tenth[0] < 10 and tenth[0] + tenth[1] == 10)

    # --- scoring ---

    def score(self) -> int:
        total = 0
        i = 0
        for _ in range(10):
            if i >= len(self.rolls):
                break
            if self.rolls[i] == 10:  # strike
                total += 10 + sum(self.rolls[i + 1:i + 3])
                i += 1
            elif i + 1 < len(self.rolls) and self.rolls[i] + self.rolls[i + 1] == 10:
                total += 10 + (self.rolls[i + 2] if i + 2 < len(self.rolls) else 0)
                i += 2
            else:
                total += sum(self.rolls[i:i + 2])
                i += 2
        return total

    def frame_views(self) -> list[FrameView]:
        """Marks and (resolved) cumulative score per frame, for the scoreboard."""
        views: list[FrameView] = []
        frames = self._frames()
        cumulative = 0
        resolved = True
        i = 0  # index into self.rolls of the current frame's first roll
        for f, frame in enumerate(frames):
            marks = _marks(frame, tenth=(f == 9))
            frame_score: int | None = None
            if f < 9:
                if frame == [10]:
                    bonus = self.rolls[i + 1:i + 3]
                    if len(bonus) == 2:
                        frame_score = 10 + sum(bonus)
                elif len(frame) == 2:
                    if sum(frame) == 10:
                        if i + 2 < len(self.rolls):
                            frame_score = 10 + self.rolls[i + 2]
                    else:
                        frame_score = sum(frame)
            else:
                if len(frame) == 3 or (len(frame) == 2 and sum(frame[:2]) < 10 and frame[0] != 10):
                    frame_score = sum(frame)
            if resolved and frame_score is not None:
                cumulative += frame_score
                views.append(FrameView(marks, cumulative))
            else:
                resolved = False
                views.append(FrameView(marks, None))
            i += len(frame)
        while len(views) < 10:
            views.append(FrameView([], None))
        return views


def _marks(frame: list[int], tenth: bool) -> list[str]:
    marks: list[str] = []
    prev: int | None = None
    fresh = True  # is this roll on a full rack?
    for pins in frame:
        if pins == 10 and fresh:
            marks.append("X")
            prev, fresh = None, True
        elif prev is not None and prev + pins == 10:
            marks.append("/")
            prev, fresh = None, True
        else:
            marks.append("-" if pins == 0 else str(pins))
            if prev is None and not tenth:
                prev, fresh = pins, False
            elif prev is None:
                prev, fresh = pins, False
            else:
                prev, fresh = None, True
    return marks
