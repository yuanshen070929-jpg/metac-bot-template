"""Which open questions to forecast on this run."""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from forecasting_tools import MetaculusQuestion

from ensemble_bot.dates import _as_utc, minutes_until_close


def _close_key(question: MetaculusQuestion) -> tuple[int, float]:
    if question.close_time is None:
        return (1, 0.0)
    return (0, _as_utc(question.close_time).timestamp())


def select_questions(
    questions: Sequence[MetaculusQuestion],
    *,
    skip_forecasted: bool,
    min_minutes_before_close: int,
    max_questions: int,
    now: datetime | None = None,
) -> tuple[list[MetaculusQuestion], list[str]]:
    """Drop duplicates, already-forecast and about-to-close questions; soonest-closing first; cap."""
    notes: list[str] = []
    seen: set[object] = set()
    candidates: list[MetaculusQuestion] = []
    for question in questions:
        key = question.id_of_question or question.page_url or question.question_text
        if key in seen:
            continue
        seen.add(key)
        if skip_forecasted and question.already_forecasted:
            continue
        minutes_left = minutes_until_close(question, now)
        if minutes_left is not None and minutes_left < min_minutes_before_close:
            notes.append(f"Skipped (closes in {minutes_left:.0f} min): {question.page_url}")
            continue
        candidates.append(question)

    candidates.sort(key=_close_key)
    if max_questions > 0 and len(candidates) > max_questions:
        notes.append(
            f"{len(candidates) - max_questions} more question(s) left for the next run "
            f"(MAX_QUESTIONS_PER_RUN={max_questions})"
        )
        candidates = candidates[:max_questions]
    return candidates, notes
