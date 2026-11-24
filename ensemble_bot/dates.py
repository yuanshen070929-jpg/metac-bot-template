"""Question timing, computed in code so the models never have to guess it."""

from __future__ import annotations

from datetime import datetime, timezone

from forecasting_tools import MetaculusQuestion


def _as_utc(moment: datetime) -> datetime:
    if moment.tzinfo is None:
        return moment.replace(tzinfo=timezone.utc)
    return moment.astimezone(timezone.utc)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def days_between(start: datetime, end: datetime) -> float:
    return (_as_utc(end) - _as_utc(start)).total_seconds() / 86400


def _relative(moment: datetime, now: datetime) -> str:
    days = days_between(now, moment)
    if abs(days) < 1:
        hours = days * 24
        return f"in {hours:.1f} hours" if hours >= 0 else f"{-hours:.1f} hours ago"
    return f"in {days:.1f} days" if days >= 0 else f"{-days:.1f} days ago"


def _fmt(moment: datetime) -> str:
    return _as_utc(moment).strftime("%Y-%m-%d %H:%M UTC")


def describe_timing(question: MetaculusQuestion, now: datetime | None = None) -> str:
    """A short, reliable block about when things happen for this question."""
    now = _as_utc(now or utc_now())
    lines = [f"- Today: {_fmt(now)}"]
    if question.open_time:
        lines.append(f"- Question opened: {_fmt(question.open_time)} ({_relative(question.open_time, now)})")
    if question.close_time:
        lines.append(
            f"- Forecasting closes: {_fmt(question.close_time)} ({_relative(question.close_time, now)})"
        )
    if question.scheduled_resolution_time:
        lines.append(
            "- Scheduled resolution: "
            f"{_fmt(question.scheduled_resolution_time)} ({_relative(question.scheduled_resolution_time, now)})"
        )
        window = days_between(now, question.scheduled_resolution_time)
        if window >= 0:
            lines.append(
                f"- Time left for the outcome to change: about {window:.1f} days "
                f"({window / 7:.1f} weeks, {window / 30.44:.1f} months)"
            )
    return "\n".join(lines)


def today_str(now: datetime | None = None) -> str:
    return _as_utc(now or utc_now()).strftime("%Y-%m-%d")


def minutes_until_close(question: MetaculusQuestion, now: datetime | None = None) -> float | None:
    if not question.close_time:
        return None
    return days_between(now or utc_now(), question.close_time) * 24 * 60
