"""Unit tests for the small helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from forecasting_tools import BinaryQuestion

from ensemble_bot import prompts, sources
from ensemble_bot.config import BotSettings, ModelSpec, short_model_name
from ensemble_bot.dates import describe_timing, minutes_until_close
from ensemble_bot.selection import select_questions
from ensemble_bot.sources import extract_urls, html_to_text

NOW = datetime(2026, 11, 24, 20, 0, tzinfo=timezone.utc)


def question(qid: int, close_in_min: float | None, forecasted: bool = False, **extra) -> BinaryQuestion:
    return BinaryQuestion(
        question_text=f"Question {qid}?",
        id_of_question=qid,
        page_url=f"https://www.metaculus.com/questions/{qid}/",
        close_time=None if close_in_min is None else NOW + timedelta(minutes=close_in_min),
        already_forecasted=forecasted,
        **extra,
    )


# ---------------------------------------------------------------- sources

def test_extract_urls_cleans_and_filters() -> None:
    text = (
        "Per [the tracker](https://example.org/data?id=3). Also https://en.wikipedia.org/wiki/Foo_(bar), "
        "see https://www.metaculus.com/questions/1/ and https://example.org/data?id=3 again; "
        "chart at https://cdn.example.org/chart.png."
    )
    assert extract_urls(text) == ["https://example.org/data?id=3", "https://en.wikipedia.org/wiki/Foo_(bar)"]


def test_extract_urls_limit_and_order() -> None:
    urls = extract_urls("https://a.org/1", None, "https://b.org/2 https://c.org/3", limit=2)
    assert urls == ["https://a.org/1", "https://b.org/2"]


def test_html_to_text() -> None:
    html = (
        "<html><head><title>Stats page</title><style>p{color:red}</style></head><body>"
        "<nav>Home | About</nav><h1>Latest</h1><p>Value: <b>42</b></p>"
        "<table><tr><td>2026-11</td><td>42</td></tr></table><script>alert(1)</script></body></html>"
    )
    text = html_to_text(html)
    assert text.startswith("Stats page")
    assert "Value: 42" in text
    assert "2026-11 | 42" in text
    assert "alert" not in text and "color:red" not in text and "About" not in text


class _FakeResponse:
    def __init__(self, body: bytes, content_type: str, status: int = 200) -> None:
        self._body = body
        self.headers = {"Content-Type": content_type}
        self.status_code = status

    def iter_content(self, chunk_size: int):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i : i + chunk_size]

    def close(self) -> None:
        pass


def test_fetch_text_html_utf8_without_charset(monkeypatch) -> None:
    body = "<html><body><p>Température moyenne: 42 °C — latest reading from the agency</p></body></html>".encode()
    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: _FakeResponse(body, "text/html"))
    assert sources.fetch_text("https://example.org") == "Température moyenne: 42 °C — latest reading from the agency"


def test_fetch_text_ignores_near_empty_pages(monkeypatch) -> None:
    body = b"<html><body><div id='app'></div><script>render()</script></body></html>"
    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: _FakeResponse(body, "text/html"))
    assert sources.fetch_text("https://example.org/js-app") is None


def test_fetch_text_json_and_rejections(monkeypatch) -> None:
    payload = b'{"series": [{"date": "2026-11-20", "value": 41.5}], "source": "agency"}'
    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: _FakeResponse(payload, "application/json"))
    assert '"value": 41.5' in sources.fetch_text("https://example.org/api")

    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: _FakeResponse(b"%PDF-1.7", "application/pdf"))
    assert sources.fetch_text("https://example.org/report.pdf") is None

    monkeypatch.setattr(sources.requests, "get", lambda *a, **k: _FakeResponse(b"nope", "text/html", status=404))
    assert sources.fetch_text("https://example.org/missing") is None


# ---------------------------------------------------------------- dates

def test_describe_timing() -> None:
    q = BinaryQuestion(
        question_text="x",
        open_time=NOW - timedelta(hours=1),
        close_time=NOW + timedelta(minutes=90),
        scheduled_resolution_time=NOW + timedelta(days=14),
    )
    text = describe_timing(q, now=NOW)
    assert "Today: 2026-11-24 20:00 UTC" in text
    assert "Forecasting closes: 2026-11-24 21:30 UTC (in 1.5 hours)" in text
    assert "about 14.0 days (2.0 weeks" in text
    assert minutes_until_close(q, now=NOW) == pytest.approx(90)


def test_naive_datetimes_are_treated_as_utc() -> None:
    q = BinaryQuestion(question_text="x", close_time=datetime(2026, 11, 24, 21, 0))
    assert minutes_until_close(q, now=NOW) == pytest.approx(60)


# ---------------------------------------------------------------- selection

def test_selection_filters_sorts_and_caps() -> None:
    questions = [
        question(1, 300),
        question(2, 30),
        question(2, 30),  # duplicate
        question(3, 3),  # closes too soon
        question(4, 60, forecasted=True),
        question(5, None),
        question(6, 120),
    ]
    selected, notes = select_questions(
        questions, skip_forecasted=True, min_minutes_before_close=6, max_questions=3, now=NOW
    )
    assert [q.id_of_question for q in selected] == [2, 6, 1]
    assert any("closes in 3 min" in n for n in notes)
    assert any("1 more question(s)" in n for n in notes)


def test_selection_keeps_forecasted_when_reforecasting() -> None:
    selected, _ = select_questions(
        [question(4, 60, forecasted=True)], skip_forecasted=False, min_minutes_before_close=6, max_questions=0, now=NOW
    )
    assert len(selected) == 1


# ---------------------------------------------------------------- config

def test_default_profile() -> None:
    settings = BotSettings.from_env()
    assert settings.profile == "standard"
    assert [m.primary for m in settings.forecasters] == [
        "openrouter/openai/gpt-6.2-sol",
        "openrouter/anthropic/claude-opus-5.5",
        "openrouter/google/gemini-4-argon",
    ]
    assert settings.forecasters[2].fallback == "openrouter/google/gemini-3.8-flash"
    assert settings.use_asknews is False


def test_env_overrides(monkeypatch) -> None:
    monkeypatch.setenv("BOT_PROFILE", "budget")
    monkeypatch.setenv("RESEARCH_MODELS", "none")
    monkeypatch.setenv("FORECASTER_MODELS", "openrouter/a/x|openrouter/a/y, openrouter/b/z")
    monkeypatch.setenv("ASKNEWS_CLIENT_ID", "id")
    monkeypatch.setenv("ASKNEWS_SECRET", "secret")
    monkeypatch.setenv("REASONING_EFFORT", "none")
    settings = BotSettings.from_env()
    assert settings.profile == "budget"
    assert settings.researchers == []
    assert settings.reasoning_effort is None  # explicit "none" beats the budget default (medium)
    assert settings.forecasters == [ModelSpec("openrouter/a/x", "openrouter/a/y"), ModelSpec("openrouter/b/z")]
    assert settings.use_asknews is True


def test_budget_profile_defaults(monkeypatch) -> None:
    monkeypatch.setenv("BOT_PROFILE", "budget")
    settings = BotSettings.from_env()
    assert settings.reasoning_effort == "medium"
    assert [short_model_name(m.primary) for m in settings.forecasters] == [
        "gpt-6.1-terra", "claude-sonnet-5.5", "gemini-3.8-flash",
    ]


def test_placeholder_values_are_ignored(monkeypatch) -> None:
    monkeypatch.setenv("BOT_PROFILE", "REPLACE_ME")
    assert BotSettings.from_env().profile == "standard"


def test_bad_profile(monkeypatch) -> None:
    monkeypatch.setenv("BOT_PROFILE", "turbo")
    with pytest.raises(ValueError):
        BotSettings.from_env()


def test_short_model_name() -> None:
    assert short_model_name("openrouter/openai/gpt-6.1-terra:online") == "gpt-6.1-terra:online"


# ---------------------------------------------------------------- prompts keep the parser's formats

def test_timing_block_not_repeated_in_prompt() -> None:
    q = BinaryQuestion(question_text="Will it happen?")
    research = f"{prompts.TIMING_SECTION_TITLE}\n- Today: X\n\n## Web search (m)\nfacts"
    prompt = prompts.binary_prompt(q, research, "- Today: X")
    assert prompt.count("- Today: X") == 1
    assert "## Web search (m)\nfacts" in prompt


def test_prompts_keep_final_answer_formats() -> None:
    q = BinaryQuestion(question_text="Will it happen?", resolution_criteria="If it happens.")
    assert 'Probability: ZZ%' in prompts.binary_prompt(q, "research", "timing")
    research = prompts.web_research_prompt(q, "timing")
    assert "Do not make a forecast" in research and "Will it happen?" in research
