"""End-to-end runs of the bot with fake models: research -> ensemble -> aggregate -> comment."""

from __future__ import annotations

import asyncio

import pytest
from forecasting_tools import ForecastReport, NumericDistribution, PredictedOptionList

from ensemble_bot.bot import EnsembleBot, label_from_reasoning
from ensemble_bot.config import BotSettings


def make_bot(**env_overrides: str) -> EnsembleBot:
    settings = BotSettings.from_env()
    return EnsembleBot(
        settings=settings,
        publish_reports_to_metaculus=False,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
    )


def run(bot: EnsembleBot, questions: list) -> list:
    return asyncio.run(bot.forecast_questions(questions, return_exceptions=True))


def only_report(results: list) -> ForecastReport:
    assert len(results) == 1
    report = results[0]
    assert isinstance(report, ForecastReport), f"forecast failed: {report!r}"
    return report


def test_binary_median_of_three_vendors(world, binary_question) -> None:
    report = only_report(run(make_bot(), [binary_question]))
    # members answer 20%, 30%, 60% -> median 30%
    assert report.prediction == pytest.approx(0.30)
    forecast_calls = [model for model, kind in world.calls if kind == "binary"]
    assert sorted(forecast_calls) == ["claude-opus-5.5", "gemini-4-argon", "gpt-6.2-sol"]
    # each member is named in the published comment
    for label in ("gpt-6.2-sol", "claude-opus-5.5", "gemini-4-argon"):
        assert f"*{label}*" in report.explanation
    assert "Question timing (computed by code)" in report.explanation


def test_research_combines_sources(world, binary_question) -> None:
    run(make_bot(), [binary_question])
    kinds = [(model, kind) for model, kind in world.calls if kind in {"research", "source"}]
    assert ("gpt-6.1-terra:online", "research") in kinds
    assert ("sonar-pro", "research") in kinds
    assert ("claude-haiku-5.5", "source") in kinds  # resolution source page was read
    forecast_prompt = next(p for m, k, p in world.prompts if k == "binary")
    assert "Briefing from sonar-pro" in forecast_prompt
    assert "Resolution sources (read directly)" in forecast_prompt
    assert "https://example.org/stats" in forecast_prompt
    assert "Time left for the outcome to change" in forecast_prompt


def test_fallback_model_replaces_failed_member(world, binary_question) -> None:
    world.fail_forecast.add("gemini-4-argon")
    report = only_report(run(make_bot(), [binary_question]))
    # 20%, 30%, 45% (fallback) -> median 30%
    assert report.prediction == pytest.approx(0.30)
    assert "gemini-3.8-flash (fallback for gemini-4-argon)" in report.explanation


def test_one_surviving_member_still_publishes(world, binary_question, monkeypatch) -> None:
    monkeypatch.setenv("FORECASTER_MODELS", "openrouter/openai/gpt-6.2-sol,openrouter/anthropic/claude-opus-5.5")
    world.fail_forecast.add("gpt-6.2-sol")
    report = only_report(run(make_bot(), [binary_question]))
    assert report.prediction == pytest.approx(0.30)


def test_all_members_failing_is_an_error(world, binary_question, monkeypatch) -> None:
    monkeypatch.setenv("FORECASTER_MODELS", "openrouter/openai/gpt-6.2-sol")
    world.fail_forecast.add("gpt-6.2-sol")
    results = run(make_bot(), [binary_question])
    assert isinstance(results[0], BaseException)


def test_research_outage_does_not_stop_forecast(world, binary_question) -> None:
    world.fail_research.update({"gpt-6.1-terra:online", "sonar-pro"})
    report = only_report(run(make_bot(), [binary_question]))
    assert report.prediction == pytest.approx(0.30)


def test_multiple_choice_mean(world, mc_question) -> None:
    report = only_report(run(make_bot(), [mc_question]))
    prediction = report.prediction
    assert isinstance(prediction, PredictedOptionList)
    probabilities = {o.option_name: o.probability for o in prediction.predicted_options}
    assert probabilities["Alpha"] == pytest.approx(0.50)  # mean of 60, 50, 40
    assert probabilities["Beta"] == pytest.approx(1 / 3, abs=1e-6)  # mean of 30, 30, 40
    assert sum(probabilities.values()) == pytest.approx(1.0)


def test_numeric_distribution(world, numeric_question) -> None:
    report = only_report(run(make_bot(), [numeric_question]))
    distribution = report.prediction
    assert isinstance(distribution, NumericDistribution)
    cdf = [p.percentile for p in distribution.get_cdf()]
    assert all(b >= a for a, b in zip(cdf, cdf[1:]))
    assert cdf[0] == pytest.approx(0.0, abs=1e-9)  # closed lower bound
    median_value = next(p.value for p in distribution.get_cdf() if p.percentile >= 0.5)
    assert 38 <= median_value <= 48


def test_discrete_question(world, discrete_question) -> None:
    report = only_report(run(make_bot(), [discrete_question]))
    assert isinstance(report.prediction, NumericDistribution)
    numeric_prompt = next(p for m, k, p in world.prompts if k == "numeric")
    assert "whole number" in numeric_prompt


def test_two_samples_per_model(world, binary_question, monkeypatch) -> None:
    monkeypatch.setenv("SAMPLES_PER_MODEL", "2")
    bot = make_bot()
    assert bot.predictions_per_research_report == 6
    only_report(run(bot, [binary_question]))
    assert len([1 for _, kind in world.calls if kind == "binary"]) == 6


def test_label_parsing() -> None:
    assert label_from_reasoning("*Model*: gpt-6.2-sol\n\nreasoning") == "gpt-6.2-sol"
    assert label_from_reasoning("no label here") is None
