"""Offline test fixtures: fake LLMs, fake parser, fake web pages. No network, no keys."""

from __future__ import annotations

import ast
import re
from datetime import datetime, timedelta, timezone
from typing import Any, get_args, get_origin

import pytest
from forecasting_tools import (
    BinaryQuestion,
    DiscreteQuestion,
    GeneralLlm,
    MultipleChoiceQuestion,
    NumericQuestion,
)
from forecasting_tools.data_models.binary_report import BinaryPrediction
from forecasting_tools.data_models.multiple_choice_report import PredictedOption, PredictedOptionList
from forecasting_tools.data_models.numeric_report import Percentile

import forecasting_tools.forecast_bots.official_bots.template_bot_2026_fall as template_module
from ensemble_bot import sources

CONFIG_VARS = [
    "BOT_PROFILE", "FORECASTER_MODELS", "RESEARCH_MODELS", "SAMPLES_PER_MODEL", "REASONING_EFFORT",
    "PARSER_MODEL", "SUMMARIZER_MODEL", "SOURCE_READER_MODEL", "USE_ASKNEWS", "FETCH_RESOLUTION_SOURCES",
    "MAX_SOURCES_TO_FETCH", "MAX_QUESTIONS_PER_RUN", "MIN_MINUTES_BEFORE_CLOSE", "FORECASTER_TIMEOUT_S",
    "RESEARCH_TIMEOUT_S", "TOURNAMENT_IDS", "ASKNEWS_CLIENT_ID", "ASKNEWS_SECRET", "ASKNEWS_API_KEY",
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in CONFIG_VARS:
        monkeypatch.delenv(name, raising=False)


def classify(prompt: str) -> str:
    if "You are the research assistant of a professional forecaster" in prompt:
        return "research"
    if "names a web page as (part of) its resolution source" in prompt:
        return "source"
    if "Please summarize the following research" in prompt:
        return "summary"
    if 'Probability: ZZ%' in prompt:
        return "binary"
    if "Option_A: Probability_A" in prompt:
        return "mc"
    if "Percentile 10: XX" in prompt:
        return "numeric"
    return "other"


class FakeWorld:
    """Pretends to be every LLM. Each model gives a fixed, distinct answer."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.prompts: list[tuple[str, str, str]] = []
        self.fail_forecast: set[str] = set()
        self.fail_research: set[str] = set()
        self.binary = {"gpt-6.2-sol": 20, "claude-opus-5.5": 30, "gemini-4-argon": 60, "gemini-3.8-flash": 45}
        self.mc = {
            "gpt-6.2-sol": [60, 30, 10],
            "claude-opus-5.5": [50, 30, 20],
            "gemini-4-argon": [40, 40, 20],
            "gemini-3.8-flash": [70, 20, 10],
        }
        self.numeric = {
            "gpt-6.2-sol": [30, 35, 40, 45, 50, 60],
            "claude-opus-5.5": [32, 36, 41, 46, 52, 58],
            "gemini-4-argon": [28, 34, 42, 47, 55, 65],
            "gemini-3.8-flash": [30, 35, 40, 45, 50, 60],
        }
        self.discrete = {
            "gpt-6.2-sol": [2, 3, 4, 5, 6, 7],
            "claude-opus-5.5": [2, 3, 4, 5, 7, 8],
            "gemini-4-argon": [1, 2, 4, 5, 6, 8],
            "gemini-3.8-flash": [2, 3, 4, 5, 6, 7],
        }

    async def invoke(self, llm: GeneralLlm, prompt: str) -> str:
        model = llm.model.rsplit("/", 1)[-1]
        kind = classify(prompt)
        self.calls.append((model, kind))
        self.prompts.append((model, kind, prompt))
        if kind == "research":
            if model in self.fail_research:
                raise RuntimeError(f"{model} search is down")
            return f"Briefing from {model}: the tracked value was 41 on 2026-11-20, per the agency."
        if kind == "source":
            return "The page shows the latest value: 42 (updated 2026-11-23)."
        if kind == "summary":
            return "Short summary of the research."
        if model in self.fail_forecast:
            raise RuntimeError(f"{model} is overloaded")
        if kind == "binary":
            return f"Some reasoning by {model}.\nProbability: {self.binary[model]}%"
        if kind == "mc":
            options = ast.literal_eval(re.search(r"in this order (\[.*?\]) as:", prompt).group(1))
            lines = [f"{option}: {p}%" for option, p in zip(options, self.mc[model])]
            return f"Some reasoning by {model}.\n" + "\n".join(lines)
        if kind == "numeric":
            values = self.discrete[model] if "whole number" in prompt else self.numeric[model]
            pcts = [10, 20, 40, 60, 80, 90]
            return f"Some reasoning by {model}.\n" + "\n".join(f"Percentile {p}: {v}" for p, v in zip(pcts, values))
        return "ok"


async def fake_structure_output(text_to_structure: str, output_type: Any, **_: Any) -> Any:
    text = text_to_structure
    if output_type is BinaryPrediction:
        value = float(re.findall(r"Probability:\s*([\d.]+)%", text)[-1]) / 100
        return BinaryPrediction(prediction_in_decimal=value)
    if output_type is PredictedOptionList:
        pairs = re.findall(r"^([A-Za-z][\w ]*?):\s*([\d.]+)%\s*$", text, re.M)
        return PredictedOptionList(
            predicted_options=[PredictedOption(option_name=n, probability=float(p) / 100) for n, p in pairs]
        )
    if get_origin(output_type) is list and get_args(output_type)[0] is Percentile:
        pairs = re.findall(r"Percentile\s+(\d+):\s*([-\d.]+)", text)
        return [Percentile(percentile=int(p) / 100, value=float(v)) for p, v in pairs]
    raise AssertionError(f"Unexpected output type {output_type}")


PAGES = {
    "https://example.org/stats": "<html><head><title>Official stats</title></head>"
    "<body><nav>Menu</nav><p>Latest value: 42</p><script>var x=1;</script></body></html>",
}


def fake_fetch_text(url: str, **_: Any) -> str | None:
    html = PAGES.get(url)
    return sources.html_to_text(html) if html else None


@pytest.fixture
def world(monkeypatch: pytest.MonkeyPatch) -> FakeWorld:
    fake = FakeWorld()

    async def invoke(self: GeneralLlm, prompt: str, *args: Any, **kwargs: Any) -> str:
        return await fake.invoke(self, prompt)

    monkeypatch.setattr(GeneralLlm, "invoke", invoke)
    monkeypatch.setattr(template_module, "structure_output", fake_structure_output)
    monkeypatch.setattr(sources, "fetch_text", fake_fetch_text)

    def no_network(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("tests must not touch the network")

    monkeypatch.setattr("requests.get", no_network)
    return fake


def _times() -> dict[str, datetime]:
    now = datetime.now(timezone.utc)
    return {
        "open_time": now - timedelta(minutes=10),
        "close_time": now + timedelta(minutes=80),
        "scheduled_resolution_time": now + timedelta(days=30),
    }


COMMON = {
    "background_info": "Background text.",
    "resolution_criteria": "Resolves Yes if the value on https://example.org/stats exceeds 40 on the resolution date.",
    "fine_print": "Uses the official figure.",
    "already_forecasted": False,
}


@pytest.fixture
def binary_question() -> BinaryQuestion:
    return BinaryQuestion(
        question_text="Will the tracked value exceed 40?",
        id_of_question=101, id_of_post=101, page_url="https://www.metaculus.com/questions/101/",
        **COMMON, **_times(),
    )


@pytest.fixture
def mc_question() -> MultipleChoiceQuestion:
    return MultipleChoiceQuestion(
        question_text="Which option will win?",
        options=["Alpha", "Beta", "Gamma"],
        id_of_question=102, id_of_post=102, page_url="https://www.metaculus.com/questions/102/",
        **COMMON, **_times(),
    )


@pytest.fixture
def numeric_question() -> NumericQuestion:
    return NumericQuestion(
        question_text="What will the tracked value be?",
        upper_bound=100.0, lower_bound=0.0, open_upper_bound=True, open_lower_bound=False,
        unit_of_measure="units",
        id_of_question=103, id_of_post=103, page_url="https://www.metaculus.com/questions/103/",
        **COMMON, **_times(),
    )


@pytest.fixture
def discrete_question() -> DiscreteQuestion:
    return DiscreteQuestion(
        question_text="How many launches will there be?",
        upper_bound=10.5, lower_bound=-0.5, open_upper_bound=False, open_lower_bound=False,
        cdf_size=12, unit_of_measure="launches",
        id_of_question=104, id_of_post=104, page_url="https://www.metaculus.com/questions/104/",
        **COMMON, **_times(),
    )
