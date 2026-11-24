"""The ensemble bot.

Flow for each question:
1. Research once (AskNews + web-search models + resolution-source pages).
2. Each ensemble member (a different vendor's frontier model) forecasts
   independently from the same research.
3. forecasting-tools aggregates the members: median for binary questions,
   mean for multiple choice, pointwise-median CDF for numeric questions.
4. The comment shows the research and every member's reasoning, labelled
   with the model that wrote it.

Parsing of final answers, numeric bounds, conditional questions and
publishing are inherited unchanged from the official Fall 2026 template.
"""

from __future__ import annotations

import asyncio
import logging
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from forecasting_tools import (
    BinaryQuestion,
    ForecastReport,
    GeneralLlm,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    PredictedOptionList,
    ReasonedPrediction,
    clean_indents,
)
from forecasting_tools.data_models.forecast_report import ResearchWithPredictions
from forecasting_tools.forecast_bots.official_bots.template_bot_2026_fall import FallTemplateBot2026

from ensemble_bot import prompts
from ensemble_bot.config import BotSettings, ModelSpec, short_model_name
from ensemble_bot.dates import describe_timing
from ensemble_bot.research import gather_research

logger = logging.getLogger(__name__)

# The ensemble member currently forecasting. Each member runs in its own
# asyncio task, so each task sees its own value.
_CURRENT_FORECASTER: ContextVar[GeneralLlm | None] = ContextVar("current_forecaster", default=None)
MODEL_LABEL_PREFIX = "*Model*: "


def _is_search_model(model: str) -> bool:
    return model.endswith(":online") or "/perplexity/" in model


def make_forecaster_llm(model: str, settings: BotSettings) -> GeneralLlm:
    # Reasoning tokens count toward max_tokens (and toward the bill): 20k leaves
    # ample room for thinking plus the answer while capping runaway spend.
    kwargs: dict[str, Any] = {"max_tokens": 20000}
    if settings.reasoning_effort and not _is_search_model(model):
        kwargs["reasoning"] = {"effort": settings.reasoning_effort}
    return GeneralLlm(
        model=model,
        temperature=None,  # reasoning models ignore or reject sampling settings
        timeout=settings.forecaster_timeout_s,
        allowed_tries=2,
        **kwargs,
    )


def make_research_llm(model: str, settings: BotSettings) -> GeneralLlm:
    return GeneralLlm(
        model=model,
        temperature=None,
        timeout=settings.research_timeout_s,
        allowed_tries=2,
        max_tokens=6000,
    )


def make_utility_llm(model: str) -> GeneralLlm:
    # Same settings the official template uses for its parser/summarizer.
    return GeneralLlm(model=model, temperature=0.3, timeout=120, allowed_tries=3)


def label_from_reasoning(reasoning: str) -> str | None:
    first_line = reasoning.lstrip().split("\n", 1)[0]
    if first_line.startswith(MODEL_LABEL_PREFIX):
        return first_line[len(MODEL_LABEL_PREFIX):].strip()
    return None


@dataclass
class EnsembleMember:
    label: str
    primary: GeneralLlm
    fallback: GeneralLlm | None = None

    @classmethod
    def from_spec(cls, spec: ModelSpec, settings: BotSettings) -> "EnsembleMember":
        return cls(
            label=short_model_name(spec.primary),
            primary=make_forecaster_llm(spec.primary, settings),
            fallback=make_forecaster_llm(spec.fallback, settings) if spec.fallback else None,
        )


class EnsembleBot(FallTemplateBot2026):
    """Research once, let several different models forecast, publish the aggregate."""

    def __init__(self, *, settings: BotSettings | None = None, **kwargs: Any) -> None:
        self.settings = settings or BotSettings.from_env()
        if not self.settings.forecasters:
            raise ValueError("Configure at least one forecaster model (FORECASTER_MODELS)")
        self.members = [EnsembleMember.from_spec(spec, self.settings) for spec in self.settings.forecasters]

        kwargs["predictions_per_research_report"] = len(self.members) * self.settings.samples_per_model
        kwargs.setdefault("research_reports_per_question", 1)
        # Publish even if only one member succeeds: one good model beats no forecast.
        kwargs.setdefault("required_successful_predictions", 0.3)
        kwargs.setdefault(
            "llms",
            {
                "default": self.members[0].primary,
                "summarizer": make_utility_llm(self.settings.summarizer_model),
                "researcher": "ensemble research (see research section)",
                "parser": make_utility_llm(self.settings.parser_model),
            },
        )
        super().__init__(**kwargs)
        self._asknews_lock: asyncio.Semaphore | None = None
        self._asknews_lock_loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------------ models

    def get_llm(self, purpose: str = "default", guarantee_type: Any = None) -> Any:  # type: ignore[override]
        if purpose == "default":
            current = _CURRENT_FORECASTER.get()
            if current is not None:
                return current.model if guarantee_type == "string_name" else current
        return super().get_llm(purpose, guarantee_type)

    def make_llm_dict(self) -> dict[str, Any]:
        llm_dict = super().make_llm_dict()
        llm_dict["default"] = "ensemble: " + ", ".join(
            member.label + (f" (fallback {short_model_name(member.fallback.model)})" if member.fallback else "")
            for member in self.members
        )
        llm_dict["research_sources"] = self.settings.describe()
        return llm_dict

    # ------------------------------------------------------------------ research

    def _get_asknews_lock(self) -> asyncio.Semaphore:
        loop = asyncio.get_running_loop()
        if self._asknews_lock is None or self._asknews_lock_loop is not loop:
            self._asknews_lock = asyncio.Semaphore(1)
            self._asknews_lock_loop = loop
        return self._asknews_lock

    async def run_research(self, question: MetaculusQuestion) -> str:
        research = await gather_research(
            question,
            self.settings,
            lambda model: make_research_llm(model, self.settings),
            asknews_lock=self._get_asknews_lock(),
        )
        logger.info(f"Research for {question.page_url}:\n{research}")
        return research

    # ------------------------------------------------------------------ ensemble

    async def _research_and_make_predictions(
        self, question: MetaculusQuestion
    ) -> ResearchWithPredictions:
        notepad = await self._get_notepad(question)
        notepad.total_research_reports_attempted += 1
        research = await self.run_research(question)
        summary_report = await self.summarize_research(question, research)
        research_to_use = summary_report if self.use_research_summary_to_forecast else research

        tasks = [
            self._predict_with_member(member, question, research_to_use)
            for member in self.members
            for _ in range(self.settings.samples_per_model)
        ]
        valid_predictions, errors, exception_group = await self._gather_results_and_exceptions(tasks)
        if errors:
            logger.warning(f"Some ensemble members failed on {question.page_url}: {errors}")
        if not valid_predictions:
            assert exception_group is not None
            self._reraise_exception_with_prepended_message(
                exception_group, "Every ensemble member failed"
            )
        return ResearchWithPredictions(
            research_report=research,
            summary_report=summary_report,
            errors=errors,
            predictions=valid_predictions,
        )

    async def _predict_with_member(
        self, member: EnsembleMember, question: MetaculusQuestion, research: str
    ) -> ReasonedPrediction:
        attempts: list[tuple[str, GeneralLlm]] = [(member.label, member.primary)]
        if member.fallback is not None:
            attempts.append(
                (f"{short_model_name(member.fallback.model)} (fallback for {member.label})", member.fallback)
            )
        # Generous outer limit: the model call has its own timeout and retries,
        # this only stops a member from hanging the whole run.
        outer_timeout = self.settings.forecaster_timeout_s * 2 + 180
        last_error: BaseException | None = None
        for label, llm in attempts:
            token = _CURRENT_FORECASTER.set(llm)
            try:
                prediction = await asyncio.wait_for(self._make_prediction(question, research), outer_timeout)
            except Exception as error:
                last_error = error
                logger.warning(f"{label} failed on {question.page_url}: {type(error).__name__}: {error}")
                continue
            finally:
                _CURRENT_FORECASTER.reset(token)
            return ReasonedPrediction(
                prediction_value=prediction.prediction_value,
                reasoning=f"{MODEL_LABEL_PREFIX}{label}\n\n{prediction.reasoning}",
            )
        raise RuntimeError(f"{member.label} produced no forecast: {last_error}") from last_error

    # ------------------------------------------------------------------ prompts

    async def _run_forecast_on_binary(
        self, question: BinaryQuestion, research: str
    ) -> ReasonedPrediction[float]:
        prompt = prompts.binary_prompt(
            question, research, describe_timing(question), self._get_conditional_disclaimer_if_necessary(question)
        )
        return await self._binary_prompt_to_forecast(question, prompt)

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        prompt = prompts.multiple_choice_prompt(
            question, research, describe_timing(question), self._get_conditional_disclaimer_if_necessary(question)
        )
        return await self._multiple_choice_prompt_to_forecast(question, prompt)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_bound_message, lower_bound_message = self._create_upper_and_lower_bound_messages(question)
        prompt = prompts.numeric_prompt(
            question,
            research,
            describe_timing(question),
            lower_bound_message,
            upper_bound_message,
            self._get_conditional_disclaimer_if_necessary(question),
        )
        return await self._numeric_prompt_to_forecast(question, prompt)

    # ------------------------------------------------------------------ comment

    @classmethod
    def _format_and_expand_research_summary(
        cls,
        report_number: int,
        report_type: type[ForecastReport],
        predicted_research: ResearchWithPredictions,
    ) -> str:
        forecaster_prediction_bullet_points = ""
        for j, forecast in enumerate(predicted_research.predictions):
            label = label_from_reasoning(forecast.reasoning) or f"Forecaster {j + 1}"
            readable_prediction = report_type.make_readable_prediction(forecast.prediction_value)
            forecaster_prediction_bullet_points += f"*{label}*: {readable_prediction}\n"

        return clean_indents(
            f"""
            ## Report {report_number} Summary
            ### Forecasts
            {forecaster_prediction_bullet_points}

            ### Research Summary
            {predicted_research.summary_report}
            """
        )
