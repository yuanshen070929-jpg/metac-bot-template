"""Settings, read from environment variables.

Everything here can be changed without touching code: locally through `.env`,
on GitHub through repository *variables* (Settings -> Secrets and variables ->
Actions -> Variables). See README.md for the full table.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Values shipped in .env.template that mean "not set".
_PLACEHOLDERS = {"", "REPLACE_ME", "1234567890", "your-token-here", "your-api-key-here"}

# Model slugs are OpenRouter ids with litellm's "openrouter/" prefix.
# "primary|fallback" means: if the primary call fails, retry once with the
# fallback model so the ensemble keeps three members.
#
# Checked against the live OpenRouter model list on 2026-11-24. New models come
# out every few weeks: re-check at the start of each season (README, "赛季切换").
PROFILES: dict[str, dict] = {
    "standard": {
        "reasoning_effort": "high",
        "forecasters": [
            "openrouter/openai/gpt-6.2-sol",
            "openrouter/anthropic/claude-opus-5.5",
            "openrouter/google/gemini-4-argon|openrouter/google/gemini-3.8-flash",
        ],
        "researchers": [
            "openrouter/openai/gpt-6.1-terra:online",
            "openrouter/perplexity/sonar-pro",
        ],
    },
    "budget": {
        "reasoning_effort": "medium",
        "forecasters": [
            "openrouter/openai/gpt-6.1-terra",
            "openrouter/anthropic/claude-sonnet-5.5",
            "openrouter/google/gemini-3.8-flash",
        ],
        "researchers": [
            "openrouter/openai/gpt-6.1-terra:online",
        ],
    },
}

DEFAULT_PROFILE = "standard"
DEFAULT_UTILITY_MODEL = "openrouter/anthropic/claude-haiku-5.5"


def env_str(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    if value in _PLACEHOLDERS:
        return default
    return value


def env_int(name: str, default: int) -> int:
    value = env_str(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        raise ValueError(f"{name} must be an integer, got {value!r}") from None


def env_bool(name: str, default: bool) -> bool:
    value = env_str(name)
    if value is None:
        return default
    lowered = value.lower()
    if lowered in {"1", "true", "yes", "on"}:
        return True
    if lowered in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true/false, got {value!r}")


def env_list(name: str) -> list[str] | None:
    """Comma-separated list. Returns None when unset, [] for 'none'."""
    value = env_str(name)
    if value is None:
        return None
    if value.lower() in {"none", "off", "no"}:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass(frozen=True)
class ModelSpec:
    primary: str
    fallback: str | None = None

    @classmethod
    def parse(cls, text: str) -> "ModelSpec":
        parts = [part.strip() for part in text.split("|") if part.strip()]
        if not parts or len(parts) > 2:
            raise ValueError(f"Bad model spec {text!r}: use 'model' or 'model|fallback'")
        return cls(primary=parts[0], fallback=parts[1] if len(parts) == 2 else None)


def short_model_name(model: str) -> str:
    """'openrouter/openai/gpt-6.2-sol' -> 'gpt-6.2-sol'."""
    return model.rstrip("/").rsplit("/", 1)[-1]


@dataclass(frozen=True)
class BotSettings:
    profile: str = DEFAULT_PROFILE
    forecasters: list[ModelSpec] = field(default_factory=list)
    researchers: list[str] = field(default_factory=list)
    samples_per_model: int = 1
    reasoning_effort: str | None = "high"
    parser_model: str = DEFAULT_UTILITY_MODEL
    summarizer_model: str = DEFAULT_UTILITY_MODEL
    source_reader_model: str = DEFAULT_UTILITY_MODEL
    use_asknews: bool = False
    fetch_resolution_sources: bool = True
    max_sources_to_fetch: int = 3
    max_questions_per_run: int = 15
    min_minutes_before_close: int = 6
    forecaster_timeout_s: int = 420
    research_timeout_s: int = 240
    tournament_ids: list[str] | None = None

    @classmethod
    def from_env(cls) -> "BotSettings":
        profile = (env_str("BOT_PROFILE", DEFAULT_PROFILE) or DEFAULT_PROFILE).lower()
        if profile not in PROFILES:
            raise ValueError(f"BOT_PROFILE must be one of {sorted(PROFILES)}, got {profile!r}")
        preset = PROFILES[profile]

        forecaster_texts = env_list("FORECASTER_MODELS") or preset["forecasters"]
        researchers = env_list("RESEARCH_MODELS")
        if researchers is None:
            researchers = list(preset["researchers"])

        effort = env_str("REASONING_EFFORT", preset["reasoning_effort"])
        if effort and effort.lower() in {"none", "off", "default"}:
            effort = None

        has_asknews = bool(
            (env_str("ASKNEWS_CLIENT_ID") and env_str("ASKNEWS_SECRET")) or env_str("ASKNEWS_API_KEY")
        )
        samples = env_int("SAMPLES_PER_MODEL", 1)
        if samples < 1:
            raise ValueError("SAMPLES_PER_MODEL must be at least 1")

        return cls(
            profile=profile,
            forecasters=[ModelSpec.parse(text) for text in forecaster_texts],
            researchers=researchers,
            samples_per_model=samples,
            reasoning_effort=effort,
            parser_model=env_str("PARSER_MODEL", DEFAULT_UTILITY_MODEL) or DEFAULT_UTILITY_MODEL,
            summarizer_model=env_str("SUMMARIZER_MODEL", DEFAULT_UTILITY_MODEL) or DEFAULT_UTILITY_MODEL,
            source_reader_model=env_str("SOURCE_READER_MODEL", DEFAULT_UTILITY_MODEL)
            or DEFAULT_UTILITY_MODEL,
            use_asknews=env_bool("USE_ASKNEWS", has_asknews),
            fetch_resolution_sources=env_bool("FETCH_RESOLUTION_SOURCES", True),
            max_sources_to_fetch=env_int("MAX_SOURCES_TO_FETCH", 3),
            max_questions_per_run=env_int("MAX_QUESTIONS_PER_RUN", 15),
            min_minutes_before_close=env_int("MIN_MINUTES_BEFORE_CLOSE", 6),
            forecaster_timeout_s=env_int("FORECASTER_TIMEOUT_S", 420),
            research_timeout_s=env_int("RESEARCH_TIMEOUT_S", 240),
            tournament_ids=env_list("TOURNAMENT_IDS") or None,
        )

    def describe(self) -> str:
        members = ", ".join(
            short_model_name(m.primary) + (f" (fallback {short_model_name(m.fallback)})" if m.fallback else "")
            for m in self.forecasters
        )
        research = ", ".join(short_model_name(r) for r in self.researchers) or "none"
        return (
            f"profile={self.profile} | forecasters: {members} | x{self.samples_per_model} each | "
            f"reasoning={self.reasoning_effort or 'provider default'} | web research: {research} | "
            f"asknews={'on' if self.use_asknews else 'off'} | "
            f"resolution sources={'on' if self.fetch_resolution_sources else 'off'}"
        )
