"""Run the ensemble bot.

    uv run python run_bot.py --mode test_questions   # smoke test on the bot-testing-area tournament
    uv run python run_bot.py --mode tournament       # FutureEval seasonal tournament + MiniBench
    uv run python run_bot.py --mode minibench --dry-run

Settings come from environment variables (see .env.template and README.md).
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

import dotenv

dotenv.load_dotenv()

from bot_helpers import (  # noqa: E402
    check_environment,
    print_run_summary_banner,
    print_startup_banner,
    silence_noisy_dependencies,
)

silence_noisy_dependencies()

from forecasting_tools import ForecastReport, MetaculusClient, MetaculusQuestion  # noqa: E402

from ensemble_bot.bot import EnsembleBot  # noqa: E402
from ensemble_bot.config import BotSettings  # noqa: E402
from ensemble_bot.selection import select_questions  # noqa: E402

logger = logging.getLogger("run_bot")

MODES = ("tournament", "main", "minibench", "metaculus_cup", "test_questions")
TOURNAMENT_URLS = {
    "tournament": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
    "main": "https://www.metaculus.com/tournament/fall-futureeval-2026/",
    "minibench": "https://www.metaculus.com/aib/minibench/",
    "metaculus_cup": "https://www.metaculus.com/tournament/metaculus-cup-fall-2026/",
    "test_questions": "https://www.metaculus.com/tournament/bot-testing-area/",
}


def _as_tournament_id(value: str) -> int | str:
    return int(value) if value.isdigit() else value


def tournament_targets(mode: str, settings: BotSettings, client: MetaculusClient) -> list[int | str]:
    if mode == "tournament" and settings.tournament_ids:
        return [_as_tournament_id(t) for t in settings.tournament_ids]
    return {
        "tournament": [client.CURRENT_AI_COMPETITION_ID, client.CURRENT_MINIBENCH_ID],
        "main": [client.CURRENT_AI_COMPETITION_ID],
        "minibench": [client.CURRENT_MINIBENCH_ID],
        "metaculus_cup": [client.CURRENT_METACULUS_CUP_ID],
        "test_questions": ["bot-testing-area"],
    }[mode]


def fetch_open_questions(client: MetaculusClient, targets: list[int | str]) -> list[MetaculusQuestion]:
    questions: list[MetaculusQuestion] = []
    for target in targets:
        try:
            found = client.get_all_open_questions_from_tournament(target)
        except Exception as error:  # e.g. a MiniBench round that has not opened yet
            logger.warning(f"Could not load questions from tournament {target!r}: {type(error).__name__}: {error}")
            continue
        logger.info(f"Tournament {target!r}: {len(found)} open question(s)")
        questions.extend(found)
    return questions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the ensemble forecasting bot")
    parser.add_argument("--mode", choices=MODES, default="tournament")
    parser.add_argument("--dry-run", action="store_true", help="forecast but do not publish to Metaculus")
    parser.add_argument("--max-questions", type=int, default=None, help="override MAX_QUESTIONS_PER_RUN")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    check_environment(strict=True)

    settings = BotSettings.from_env()
    publish = not args.dry_run
    print_startup_banner(args.mode, will_publish=publish)
    print(f"⚙️   {settings.describe()}\n")

    reforecast = args.mode in {"test_questions", "metaculus_cup"}
    bot = EnsembleBot(
        settings=settings,
        research_reports_per_question=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=not reforecast,
        extra_metadata_in_explanation=True,
    )

    client = MetaculusClient()
    questions = fetch_open_questions(client, tournament_targets(args.mode, settings, client))
    max_questions = args.max_questions if args.max_questions is not None else settings.max_questions_per_run
    selected, notes = select_questions(
        questions,
        skip_forecasted=not reforecast,
        min_minutes_before_close=settings.min_minutes_before_close,
        max_questions=max_questions,
    )
    for note in notes:
        logger.info(note)
    logger.info(f"Forecasting on {len(selected)} question(s) this run")

    reports: list[ForecastReport | BaseException] = []
    if selected:
        reports = asyncio.run(bot.forecast_questions(selected, return_exceptions=True))
        bot.log_report_summary(reports, raise_errors=False)
    print_run_summary_banner(reports, will_publish=publish, tournament_url=TOURNAMENT_URLS.get(args.mode))

    failures = [report for report in reports if isinstance(report, BaseException)]
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
