"""Research: several cheap sources in parallel, merged into one briefing.

Sources (each can fail on its own without stopping the others):
- AskNews news search (free for tournament participants, if you have keys)
- one or more web-search models on OpenRouter (``model:online`` or Perplexity)
- the resolution-source pages linked in the question, read directly
"""

from __future__ import annotations

import asyncio
import logging
from typing import Awaitable, Callable

from forecasting_tools import AskNewsSearcher, GeneralLlm, MetaculusQuestion

from ensemble_bot import prompts, sources
from ensemble_bot.config import BotSettings, short_model_name
from ensemble_bot.dates import describe_timing

logger = logging.getLogger(__name__)

LlmFactory = Callable[[str], GeneralLlm]
NO_CONTENT = "NO RELEVANT CONTENT"


async def _asknews(question: MetaculusQuestion, lock: asyncio.Semaphore | None) -> str:
    # AskNews rate-limits free accounts, so calls go one at a time.
    if lock is None:
        return await AskNewsSearcher().get_formatted_news_async(question.question_text)
    async with lock:
        return await AskNewsSearcher().get_formatted_news_async(question.question_text)


async def _web_search(llm: GeneralLlm, question: MetaculusQuestion, timing: str) -> str:
    return await llm.invoke(prompts.web_research_prompt(question, timing))


async def _read_one_source(llm: GeneralLlm, question: MetaculusQuestion, url: str) -> str | None:
    page = await sources.fetch_text_async(url)
    if not page:
        return None
    summary = (await llm.invoke(prompts.source_reader_prompt(question, url, page))).strip()
    if not summary or NO_CONTENT in summary.upper():
        return None
    return f"**{url}**\n{summary}"


async def _resolution_sources(llm: GeneralLlm, question: MetaculusQuestion, max_sources: int) -> str:
    urls = sources.extract_urls(
        question.resolution_criteria, question.fine_print, question.background_info, limit=max_sources
    )
    if not urls:
        return ""
    results = await asyncio.gather(*[_read_one_source(llm, question, url) for url in urls], return_exceptions=True)
    parts = []
    for url, result in zip(urls, results, strict=True):
        if isinstance(result, BaseException):
            logger.info(f"Could not read resolution source {url}: {type(result).__name__}: {result}")
        elif result:
            parts.append(result)
    return "\n\n".join(parts)


async def _with_timeout(name: str, coro: Awaitable[str], timeout_s: float) -> str:
    try:
        return await asyncio.wait_for(coro, timeout=timeout_s)
    except Exception as error:  # a failed source should never sink the forecast
        logger.warning(f"Research source '{name}' failed: {type(error).__name__}: {error}")
        return ""


async def gather_research(
    question: MetaculusQuestion,
    settings: BotSettings,
    make_llm: LlmFactory,
    asknews_lock: asyncio.Semaphore | None = None,
) -> str:
    timing = describe_timing(question)
    jobs: dict[str, Awaitable[str]] = {}
    if settings.use_asknews:
        jobs["News (AskNews)"] = _asknews(question, asknews_lock)
    for model in settings.researchers:
        jobs[f"Web search ({short_model_name(model)})"] = _web_search(make_llm(model), question, timing)
    if settings.fetch_resolution_sources and settings.max_sources_to_fetch > 0:
        jobs["Resolution sources (read directly)"] = _resolution_sources(
            make_llm(settings.source_reader_model), question, settings.max_sources_to_fetch
        )

    results = await asyncio.gather(
        *[_with_timeout(name, job, settings.research_timeout_s) for name, job in jobs.items()]
    )

    sections = [f"{prompts.TIMING_SECTION_TITLE}\n{timing}"]
    found_any = False
    for name, text in zip(jobs, results, strict=True):
        text = (text or "").strip()
        if text:
            found_any = True
            sections.append(f"## {name}\n{text}")
    if not found_any:
        logger.warning(f"No research source returned anything for {question.page_url}")
        sections.append("## Note\nNo research source returned results for this question.")
    return "\n\n".join(sections)
