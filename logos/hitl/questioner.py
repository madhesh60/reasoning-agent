"""logos/hitl/questioner.py — Clarifying question generator."""

from __future__ import annotations

import os
import re
import asyncio
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from logos.config import Config


async def generate_questions(
    query: str,
    memory_context: str = "",
    max_questions: int = 3,
    cfg: 'Config | None' = None,
) -> list[str]:
    """
    Generate targeted clarifying questions for a research query.

    Calls the LLM endpoint directly (fast, ~2-3s).
    Falls back to a set of smart default questions if the call fails.
    """
    if not cfg:
        from logos.config import Config
        cfg = Config()

    try:
        return await _llm_questions(query, memory_context, max_questions, cfg)
    except Exception as e:
        import logging
        logging.error(f"Failed to generate questions: {e}")
        return _default_questions(query)[:max_questions]


async def _llm_questions(query: str, memory_context: str, max_q: int, cfg: 'Config') -> list[str]:
    system = (
        "You are a senior research coordinator for an intelligence agency. "
        f"A researcher has submitted a query. Your job is to generate exactly {max_q} "
        "sharp, strategic clarifying questions that will significantly improve the quality "
        "and relevance of the research report. "
        "Questions must be specific to this query — not generic. "
        "Do NOT explain or introduce the questions. "
        "Output ONLY a numbered list: 1. ... 2. ... 3. ..."
    )

    user_parts = [f"Research query: {query}"]
    if memory_context:
        user_parts.append(f"\nResearcher context:\n{memory_context}")
    user_parts.append(f"\nGenerate {max_q} clarifying questions.")
    user_msg = "\n".join(user_parts)

    loop = asyncio.get_running_loop()

    def _call() -> str:
        client = cfg.build_openai_client()
        resp   = client.chat.completions.create(
            model=cfg.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=400,
            temperature=0.4,
        )
        return resp.choices[0].message.content or ""

    raw = await loop.run_in_executor(None, _call)
    return _parse_questions(raw)[:max_q]


def _parse_questions(raw: str) -> list[str]:
    """Extract numbered questions from raw LLM output."""
    questions = []
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        # Remove leading number+punctuation: "1.", "1)", "Q1:", etc.
        cleaned = re.sub(r"^[\d]+[.):\s]+", "", line).strip()
        if len(cleaned) > 15:
            questions.append(cleaned)
    return questions


def _default_questions(query: str) -> list[str]:
    """Fallback questions when LLM is unavailable."""
    return [
        "What is the primary purpose of this research — market evaluation, "
        "product strategy, academic study, or competitive intelligence?",
        "Which geographic market or regulatory environment is most relevant "
        "to your analysis?",
        "How current does the data need to be — are you focused on the last "
        "quarter, the last year, or a longer historical trend?",
    ]
