"""Drafting, review, and recall prompts used by the evaluation pipeline."""
from __future__ import annotations

PROMPT_DRAFT = """You are an assistant that drafts a first-draft police narrative for a
California peace officer to review and finalize. Use the body-worn-camera
transcript below. Follow the {category} category template (incident,
contact, observations, statements, actions taken, evidence). Preserve
verbatim any utterances enclosed in <Q>...</Q> tags. Do not invent
people, places, dates, or facts not present in the transcript. If
information is missing, write [TO BE COMPLETED BY OFFICER].

INCIDENT CATEGORY: {category}
TRANSCRIPT:
{transcript}

DRAFT:
"""

PROMPT_REVIEW = """You are a peace officer reviewing an AI-generated draft of your own
report. Edit the draft for accuracy. Remove any sentence that is
factually impossible or that does not reflect what you observed.
Output the final report only.

DRAFT:
{draft}
"""

# Axis C (C2): latent-recall prompt -- the reviewer recalls facts from memory.
PROMPT_RECALL = """You are a peace officer who just finished an encounter.  WITHOUT
re-reading the body-worn-camera transcript, list every fact you remember
from the encounter as a numbered list.  Be terse: one fact per line.

ENCOUNTER TYPE: {category}

FACTS:
"""

# Optional LLM-as-judge prompt, handy for spot-checking a flagged metric by
# hand against the transcript and draft.
PROMPT_JUDGE = """You are an impartial expert evaluator.  Read the body-worn-camera
TRANSCRIPT and the AI-DRAFT report below.  An automatic metric flagged
the following claim about the draft:

CLAIM: {claim}
METRIC DEFINITION: {definition}

Answer "AGREE", "DISAGREE", or "UNCERTAIN" on a single line, followed by
one sentence of justification.

TRANSCRIPT:
{transcript}

AI-DRAFT:
{draft}

VERDICT:
"""


def render_prompt_draft(transcript: str, category: str) -> str:
    return PROMPT_DRAFT.format(category=category, transcript=transcript)


def render_prompt_review(draft: str) -> str:
    return PROMPT_REVIEW.format(draft=draft)


def render_prompt_recall(category: str) -> str:
    return PROMPT_RECALL.format(category=category)


def render_prompt_judge(transcript: str, draft: str, claim: str, definition: str) -> str:
    return PROMPT_JUDGE.format(
        transcript=transcript, draft=draft, claim=claim, definition=definition
    )
