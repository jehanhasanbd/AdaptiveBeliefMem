# src/reasoner/prompt_builder.py
import json
from typing import Dict, Any, List


class PromptBuilder:
    """Builds inference prompts from decision traces."""

    def __init__(self):
        self.system_prompt = """You are an AI assistant answering questions based on conversation history."""

    def build_prompt(self, trace: Dict[str, Any], question: str, choices: List[str]) -> str:
        """Build complete inference prompt."""
        trace_json = json.dumps(trace, indent=2, default=str)

        # Format choices with letters
        choice_lines = []
        for i, choice in enumerate(choices):
            letter = chr(ord('A') + i)
            choice_lines.append(f"{letter}. {choice}")

        choices_text = "\n".join(choice_lines)

        prompt = f"""{self.system_prompt}

## RELEVANT INFORMATION EXTRACTED (Decision Trace):
{trace_json}

## QUESTION:
{question}

## CHOICES:
{choices_text}

## INSTRUCTIONS:
1. Answer using ONLY the information in the Relevant Information section.
2. If the information is insufficient, state "INSUFFICIENT_EVIDENCE".
3. Output format: Single letter (A-{chr(ord('A') + len(choices) - 1)}) or "INSUFFICIENT_EVIDENCE".

## ANSWER:
"""
        return prompt

    def build_without_trace(self, question: str, choices: List[str]) -> str:
        """Build prompt without decision trace (for baselines)."""
        choice_lines = []
        for i, choice in enumerate(choices):
            letter = chr(ord('A') + i)
            choice_lines.append(f"{letter}. {choice}")

        choices_text = "\n".join(choice_lines)

        prompt = f"""{self.system_prompt}

## QUESTION:
{question}

## CHOICES:
{choices_text}

## INSTRUCTIONS:
Answer with a single letter (A-{chr(ord('A') + len(choices) - 1)}) or "INSUFFICIENT_EVIDENCE".

## ANSWER:
"""
        return prompt