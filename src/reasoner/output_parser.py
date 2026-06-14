# src/reasoner/output_parser.py
import re
from typing import Tuple, Optional


class OutputParser:
    """Parses model output to extract answer."""

    def __init__(self):
        self.insufficient_patterns = [
            r"INSUFFICIENT_EVIDENCE",
            r"insufficient evidence",
            r"cannot answer",
            r"not enough information"
        ]

    def parse(self, output: str, num_choices: int) -> Tuple[Optional[int], bool]:
        """
        Parse model output to get answer index.

        Returns:
            (answer_index, is_insufficient)
        """
        output_clean = output.strip().upper()

        # Check for insufficient evidence
        for pattern in self.insufficient_patterns:
            if re.search(pattern, output_clean, re.IGNORECASE):
                return None, True

        # Look for single letter answer
        letter_match = re.search(r'^([A-J])\b', output_clean)
        if letter_match:
            letter = letter_match.group(1)
            idx = ord(letter) - ord('A')
            if 0 <= idx < num_choices:
                return idx, False

        # Look for answer anywhere in output
        letter_match = re.search(r'([A-J])\b', output_clean)
        if letter_match:
            letter = letter_match.group(1)
            idx = ord(letter) - ord('A')
            if 0 <= idx < num_choices:
                return idx, False

        return None, False

    def parse_with_confidence(self, output: str, num_choices: int) -> Tuple[Optional[int], bool, float]:
        """
        Parse with confidence score based on output patterns.

        Returns:
            (answer_index, is_insufficient, confidence)
        """
        answer_idx, is_insufficient = self.parse(output, num_choices)

        # Calculate confidence based on output clarity
        confidence = 0.5
        if answer_idx is not None:
            output_clean = output.strip().upper()
            if re.match(r'^[A-J]$', output_clean):
                confidence = 1.0
            elif re.search(rf'^{chr(ord("A") + answer_idx)}[^A-Z]', output_clean):
                confidence = 0.9
            else:
                confidence = 0.7

        return answer_idx, is_insufficient, confidence