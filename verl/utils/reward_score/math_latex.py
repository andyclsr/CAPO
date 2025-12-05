
import re
from typing import Optional

from latex2sympy2_extended import NormalizationConfig
from math_verify import LatexExtractionConfig, parse, verify
import warnings

# import logging
# logging.getLogger('math_verify').setLevel(logging.ERROR)

# The message to match (can be a regex)
warning_message = r"equations is deprecated, as it handled by the parser now"
# The module where the warning originates (can be a regex)
warning_module = r"latex2sympy2_extended\.math_normalization" # Escape the dot for regex

# Ignore specific warnings that match the message and module
warnings.filterwarnings("ignore", message=warning_message, module=warning_module)

GENERATION_EXTRACTION_CONFIG = [
    LatexExtractionConfig(
        normalization_config=NormalizationConfig(
            nits=False, malformed_operators=False, basic_latex=True, boxed="all", units=True,
        ),
        boxed_match_priority=0,
        try_extract_without_anchor=False, # Strict boxed extraction preferred for math
    )
]
GOLD_EXTRACTION_CONFIG = [LatexExtractionConfig()]

def verify_answer_math_verify(generation_output: str, ground_truth: str):
    """Verifies math-style answers using math_verify."""
    gold_parsed_list = parse(ground_truth, extraction_mode="first_match", extraction_config=GOLD_EXTRACTION_CONFIG) # , parsing_timeout=None
    if not gold_parsed_list:
        return None, None, False
    gold_parsed = gold_parsed_list[0]

    answer_parsed_list = parse(generation_output, extraction_config=GENERATION_EXTRACTION_CONFIG, extraction_mode="first_match") # , parsing_timeout=None
    if not answer_parsed_list:
        return gold_parsed, None, False
    answer_parsed = answer_parsed_list[0]

    is_match = verify(answer_parsed, gold_parsed)
    return gold_parsed, answer_parsed, bool(is_match)


# def verify(
#     solution_str: str, answer: str, strict_box_verify: bool = False, pause_tokens_index: Optional[list[int]] = None
# ) -> bool:
#     """Verify if the solution is correct.

#     Args:
#         solution_str: The solution string to verify
#         answer: The ground truth answer
#         strict_box_verify: Whether to use strict box verification
#         pause_tokens_index: Indices of pause tokens

#     Returns:
#         True if the solution is correct, False otherwise
#     """
#     return verify_answer_math_verify(solution_str, answer)


def compute_score(
    solution_str: str,
    ground_truth: str,
    strict_box_verify: bool = False,
    pause_tokens_index: Optional[list[int]] = None,
) -> float:
    """Compute the reward score for a solution.

    Args:
        solution_str: The solution string
        ground_truth: The ground truth answer
        strict_box_verify: Whether to use strict box verification
        pause_tokens_index: Indices of pause tokens

    Returns:
        Reward score (1.0 for correct, -1.0 for incorrect)
    """
    # Limit solution length for efficiency
    solution_str = solution_str[-300:]  # The longest answer in MATH-500 has 159 characters

    # Verify the solution
    gold, pred, correct = verify_answer_math_verify(solution_str, ground_truth)

    reward = 1.0 if correct else -1.0
    acc = correct

    return {
        "score": reward,
        "acc": acc,
        "pred": pred,
        "gold": gold,
    }
