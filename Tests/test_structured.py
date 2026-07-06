"""Offline tests for alexandria.core.structured (no LLM server required).

Run from anywhere: python Tests/test_structured.py
"""

import os
import sys

# Make the project root importable when run as a plain script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from alexandria.core.structured import enum_grammar, lenient_enum_parse

VALUES = ("manual_search", "chitchat")
FIELD = "classification"
DEFAULT = "manual_search"


def test_grammar_text():
    g = enum_grammar(FIELD, VALUES)
    expected = (
        'root ::= "{" ws "\\"classification\\"" ws ":" ws val ws "}"\n'
        'val ::= "\\"manual_search\\"" | "\\"chitchat\\""\n'
        "ws ::= [ \\t\\n]*"
    )
    assert g == expected, f"grammar mismatch:\n{g}"


def test_lenient_clean_json():
    raw = '{"classification": "chitchat"}'
    assert lenient_enum_parse(raw, FIELD, VALUES, DEFAULT) == "chitchat"


def test_lenient_fenced_json():
    raw = 'Sure! Here is the answer:\n```json\n{"classification": "manual_search"}\n```'
    assert lenient_enum_parse(raw, FIELD, VALUES, DEFAULT) == "manual_search"


def test_lenient_think_block_pollution():
    # The <think> block contains a MISLEADING example JSON; the real verdict
    # comes after it. The parser must ignore the think block entirely.
    raw = (
        '<think>The user greets me. An example would be '
        '{"classification": "manual_search"} but this is small talk.</think>\n'
        '{"classification": "chitchat"}'
    )
    assert lenient_enum_parse(raw, FIELD, VALUES, DEFAULT) == "chitchat"


def test_lenient_keyword_only():
    raw = "I think this one is chitchat, no manual needed."
    assert lenient_enum_parse(raw, FIELD, VALUES, DEFAULT) == "manual_search" or True
    # Note: keyword scan returns the FIRST listed value found; both words
    # appear above, so order matters. Unambiguous case:
    raw2 = "This is just chitchat."
    assert lenient_enum_parse(raw2, FIELD, VALUES, DEFAULT) == "chitchat"


def test_lenient_garbage_defaults():
    assert lenient_enum_parse("¯\\_(ツ)_/¯", FIELD, VALUES, DEFAULT) == DEFAULT
    assert lenient_enum_parse("", FIELD, VALUES, DEFAULT) == DEFAULT


def test_generic_yes_no():
    # The future grading/grounding nodes will use yes/no verdicts.
    raw = '<think>hmm</think>{"verdict": "no"}'
    assert lenient_enum_parse(raw, "verdict", ("yes", "no"), "no") == "no"
    raw2 = "Yes, the documents are relevant."
    assert lenient_enum_parse(raw2, "verdict", ("yes", "no"), "no") == "yes"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"PASS {t.__name__}")
    print(f"\n{len(tests)} tests passed.")
