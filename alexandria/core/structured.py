"""Constrained structured-output helpers for the local llama.cpp server.

The agentic control flow (routing, document grading, grounding checks) relies
on the LLM returning machine-readable verdicts. Local quantized models cannot
be trusted to emit valid JSON on request alone, so the primary path here uses
llama.cpp's GBNF grammar-constrained decoding: the request carries a grammar
(via the server's `grammar` extension field) and the sampler is only allowed
to emit tokens that match it, making malformed output impossible by
construction.

A layered fallback keeps verdicts flowing even if the constrained call fails
(older server build, template conflict, network error):

    Layer 1: grammar-constrained call -> strict JSON parse
    Layer 2: unconstrained call       -> lenient parse (strip <think> blocks,
                                         extract JSON, keyword scan)
    Layer 3: hard default             -> a verdict node must never crash the graph

Side benefit: because the grammar forbids any free text, reasoning models skip
their <think> preamble on verdict calls, which also makes them faster.
"""

import json
import re
from typing import Sequence


def enum_grammar(field: str, values: Sequence[str]) -> str:
    """Build a GBNF grammar that only accepts {"<field>": "<one of values>"}."""
    alternatives = " | ".join(f'"\\"{v}\\""' for v in values)
    return "\n".join([
        f'root ::= "{{" ws "\\"{field}\\"" ws ":" ws val ws "}}"',
        f"val ::= {alternatives}",
        "ws ::= [ \\t\\n]*",
    ])


def lenient_enum_parse(raw: str, field: str, values: Sequence[str], default: str) -> str:
    """Extract an enum verdict from a noisy, unconstrained LLM response.

    Generalization of the original router band-aid: strip reasoning blocks,
    try strict JSON anywhere in the text, then keyword-scan, then default.
    """
    text = raw.strip()

    # Reasoning models wrap deliberation in <think> blocks that may themselves
    # contain example JSON — drop them before searching for the real verdict.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

    # 1. Try to find a JSON object anywhere in the response
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if match:
        try:
            decision = json.loads(match.group(0))
            value = str(decision.get(field, "")).lower()
            for v in values:
                if v.lower() == value or v.lower() in value:
                    return v
        except json.JSONDecodeError:
            pass

    # 2. Fall back to keyword detection on the raw text
    lowered = text.lower()
    for v in values:
        if v.lower() in lowered:
            return v

    # 3. Safe default
    return default


def invoke_enum(llm, messages, field: str, values: Sequence[str], default: str) -> str:
    """Ask the LLM for a one-field JSON verdict, grammar-constrained.

    Args:
        llm:      a LangChain ChatOpenAI bound to the llama.cpp server.
        messages: the prompt (list of BaseMessages).
        field:    JSON key of the verdict (e.g. "classification", "verdict").
        values:   the allowed enum values; the grammar admits nothing else.
        default:  returned if every layer fails — choose the safe branch.
    """
    grammar = enum_grammar(field, values)

    # Layer 1: grammar-constrained decoding
    try:
        constrained = llm.bind(extra_body={"grammar": grammar})
        response = constrained.invoke(messages)
        value = json.loads(response.content)[field]
        if value in values:
            return value
        print(f"WARNING: constrained decoding returned unexpected value '{value}'.")
    except Exception as e:
        print(f"WARNING: grammar-constrained call failed ({e}). Falling back to lenient parsing.")

    # Layer 2: unconstrained call + lenient parsing
    try:
        response = llm.invoke(messages)
        return lenient_enum_parse(response.content, field, values, default)
    except Exception as e:
        print(f"WARNING: fallback LLM call failed ({e}). Using default '{default}'.")

    # Layer 3: never crash the graph on a verdict
    return default
