"""
Calculator Tool — safe math evaluation for FLOPs and parameter counts.
======================================================================

Why is a calculator tool needed?
----------------------------------
Test question Q3 asks: "Computational complexity of self-attention vs.
recurrent layers — show the math."

The paper states:
  - Self-attention:  O(n² · d)  per layer
  - Recurrent:       O(n · d²)  per layer

To answer which is better and *when*, the agent needs to actually compare
these expressions for typical values (n=512 sequence length, d=512 model dim):
  - Self-attention:  512² × 512  = 134,217,728  ≈ 1.34 × 10⁸  ops
  - Recurrent:       512  × 512² = 134,217,728  ≈ 1.34 × 10⁸  ops  (same here)
  - But for n > d: self-attention grows as n², recurrence grows as d²

The LLM knows these formulas conceptually but can make arithmetic errors for
large numbers. Routing through a calculator ensures precision.

Safety of eval()
-----------------
We use Python's eval() with a restricted namespace — only math functions are
allowed. No builtins, no imports, no file access. This is safe for controlled
inputs from our own LLM-generated expressions.
"""

from __future__ import annotations

import math
import re

from langchain_core.tools import tool

# Allowed names in eval namespace — math functions only
_SAFE_MATH_NAMESPACE = {
    "abs": abs, "round": round, "min": min, "max": max,
    "pow": pow, "sum": sum,
    # math module functions
    "sqrt": math.sqrt, "log": math.log, "log2": math.log2,
    "log10": math.log10, "exp": math.exp, "ceil": math.ceil,
    "floor": math.floor, "pi": math.pi, "e": math.e,
    # Scientific notation helpers
    "n": None,  # placeholder — overridden per call
    "d": None,  # placeholder — overridden per call
    "k": None,  # placeholder — overridden per call
    "h": None,  # placeholder — overridden per call
}

# Blocked patterns — prevent any import or dunder access
_BLOCKED_PATTERNS = re.compile(
    r"\b(import|exec|eval|open|__|\bos\b|\bsys\b)\b", re.IGNORECASE
)


@tool
def calculate(expression: str, variables: dict | None = None) -> str:
    """
    Safely evaluate a mathematical expression and return the result.

    Supports standard Python math operators (+, -, *, **, /) and functions
    from the math module (sqrt, log, log2, log10, exp, etc.).

    Variables can be passed as a dict, e.g. {"n": 512, "d": 512, "h": 8}.
    These are substituted into the expression before evaluation.

    Args:
        expression: A mathematical expression as a string.
                    Example: "n**2 * d"  or  "sqrt(d_k)"  or  "2.3e19 / 8"
        variables:  Optional dict of variable substitutions.
                    Example: {"n": 512, "d": 512, "d_k": 64}

    Returns:
        The computed result as a formatted string, or an error message.

    Examples:
        calculate("n**2 * d", {"n": 512, "d": 512})
        → "n² × d = 512² × 512 = 134,217,728 ≈ 1.34e+08"

        calculate("n * d**2", {"n": 512, "d": 512})
        → "Result: 134,217,728 ≈ 1.34e+08"

        calculate("28.4 - 25.16")
        → "Result: 3.24"
    """
    if not expression or not expression.strip():
        return "Error: Empty expression."

    # Security check — block dangerous patterns
    if _BLOCKED_PATTERNS.search(expression):
        return "Error: Unsafe expression — only math operations are allowed."

    # Build evaluation namespace
    namespace = dict(_SAFE_MATH_NAMESPACE)
    if variables:
        # Validate variable values are numeric
        for k, v in variables.items():
            if not isinstance(v, (int, float)):
                return f"Error: Variable '{k}' must be a number, got {type(v).__name__}."
            namespace[k] = v

    # Also handle variables embedded in the expression string (e.g. n=512)
    # Pattern: "n=512, d=512" at the start of the expression
    inline_var_match = re.match(
        r"^((?:\w+\s*=\s*[\d.e+\-]+\s*,?\s*)+)(.*)", expression.strip()
    )
    if inline_var_match:
        var_str = inline_var_match.group(1)
        expression = inline_var_match.group(2).strip()
        for pair in re.findall(r"(\w+)\s*=\s*([\d.e+\-]+)", var_str):
            var_name, var_val = pair
            try:
                namespace[var_name] = float(var_val)
            except ValueError:
                pass

    if not expression:
        return "Error: No expression to evaluate after parsing variables."

    try:
        result = eval(expression, {"__builtins__": {}}, namespace)  # noqa: S307
    except ZeroDivisionError:
        return "Error: Division by zero."
    except NameError as e:
        return f"Error: Unknown variable or function — {e}. Pass values as variables dict."
    except SyntaxError as e:
        return f"Error: Invalid syntax in expression — {e}."
    except Exception as e:
        return f"Error: {e}"

    # Format result
    if isinstance(result, float):
        if result == int(result) and abs(result) < 1e15:
            result_int = int(result)
            return f"Result: {result_int:,} ≈ {result:.3e}"
        return f"Result: {result:.6g}"
    elif isinstance(result, int):
        return f"Result: {result:,}"
    else:
        return f"Result: {result}"


# ---------------------------------------------------------------------------
# Standalone helper for the complexity question (Q3 specific)
# ---------------------------------------------------------------------------

def compare_complexities(n: int = 512, d: int = 512, k: int = 1) -> str:
    """
    Compare self-attention vs recurrent layer complexity for given parameters.

    From Table 1 of the paper:
      Self-Attention:           O(n² · d)    sequential=O(1)
      Recurrent:                O(n · d²)    sequential=O(n)
      Convolutional:            O(k · n · d) sequential=O(1)
      Self-Attention (restrict):O(r · n · d) sequential=O(1)

    Args:
        n: Sequence length (default 512, as used in the paper)
        d: Model dimension (default 512, as used in the paper)
        k: Kernel size for convolution (default 1)

    Returns:
        A formatted comparison string.
    """
    self_attn = n ** 2 * d
    recurrent = n * d ** 2
    conv = k * n * d

    lines = [
        f"Complexity comparison (n={n}, d={d}, k={k}):",
        f"",
        f"  Self-Attention:  O(n²·d)  = {n}² × {d} = {self_attn:,}  (≈{self_attn:.2e})",
        f"  Recurrent:       O(n·d²)  = {n} × {d}² = {recurrent:,}  (≈{recurrent:.2e})",
        f"  Convolutional:   O(k·n·d) = {k} × {n} × {d} = {conv:,}  (≈{conv:.2e})",
        f"",
        f"  Ratio self-attn / recurrent = {self_attn / recurrent:.3f}",
        f"  → When n < d: self-attention is CHEAPER than recurrent",
        f"  → When n > d: self-attention is MORE EXPENSIVE than recurrent",
        f"",
        f"  For n=d={n}: complexities are identical ({self_attn:,} ops)",
        f"  The paper notes that for typical sequences n << d, so self-attention wins.",
        f"  Additionally, self-attention has O(1) sequential ops vs O(n) for recurrent,",
        f"  enabling full parallelisation.",
    ]
    return "\n".join(lines)
