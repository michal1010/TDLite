"""
Shared utilities used by A2.
"""


_SEPARATOR = "=" * 70   # unittest's standard separator line


def trim_test_output(output: str) -> str:
    """
    Keep only the traceback/failure section from unittest -v output.

    Full output has two parts:
      1. Per-test status lines: "test_foo (Tests) ... FAIL"  <- noisy, skip
      2. Tracebacks from === onwards                         <- keep this

    If there is no separator (all tests passed, nothing to trim),
    return the last few lines as a summary.
    """
    idx = output.find(_SEPARATOR)
    if idx == -1:
        lines = output.strip().splitlines()
        return "\n".join(lines[-5:]) if len(lines) > 5 else output.strip()
    return output[idx:].strip()



def extract_suggested_fix(report: str) -> str:
    """
    Pull the 'Suggested Fix' line out of A4's formatted debug report.

    A4 writes each report as:
        Error Analysis: ...
        Root Cause: ...
        Suggested Fix: ...

    Returns just the suggested fix text, or the full report as fallback.
    """
    for line in report.splitlines():
        if line.startswith("Suggested Fix:"):
            return line[len("Suggested Fix:"):].strip()
    # Fallback: return the last non-empty line
    lines = [l.strip() for l in report.splitlines() if l.strip()]
    return lines[-1] if lines else report