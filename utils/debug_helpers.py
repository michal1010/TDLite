_SEPARATOR = "=" * 70


def trim_test_output(output: str) -> str:
    idx = output.find(_SEPARATOR)
    if idx == -1:
        lines = output.strip().splitlines()
        return "\n".join(lines[-5:]) if len(lines) > 5 else output.strip()

    return output[idx:].strip()


def extract_suggested_fix(report: str) -> str:
    for line in report.splitlines():
        if line.startswith("Suggested Fix:"):
            return line[len("Suggested Fix:"):].strip()

    lines = [line.strip() for line in report.splitlines() if line.strip()]
    return lines[-1] if lines else report