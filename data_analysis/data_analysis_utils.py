import glob
import json
import os
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, kruskal, mannwhitneyu

def aggregate_csv(prefix: str, subfolders: list[str], root: str = ".") -> pd.DataFrame:
    """Load and concatenate results CSVs for all batches of a setup.

    Expects each subfolder to contain a results.csv.

    Args:
        prefix:     Setup name, used to locate <root>/<prefix>/<subfolder>/.
                    If the subfolders live directly under root, pass prefix="".
        subfolders: List of batch subfolder names.
        root:       Base directory that contains the setup folder (default: cwd).

    Returns:
        Combined DataFrame with a 'batch' column.
    """
    frames = []
    for sub in subfolders:
        folder = os.path.join(root, prefix, sub) if prefix else os.path.join(root, sub)
        csv_path = os.path.join(folder, "results.csv")

        passed_csv_path = os.path.join(folder, "pass_per_instance.csv")
        passed_df = pd.read_csv(passed_csv_path)

        batch_df = pd.read_csv(csv_path)
        batch_df.insert(0, "batch", sub)
        batch_df.insert(1, "passed_hidden_tests", passed_df["passed"])

        frames.append(batch_df)

    df = pd.concat(frames, ignore_index=True)
    return df


def aggregate_csv_scenarios(
    agent_scenarios: dict[str, tuple[str, list[str]]],
    root: str = ".",
) -> pd.DataFrame:
    """Load and concatenate per-instance CSVs for several scenarios.

    Adds a "scenario" column so downstream comparisons can group rows by setup.
    """
    frames = []
    for scenario, (prefix, subfolders) in agent_scenarios.items():
        if not subfolders:
            continue

        scenario_df = aggregate_csv(prefix=prefix, subfolders=subfolders, root=root)
        scenario_df.insert(0, "scenario", scenario)
        frames.append(scenario_df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, ignore_index=True)

def aggregate_json(agent_scenarios: dict[str, list[tuple[str, str]]], root: str = ".") -> pd.DataFrame:
    SUM_FIELDS = {"total", "passed", "tokens_input", "tokens_output"}
    AVG_FIELDS = {"avg_iters", "avg_line_coverage", "avg_branch_coverage", "avg_mutation_score"}

    def flatten_summary(summary: dict) -> dict:
        row = {
            "total":               summary["total"],
            "passed":              summary["passed"],
            "avg_iters":           summary["avg_iters"],
            "tokens_input":        summary["tokens"]["input"],
            "tokens_output":       summary["tokens"]["output"],
            "avg_line_coverage":   summary["avg_line_coverage"],
            "avg_branch_coverage": summary["avg_branch_coverage"],
            "avg_mutation_score":  summary["avg_mutation_score"],
        }
        for agent, vals in summary["tokens"]["by_agent"].items():
            row[f"tokens_{agent}_input"]  = vals["input"]
            row[f"tokens_{agent}_output"] = vals["output"]
            SUM_FIELDS.update({f"tokens_{agent}_input", f"tokens_{agent}_output"})
        return row

    accumulator: dict[str, dict[str, float]] = {}
    avg_counts:  dict[str, dict[str, int]]   = {}

    for scenario in agent_scenarios:
        accumulator[scenario] = {}
        avg_counts[scenario]  = {}

        prefix, subfolders = agent_scenarios[scenario]

        for subfolder in subfolders:
            folder = os.path.join(root, prefix, subfolder) if prefix else os.path.join(root, subfolder)
            json_matches = glob.glob(os.path.join(folder, "summary_*.json"))
            if not json_matches:
                continue

            with open(json_matches[0]) as f:
                summary = json.load(f)

            for key, val in flatten_summary(summary).items():
                if key in AVG_FIELDS:
                    # Exclude sentinel -1.0 from averages (no data recorded)
                    if val != -1.0:
                        accumulator[scenario][key] = accumulator[scenario].get(key, 0.0) + val
                        avg_counts[scenario][key]  = avg_counts[scenario].get(key, 0) + 1
                    else:
                        accumulator[scenario].setdefault(key, -1.0)
                        avg_counts[scenario].setdefault(key, 0)
                else:  # SUM_FIELDS
                    accumulator[scenario][key] = accumulator[scenario].get(key, 0) + val

        # Finalise averages
        for key in AVG_FIELDS:
            n = avg_counts[scenario].get(key, 0)
            if n > 0:
                accumulator[scenario][key] = accumulator[scenario][key] / n
            else:
                accumulator[scenario][key] = -1.0  # Propagate sentinel if no valid data

        # Derive pass_rate from aggregated totals (more accurate than averaging rates)
        total = accumulator[scenario].get("total", 0)
        accumulator[scenario]["pass_rate"] = (
            accumulator[scenario].get("passed", 0) / total if total > 0 else 0.0
        )

    # Build DataFrame with a consistent column order
    column_order = [
        "total", "passed", "pass_rate", "avg_iters",
        "tokens_input", "tokens_output",
        "avg_line_coverage", "avg_branch_coverage", "avg_mutation_score",
    ]
    df = pd.DataFrame.from_dict(accumulator, orient="index")
    df.index.name = "scenario"

    # Put known columns first, then any dynamically discovered token-by-agent columns
    agent_cols = sorted(c for c in df.columns if c not in column_order)
    df = df[column_order + agent_cols]

    return df


def add_unrolled_token_cost_column(
    df: pd.DataFrame,
    input_col: str = "total_input_tokens",
    output_col: str = "total_output_tokens",
    cost_col: str = "token_cost_unrolled",
    cumulative_cost_col: str = "token_cost_cumulative",
    group_cols: tuple[str, ...] = ("scenario", "batch"),
    order_col: str = "instance_id",
) -> pd.DataFrame:
    """Return a copy of df with per-instance token cost unrolled from cumulative totals."""
    result = df.copy()
    result[cumulative_cost_col] = (
        pd.to_numeric(result[input_col], errors="coerce").fillna(0)
        + pd.to_numeric(result[output_col], errors="coerce").fillna(0)
    )

    group_keys = [col for col in group_cols if col in result.columns]
    sort_cols = group_keys[:]
    if order_col in result.columns:
        sort_cols.append(order_col)

    if sort_cols:
        work = result.sort_values(sort_cols, kind="mergesort").copy()
    else:
        work = result.copy()

    if group_keys:
        work[cost_col] = work.groupby(group_keys, sort=False)[cumulative_cost_col].diff()
    else:
        work[cost_col] = work[cumulative_cost_col].diff()

    work[cost_col] = work[cost_col].fillna(work[cumulative_cost_col])
    work.loc[work[cost_col] < 0, cost_col] = work.loc[work[cost_col] < 0, cumulative_cost_col]

    if sort_cols:
        work = work.sort_index()

    return work


def add_token_cost_column(
    df: pd.DataFrame,
    input_col: str = "total_input_tokens",
    output_col: str = "total_output_tokens",
    cost_col: str = "token_cost_unrolled",
    cumulative_cost_col: str = "token_cost_cumulative",
    group_cols: tuple[str, ...] = ("scenario", "batch"),
    order_col: str = "instance_id",
) -> pd.DataFrame:
    """Backward-compatible alias for adding the unrolled token-cost column."""
    return add_unrolled_token_cost_column(
        df,
        input_col=input_col,
        output_col=output_col,
        cost_col=cost_col,
        cumulative_cost_col=cumulative_cost_col,
        group_cols=group_cols,
        order_col=order_col,
    )


def summarize_numeric_by_group(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    *,
    drop_negative: bool = False,
) -> pd.DataFrame:
    """Summarize one numeric metric by group with simple descriptive stats."""
    work = df[[group_col, value_col]].copy()
    work = work.dropna(subset=[group_col, value_col])
    work[value_col] = pd.to_numeric(work[value_col], errors="coerce")
    work = work.dropna(subset=[value_col])

    if drop_negative:
        work = work[work[value_col] >= 0]

    summary = work.groupby(group_col)[value_col].agg(["count", "mean", "median", "std", "min", "max"])
    return summary.sort_index()


def _ensure_group_column(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if group_col in df.columns:
        return df.copy()

    if df.index.name == group_col:
        return df.reset_index()

    if group_col == "scenario" and "scenario" not in df.columns:
        return df.reset_index().rename(columns={df.index.name or "index": "scenario"})

    raise KeyError(f"{group_col!r} not found in dataframe columns or index")

def compare_metric_between_groups(
    df: pd.DataFrame,
    group_col: str,
    value_col: str,
    left_group,
    right_group,
    *,
    drop_negative: bool = False,
) -> pd.DataFrame:
    """Compare one metric for two groups and return a one-row summary table."""
    work = _ensure_group_column(df, group_col)
    left = work.loc[work[group_col] == left_group, value_col]
    right = work.loc[work[group_col] == right_group, value_col]

    if drop_negative:
        left = left[left >= 0]
        right = right[right >= 0]

    u_stat, p_value = mannwhitneyu(left, right, alternative="two-sided")

    summary = pd.DataFrame(
        [
            {
                group_col: left_group,
                "comparison_group": right_group,
                "metric": value_col,
                "left_mean": left.mean(),
                "right_mean": right.mean(),
                "p_value": p_value,
                "significant_0.05": p_value < 0.05,
            }
        ]
    )
    return summary


def compare_totals_between_groups(
    df: pd.DataFrame,
    left_group,
    right_group,
    value_col: str,
    group_col: str = "scenario",
    total_col: str = "total",
) -> pd.DataFrame:
    work = _ensure_group_column(df, group_col)
    left_rows = work.loc[work[group_col] == left_group]
    right_rows = work.loc[work[group_col] == right_group]

    if len(left_rows) != 1 or len(right_rows) != 1:
        raise ValueError("Each group must map to exactly one summary row")

    left_row = left_rows.iloc[0]
    right_row = right_rows.iloc[0]

    n_left = int(left_row[total_col])
    n_right = int(right_row[total_col])
    
    left_successes = int(left_row[value_col])
    right_successes = int(right_row[value_col])

    left_fails = max(n_left - left_successes, 0)
    right_fails = max(n_right - right_successes, 0)

    table = np.asarray([[left_successes, left_fails],
                        [right_successes, right_fails]])

    odds_ratio, p_value = fisher_exact(table, alternative="two-sided")
    left_rate = left_successes / n_left if n_left > 0 else 0.0
    right_rate = right_successes / n_right if n_right > 0 else 0.0

    summary = pd.DataFrame(
        [
            {
                group_col: left_group,
                "comparison_group": right_group,
                "metric": value_col,
                "left_rate": left_rate,
                "right_rate": right_rate,
                "n_left": float(n_left),
                "n_right": float(n_right),
                "successes_left": float(left_successes),
                "successes_right": float(right_successes),
                "p_value": float(p_value),
                "significant_0.05": float(p_value) < 0.05,
            }
        ]
    )
    return summary


def compare_pass_rates(
    df: pd.DataFrame,
    left_group,
    right_group,
    *,
    group_col: str = "scenario",
    pass_col: str = "pipeline_tests_passed",
    total_col: str = "total",
    passed_col: str = "passed",
    pass_rate_col: str = "pass_rate",
) -> pd.DataFrame:
    """Compare pass rates between two scenarios or batches.

    Uses summary totals/passed counts when the dataframe provides them; otherwise
    falls back to the per-instance pass column.
    """
    work = _ensure_group_column(df, group_col)

    if total_col in work.columns and (passed_col in work.columns or pass_rate_col in work.columns):
        return compare_totals_between_groups(
            work,
            left_group,
            right_group,
            group_col=group_col,
            total_col=total_col,
            value_col=passed_col,
        )

    work = work.copy()
    work[pass_col] = work[pass_col].astype(int)
    return compare_metric_between_groups(
        work,
        group_col,
        pass_col,
        left_group,
        right_group,
    )


def compare_token_costs(
    df: pd.DataFrame,
    left_group,
    right_group,
    *,
    group_col: str = "scenario",
    input_col: str = "total_input_tokens",
    output_col: str = "total_output_tokens",
    cost_col: str = "token_cost_unrolled",
) -> pd.DataFrame:
    """Compare token cost between two scenarios or batches."""
    work = _ensure_group_column(df, group_col)
    if cost_col not in work.columns:
        work = add_unrolled_token_cost_column(work, input_col=input_col, output_col=output_col, cost_col=cost_col)

    return compare_metric_between_groups(
        work,
        group_col,
        cost_col,
        left_group,
        right_group,
    )


def compare_quality_metric(
    df: pd.DataFrame,
    left_group,
    right_group,
    *,
    metric_col: str = "line_coverage_percent",
    group_col: str = "scenario",
    drop_negative: bool = True,
) -> pd.DataFrame:
    """Compare one quality metric between two scenarios or batches."""
    return compare_metric_between_groups(
        df,
        group_col,
        metric_col,
        left_group,
        right_group,
        drop_negative=drop_negative,
    )


def plot_token_cost_distribution(
    df: pd.DataFrame,
    *,
    group_col: str = "scenario",
    input_col: str = "total_input_tokens",
    output_col: str = "total_output_tokens",
    cost_col: str = "token_cost_unrolled",
    ax=None,
):
    """Draw a boxplot of token cost by group."""
    work = _ensure_group_column(df, group_col)
    if cost_col not in work.columns:
        work = add_unrolled_token_cost_column(work, input_col=input_col, output_col=output_col, cost_col=cost_col)

    grouped = [group[cost_col].dropna().tolist() for _, group in work.groupby(group_col)]
    labels = list(work.groupby(group_col).groups.keys())

    if ax is None:
        _, ax = plt.subplots(figsize=(10, 5))

    ax.boxplot(grouped, tick_labels=labels)
    ax.set_xlabel("Scenario")
    ax.set_yscale('log')
    ax.set_ylabel("Token cost of instance")
    ax.set_title("Token cost distribution by group")
    ax.tick_params(axis="x", rotation=45)
    return ax


def plot_metric_relationships(
    df: pd.DataFrame,
    *,
    x_col: str = "token_cost_unrolled",
    y_cols: list[str] | None = None,
    group_col: str | None = "scenario",
    input_col: str = "total_input_tokens",
    output_col: str = "total_output_tokens",
    cost_col: str = "token_cost_unrolled",
    drop_negative: bool = True,
):
    """Plot simple scatter relationships between token cost and quality metrics."""
    work = df.copy()
    if x_col == cost_col and cost_col not in work.columns:
        work = add_unrolled_token_cost_column(work, input_col=input_col, output_col=output_col, cost_col=cost_col)

    if y_cols is None:
        y_cols = ["pipeline_iterations", "line_coverage_percent", "branch_coverage_percent", "mutation_score"]

    if group_col and group_col not in work.columns and work.index.name == group_col:
        work = work.reset_index()

    if drop_negative:
        for col in y_cols:
            if col in work.columns:
                work = work[work[col].isna() | (work[col] >= 0)]

    n_plots = len(y_cols)
    fig, axes = plt.subplots(n_plots, 1, figsize=(10, 4 * n_plots), squeeze=False)

    for axis, y_col in zip(axes.flat, y_cols):
        if y_col not in work.columns:
            axis.set_visible(False)
            continue

        if group_col and group_col in work.columns:
            for group_name, group in work.groupby(group_col):
                axis.scatter(group[x_col], group[y_col], alpha=0.7, label=str(group_name))
            axis.legend()
        else:
            axis.scatter(work[x_col], work[y_col], alpha=0.7)

        axis.set_xlabel(x_col)
        axis.set_ylabel(y_col)
        axis.set_title(f"{y_col} vs {x_col}")

    fig.tight_layout()
    return axes
