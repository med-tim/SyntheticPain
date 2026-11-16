"""
Skin tone fairness analysis for Heintz et al., "Using facial images to classify pain
behaviors in diverse hospitalized patients" (manuscript under review).

Reproduces the fairness results in the paper from a file of model predictions. That file
is patient-level and is not shared, so this script only writes group-level tables.

Input: a CSV with one row per participant and these columns
    y_true          1 = pain behaviors, 0 = no pain behaviors
    fitzpatrick     Fitzpatrick skin type, 1 to 6
    monk            Monk Skin Tone category, 1 to 10
    prob_primary    predicted probability of pain, primary model (standard prompts)
    prob_diversity  predicted probability of pain, sensitivity analysis model
                    (diversity-enhanced prompts)

Usage:
    python fairness_analysis.py predictions.csv [output_folder]

Outputs (group-level only):
    fairness_results.csv     accuracy, TPR, FPR, and selection rate with 95% CIs, and the
                             demographic parity ratio with its 95% CI, per model and group
    heterogeneity_tests.csv  permutation chi-square test of each metric across groups
    model_comparison.csv     diversity minus primary differences with 95% CIs, and the
                             exact McNemar test for accuracy

Methods (details in the paper's Methods and Supplemental Methods):
    - A prediction counts as "pain" when the probability is 0.5 or higher.
    - Monk 9 and 10 are combined because each had fewer than 10 participants.
    - 95% CIs: percentile method, 1,000 bootstrap resamples of participants within each
      group. Both models are scored on the same resamples.
    - Demographic parity ratio: a group's selection rate divided by the reference group's
      (Fitzpatrick II, Monk 2). Its CI comes from resampling each group separately.
    - Differences across groups: chi-square statistic with a P value from 10,000
      permutations of the group labels.
    - The fixed seeds below reproduce the paper's numbers exactly.

Requires Python 3 with numpy and pandas.
"""
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

THRESHOLD = 0.5
MODELS = {"Primary": "prob_primary", "Diversity": "prob_diversity"}
N_BOOT = 1000
N_PERM = 10000
SEED_SUBGROUP = 42        # resamples within each group (CIs and model comparison)
SEED_PARITY = 20261102    # resamples for parity ratio CIs
SEED_PERMUTATION = 7      # permutation tests

ROMAN = {1: "I", 2: "II", 3: "III", 4: "IV", 5: "V", 6: "VI"}
FITZ_GROUPED = {1: "I-II", 2: "I-II", 3: "III-IV", 4: "III-IV", 5: "V-VI", 6: "V-VI"}
MONK_GROUPED = {1: "1-2", 2: "1-2", 3: "3-4", 4: "3-4", 5: "5-6", 6: "5-6",
                7: "7-8", 8: "7-8", 9: "9-10", 10: "9-10"}

# scale: (column, groups in order, reference group for the parity ratio)
SCALES = {
    "Fitzpatrick": ("fitz", ["I", "II", "III", "IV", "V", "VI"], "II"),
    "Monk": ("monk_cat", ["1", "2", "3", "4", "5", "6", "7", "8", "9-10"], "2"),
    "Fitzpatrick, grouped": ("fitz_grouped", ["I-II", "III-IV", "V-VI"], "I-II"),
    "Monk, grouped": ("monk_grouped", ["1-2", "3-4", "5-6", "7-8", "9-10"], "1-2"),
}


def load(path):
    df = pd.read_csv(path)
    needed = ["y_true", "fitzpatrick", "monk", *MODELS.values()]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        sys.exit(f"Missing columns: {', '.join(missing)}")
    if df[needed].isna().any().any():
        sys.exit("Some rows have missing values in the required columns. Remove them first.")

    fitz = df["fitzpatrick"].astype(int)
    monk = df["monk"].astype(int)
    if not (fitz.between(1, 6).all() and monk.between(1, 10).all()):
        sys.exit("fitzpatrick must be 1 to 6 and monk 1 to 10.")
    df["fitz"] = fitz.map(ROMAN)
    df["fitz_grouped"] = fitz.map(FITZ_GROUPED)
    df["monk_cat"] = monk.map(lambda m: "9-10" if m >= 9 else str(m))
    df["monk_grouped"] = monk.map(MONK_GROUPED)
    for model, column in MODELS.items():
        df["pred_" + model] = (df[column] >= THRESHOLD).astype(int)
    return df


def rates(y, pred):
    """Accuracy, TPR, FPR, and selection rate. Works on 1-D arrays, or 2-D arrays with one
    resample per row."""
    tp = ((y == 1) & (pred == 1)).sum(-1)
    fn = ((y == 1) & (pred == 0)).sum(-1)
    fp = ((y == 0) & (pred == 1)).sum(-1)
    tn = ((y == 0) & (pred == 0)).sum(-1)
    n = y.shape[-1]
    with np.errstate(invalid="ignore", divide="ignore"):
        return {"accuracy": (tp + tn) / n, "tpr": tp / (tp + fn),
                "fpr": fp / (fp + tn), "selection_rate": (tp + fp) / n}


def resamples(n):
    """Bootstrap indices for one group: N_BOOT rows of n draws with replacement."""
    rng = np.random.default_rng(SEED_SUBGROUP)
    idx = np.arange(n)
    return np.array([rng.choice(idx, size=n, replace=True) for _ in range(N_BOOT)])


def ci95(values):
    """Percentile 95% CI, skipping resamples where the metric is undefined."""
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return np.nan, np.nan
    return np.percentile(values, 2.5), np.percentile(values, 97.5)


def mcnemar_exact(b, c):
    """Two-sided exact McNemar test (binomial test on the discordant pairs)."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2 ** n)


def all_groups(df):
    yield "Overall", "All", df
    for scale, (column, groups, _) in SCALES.items():
        for group in groups:
            yield scale, group, df[df[column] == group]


def subgroup_metrics(df):
    """Metrics with 95% CIs for each model and group, and the paired model comparison."""
    metric_rows, comparison_rows = [], []
    for scale, group, d in all_groups(df):
        y = d["y_true"].to_numpy()
        s = resamples(len(d))
        observed, boot = {}, {}
        for model in MODELS:
            pred = d["pred_" + model].to_numpy()
            observed[model], boot[model] = rates(y, pred), rates(y[s], pred[s])
            row = {"scale": scale, "group": group, "model": model, "n": len(d), "n_pain": int(y.sum())}
            for metric, value in observed[model].items():
                row[metric] = value
                row[metric + "_lo"], row[metric + "_hi"] = ci95(boot[model][metric])
            metric_rows.append(row)

        primary_ok = d["pred_Primary"].to_numpy() == y
        diversity_ok = d["pred_Diversity"].to_numpy() == y
        row = {"scale": scale, "group": group, "n": len(d)}
        for metric in observed["Primary"]:
            row[metric + "_diff"] = observed["Diversity"][metric] - observed["Primary"][metric]
            row[metric + "_diff_lo"], row[metric + "_diff_hi"] = ci95(boot["Diversity"][metric] - boot["Primary"][metric])
        row["primary_only_correct"] = int((primary_ok & ~diversity_ok).sum())
        row["diversity_only_correct"] = int((~primary_ok & diversity_ok).sum())
        row["mcnemar_p"] = mcnemar_exact(row["primary_only_correct"], row["diversity_only_correct"])
        comparison_rows.append(row)
    return pd.DataFrame(metric_rows), pd.DataFrame(comparison_rows)


def parity_ratios(df):
    """Each group's selection rate divided by the reference group's, with 95% CIs."""
    rng = np.random.default_rng(SEED_PARITY)   # one stream, scales in the order above
    rows = []
    for scale, (column, groups, reference) in SCALES.items():
        data = [df[df[column] == group] for group in groups]
        boot = {model: np.zeros((N_BOOT, len(groups))) for model in MODELS}
        for b in range(N_BOOT):
            for j, d in enumerate(data):
                s = rng.integers(0, len(d), len(d))   # resample this group on its own
                for model in MODELS:
                    boot[model][b, j] = d["pred_" + model].to_numpy()[s].mean()
        r = groups.index(reference)
        for model in MODELS:
            rate = [d["pred_" + model].mean() for d in data]
            for j, group in enumerate(groups):
                with np.errstate(invalid="ignore", divide="ignore"):
                    ratios = boot[model][:, j] / boot[model][:, r]
                ratios[~np.isfinite(ratios)] = np.nan
                lo, hi = ci95(ratios)
                rows.append({"scale": scale, "group": group, "model": model, "reference": reference,
                             "parity_ratio": rate[j] / rate[r], "parity_ratio_lo": lo, "parity_ratio_hi": hi})
    return pd.DataFrame(rows)


def chi_square(codes, x, k):
    """Pearson chi-square for k groups by a binary outcome x. codes holds one row of group
    codes per permutation."""
    n_group = np.stack([np.bincount(c, minlength=k) for c in codes])
    n_yes = np.stack([np.bincount(c, weights=x, minlength=k) for c in codes])
    p_yes = x.sum() / len(x)
    e_yes, e_no = n_group * p_yes, n_group * (1 - p_yes)
    with np.errstate(invalid="ignore", divide="ignore"):
        terms = (n_yes - e_yes) ** 2 / e_yes + ((n_group - n_yes) - e_no) ** 2 / e_no
    return np.nansum(terms, axis=1)


def heterogeneity_tests(df):
    """Does each metric differ across the groups of each scale? Permutation chi-square test."""
    rng = np.random.default_rng(SEED_PERMUTATION)   # one stream, in the loop order below
    everyone = np.ones(len(df), bool)
    rows = []
    for model in MODELS:
        pred = df["pred_" + model]
        tests = [("accuracy", everyone, df["y_true"] == pred),   # correct or not, everyone
                 ("tpr", df["y_true"] == 1, pred),               # called pain, among pain
                 ("fpr", df["y_true"] == 0, pred),               # called pain, among no pain
                 ("selection_rate", everyone, pred)]             # called pain, everyone
        for scale, (column, _, _) in SCALES.items():
            for metric, keep, outcome in tests:
                keep = np.asarray(keep)
                codes, groups = pd.factorize(df.loc[keep, column])
                x = np.asarray(outcome[keep], dtype=float)
                k = len(groups)
                observed = chi_square(codes[None, :], x, k)[0]
                permuted = np.stack([rng.permutation(codes) for _ in range(N_PERM)])
                p = (1 + (chi_square(permuted, x, k) >= observed - 1e-9).sum()) / (N_PERM + 1)
                rows.append({"model": model, "scale": scale, "metric": metric,
                             "chi_square": observed, "p_value": p})
    return pd.DataFrame(rows)


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    df = load(sys.argv[1])
    out = Path(sys.argv[2] if len(sys.argv) > 2 else "fairness_output")
    out.mkdir(parents=True, exist_ok=True)

    metrics, comparison = subgroup_metrics(df)
    parity = parity_ratios(df)
    results = metrics.merge(parity, on=["scale", "group", "model"], how="left")
    tests = heterogeneity_tests(df)

    results.round(4).to_csv(out / "fairness_results.csv", index=False)
    tests.round(4).to_csv(out / "heterogeneity_tests.csv", index=False)
    comparison.round(4).to_csv(out / "model_comparison.csv", index=False)

    print(f"{len(df)} participants, {int(df['y_true'].sum())} with pain behaviors")
    overall = results[results["scale"] == "Overall"].set_index("model")
    for model in MODELS:
        r = overall.loc[model]
        print(f"{model}: accuracy {r.accuracy:.2f} ({r.accuracy_lo:.2f}-{r.accuracy_hi:.2f}), "
              f"TPR {r.tpr:.2f}, FPR {r.fpr:.2f}, selection rate {r.selection_rate:.2f}")
    print(f"Tables written to {out}/")


if __name__ == "__main__":
    main()
