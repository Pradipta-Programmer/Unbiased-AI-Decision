"""
╔══════════════════════════════════════════════════════════════╗
║        UNBIASED AI DECISION SYSTEM — Open Innovation         ║
║        H2S Hackathon Prototype | Python 3.x                  ║
╚══════════════════════════════════════════════════════════════╝

Challenge: [Unbiased AI Decision] Ensuring Fairness and
           Detecting Bias in Automated Decisions

This prototype demonstrates:
  1. Bias Detection across demographic features
  2. Fairness Metrics (Demographic Parity, Equal Opportunity,
     Disparate Impact, Equalized Odds)
  3. Bias Mitigation via re-weighting
  4. Transparent Explainability via feature importance
  5. Audit Trail for every decision

Run:  python unbiased_ai_decision.py
"""

import random
import math
import json
from collections import defaultdict


# ─────────────────────────────────────────────
#  SECTION 1 — SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────

def generate_dataset(n=500, seed=42):
    """
    Simulate a loan-approval dataset with realistic (and biased) patterns.
    Features: credit_score, income, debt_ratio, age, gender, ethnicity
    Label:    approved (0 = denied, 1 = approved)
    """
    random.seed(seed)
    data = []

    genders    = ["Male", "Female"]
    ethnicities = ["GroupA", "GroupB", "GroupC"]   # anonymised

    for i in range(n):
        gender    = random.choice(genders)
        ethnicity = random.choices(ethnicities, weights=[0.5, 0.3, 0.2])[0]

        # Inject structural bias: GroupB gets lower income on average
        income_base = {"GroupA": 55000, "GroupB": 42000, "GroupC": 50000}[ethnicity]
        income      = max(15000, int(random.gauss(income_base, 12000)))

        credit_score = int(random.gauss(680, 80))
        credit_score = max(300, min(850, credit_score))

        debt_ratio   = round(random.uniform(0.1, 0.7), 2)
        age          = int(random.gauss(38, 10))
        age          = max(18, min(70, age))

        # "True" merit-based label (no demographic influence)
        merit_score = (
            (credit_score - 300) / 550 * 0.5 +
            min(income, 100000) / 100000 * 0.3 +
            (1 - debt_ratio) * 0.2
        )
        true_label = 1 if merit_score > 0.45 else 0

        # Biased label: GroupB has +15% denial rate added artificially
        biased_label = true_label
        if ethnicity == "GroupB" and true_label == 1:
            if random.random() < 0.25:   # flip 25% of approvals → denial
                biased_label = 0
        if gender == "Female" and true_label == 1:
            if random.random() < 0.10:   # flip 10% of approvals → denial
                biased_label = 0

        data.append({
            "id":           i,
            "credit_score": credit_score,
            "income":       income,
            "debt_ratio":   debt_ratio,
            "age":          age,
            "gender":       gender,
            "ethnicity":    ethnicity,
            "true_label":   true_label,
            "label":        biased_label,   # what the biased model learned
        })

    return data


# ─────────────────────────────────────────────
#  SECTION 2 — SIMPLE LOGISTIC REGRESSION
#  (from scratch, no external libraries)
# ─────────────────────────────────────────────

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, z))))


def normalize(data, feature_cols):
    stats = {}
    for col in feature_cols:
        vals = [row[col] for row in data]
        mu  = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1
        stats[col] = (mu, std)
    return stats


def featurize(row, feature_cols, stats):
    return [(row[c] - stats[c][0]) / stats[c][1] for c in feature_cols]


def train_logistic(data, feature_cols, label_col, stats,
                   epochs=200, lr=0.05, weights_override=None):
    """Mini logistic regression with optional sample weighting for mitigation."""
    n_feat  = len(feature_cols)
    w       = [0.0] * n_feat
    b       = 0.0
    weights = weights_override or [1.0] * len(data)

    for _ in range(epochs):
        dw = [0.0] * n_feat
        db = 0.0
        total_w = sum(weights)

        for row, wt in zip(data, weights):
            x   = featurize(row, feature_cols, stats)
            y   = row[label_col]
            pred = sigmoid(sum(wi * xi for wi, xi in zip(w, x)) + b)
            err  = (pred - y) * wt / total_w
            for j in range(n_feat):
                dw[j] += err * x[j]
            db += err

        for j in range(n_feat):
            w[j] -= lr * dw[j]
        b -= lr * db

    return w, b


def predict(row, w, b, feature_cols, stats, threshold=0.5):
    x    = featurize(row, feature_cols, stats)
    prob = sigmoid(sum(wi * xi for wi, xi in zip(w, x)) + b)
    return int(prob >= threshold), round(prob, 4)


# ─────────────────────────────────────────────
#  SECTION 3 — FAIRNESS METRICS
# ─────────────────────────────────────────────

def compute_fairness(data, predictions, sensitive_attr):
    """
    Compute four canonical fairness metrics for a binary sensitive attribute.
    Returns a dict of per-group stats and disparity ratios.
    """
    groups = defaultdict(lambda: {"TP": 0, "FP": 0, "TN": 0, "FN": 0,
                                   "pos_pred": 0, "total": 0,
                                   "actual_pos": 0})

    for row, (pred, prob) in zip(data, predictions):
        g   = row[sensitive_attr]
        y   = row["true_label"]
        groups[g]["total"] += 1
        groups[g]["pos_pred"] += pred
        if y == 1:
            groups[g]["actual_pos"] += 1
            if pred == 1:
                groups[g]["TP"] += 1
            else:
                groups[g]["FN"] += 1
        else:
            if pred == 1:
                groups[g]["FP"] += 1
            else:
                groups[g]["TN"] += 1

    metrics = {}
    for g, s in groups.items():
        tpr = s["TP"] / max(s["actual_pos"], 1)         # Equal Opportunity
        fpr = s["FP"] / max(s["total"] - s["actual_pos"], 1)
        ppr = s["pos_pred"] / max(s["total"], 1)         # Demographic Parity Rate
        metrics[g] = {
            "total":              s["total"],
            "approval_rate":      round(ppr, 4),
            "true_positive_rate": round(tpr, 4),
            "false_positive_rate":round(fpr, 4),
        }

    # Disparate Impact = min_group_ppr / max_group_ppr  (≥0.8 is the 4/5 rule)
    pprs = [v["approval_rate"] for v in metrics.values()]
    di   = round(min(pprs) / max(pprs), 4) if max(pprs) > 0 else 1.0

    # Demographic Parity Difference
    dp_diff = round(max(pprs) - min(pprs), 4)

    # Equalized Odds Difference (average of TPR gap and FPR gap)
    tprs = [v["true_positive_rate"] for v in metrics.values()]
    fprs = [v["false_positive_rate"] for v in metrics.values()]
    eo_diff = round(
        (max(tprs) - min(tprs) + max(fprs) - min(fprs)) / 2, 4
    )

    return {
        "groups":            metrics,
        "disparate_impact":  di,
        "dp_difference":     dp_diff,
        "eo_difference":     eo_diff,
        "di_passes_4_5_rule": di >= 0.8,
    }


# ─────────────────────────────────────────────
#  SECTION 4 — BIAS MITIGATION (Re-weighting)
# ─────────────────────────────────────────────

def compute_reweighting(data, sensitive_attr, label_col):
    """
    Pre-processing mitigation: assign sample weights so each
    (group, label) cell has equal influence during training.
    Based on Calders & Verwer (2010) re-weighting strategy.
    """
    cell_counts = defaultdict(int)
    for row in data:
        cell_counts[(row[sensitive_attr], row[label_col])] += 1

    n = len(data)
    group_counts = defaultdict(int)
    label_counts = defaultdict(int)
    for row in data:
        group_counts[row[sensitive_attr]] += 1
        label_counts[row[label_col]] += 1

    weights = []
    for row in data:
        g = row[sensitive_attr]
        l = row[label_col]
        expected = (group_counts[g] / n) * (label_counts[l] / n) * n
        actual   = cell_counts[(g, l)]
        weights.append(expected / max(actual, 1))

    return weights


# ─────────────────────────────────────────────
#  SECTION 5 — EXPLAINABILITY
# ─────────────────────────────────────────────

def explain_decision(row, w, b, feature_cols, stats):
    """
    LIME-style local explanation: how much does each feature
    push the prediction toward approval or denial?
    Returns sorted list of (feature, contribution, direction).
    """
    x = featurize(row, feature_cols, stats)
    contributions = [(feature_cols[j], round(w[j] * x[j], 4))
                     for j in range(len(feature_cols))]
    contributions.sort(key=lambda t: abs(t[1]), reverse=True)

    explained = []
    for feat, contrib in contributions:
        direction = "↑ Approval" if contrib > 0 else "↓ Denial"
        explained.append({
            "feature":      feat,
            "contribution": abs(contrib),
            "direction":    direction,
            "raw_value":    row[feat],
        })
    return explained


# ─────────────────────────────────────────────
#  SECTION 6 — AUDIT TRAIL
# ─────────────────────────────────────────────

AUDIT_LOG = []

def audit_decision(applicant_id, pred, prob, explanation, model_type):
    entry = {
        "applicant_id": applicant_id,
        "model":        model_type,
        "decision":     "APPROVED" if pred == 1 else "DENIED",
        "confidence":   f"{prob*100:.1f}%",
        "top_reasons":  explanation[:3],
    }
    AUDIT_LOG.append(entry)
    return entry


# ─────────────────────────────────────────────
#  SECTION 7 — PRETTY PRINTING UTILITIES
# ─────────────────────────────────────────────

def banner(text):
    w = 62
    print("\n" + "═" * w)
    print(f"  {text}")
    print("═" * w)


def section(text):
    print(f"\n  ── {text} {'─'*(54 - len(text))}")


def print_fairness_report(label, result):
    print(f"\n  📊  Fairness Report — {label}")
    print(f"  {'Group':<12} {'Approval%':>10} {'TPR':>8} {'FPR':>8} {'N':>6}")
    print("  " + "─" * 48)
    for g, m in result["groups"].items():
        print(f"  {g:<12} {m['approval_rate']*100:>9.1f}%"
              f" {m['true_positive_rate']*100:>7.1f}%"
              f" {m['false_positive_rate']*100:>7.1f}%"
              f" {m['total']:>6}")

    di    = result["disparate_impact"]
    dp    = result["dp_difference"]
    eo    = result["eo_difference"]
    flag  = "✅ PASS" if result["di_passes_4_5_rule"] else "❌ FAIL"

    print(f"\n  Disparate Impact        : {di:.4f}  {flag} (threshold ≥ 0.80)")
    print(f"  Demographic Parity Diff : {dp:.4f}  {'✅' if dp < 0.1 else '⚠️ '} (threshold < 0.10)")
    print(f"  Equalized Odds Diff     : {eo:.4f}  {'✅' if eo < 0.1 else '⚠️ '} (threshold < 0.10)")


def print_explanation(explanation, decision, confidence):
    icon = "✅" if decision == "APPROVED" else "❌"
    print(f"\n  Decision : {icon} {decision}  (confidence {confidence})")
    print(f"  {'Feature':<15} {'Value':>10}  {'Weight':>8}  Direction")
    print("  " + "─" * 52)
    for e in explanation:
        print(f"  {e['feature']:<15} {str(e['raw_value']):>10}"
              f"  {e['contribution']:>8.4f}  {e['direction']}")


# ─────────────────────────────────────────────
#  SECTION 8 — MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    banner("UNBIASED AI DECISION SYSTEM  ·  H2S Open Innovation")

    # ── 8.1  Generate data ──────────────────────────────────
    section("Generating Synthetic Loan Dataset")
    data = generate_dataset(n=600)
    split = int(0.7 * len(data))
    train, test = data[:split], data[split:]
    print(f"  Total samples : {len(data)}  |  Train: {len(train)}  Test: {len(test)}")

    FEATURE_COLS = ["credit_score", "income", "debt_ratio", "age"]
    stats = normalize(train, FEATURE_COLS)

    # ── 8.2  Train BIASED model (learns from biased labels) ──
    section("Training Biased Model  (learns from discriminatory labels)")
    w_biased, b_biased = train_logistic(
        train, FEATURE_COLS, "label", stats, epochs=300
    )
    biased_preds = [predict(r, w_biased, b_biased, FEATURE_COLS, stats)
                    for r in test]
    print("  Biased model trained.")

    # ── 8.3  Train FAIR model (re-weighted, uses true labels) ─
    section("Training Fair Model  (re-weighted, merit-only labels)")
    rw = compute_reweighting(train, "ethnicity", "true_label")
    w_fair, b_fair = train_logistic(
        train, FEATURE_COLS, "true_label", stats, epochs=300, weights_override=rw
    )
    fair_preds = [predict(r, w_fair, b_fair, FEATURE_COLS, stats)
                  for r in test]
    print("  Fair model trained.")

    # ── 8.4  Fairness metrics ─────────────────────────────────
    banner("FAIRNESS ANALYSIS — by Ethnicity")
    biased_fairness = compute_fairness(test, biased_preds, "ethnicity")
    fair_fairness   = compute_fairness(test, fair_preds,   "ethnicity")
    print_fairness_report("BIASED Model", biased_fairness)
    print_fairness_report("FAIR   Model", fair_fairness)

    banner("FAIRNESS ANALYSIS — by Gender")
    biased_gender = compute_fairness(test, biased_preds, "gender")
    fair_gender   = compute_fairness(test, fair_preds,   "gender")
    print_fairness_report("BIASED Model", biased_gender)
    print_fairness_report("FAIR   Model", fair_gender)

    # ── 8.5  Per-applicant decision with explanation ──────────
    banner("SAMPLE APPLICANT DECISIONS WITH EXPLANATIONS")

    samples = [
        {"credit_score": 720, "income": 65000, "debt_ratio": 0.25,
         "age": 34, "gender": "Female", "ethnicity": "GroupB",
         "true_label": 1, "label": 0, "id": 9001},
        {"credit_score": 580, "income": 35000, "debt_ratio": 0.55,
         "age": 28, "gender": "Male",   "ethnicity": "GroupA",
         "true_label": 0, "label": 0, "id": 9002},
        {"credit_score": 760, "income": 90000, "debt_ratio": 0.18,
         "age": 45, "gender": "Female", "ethnicity": "GroupC",
         "true_label": 1, "label": 1, "id": 9003},
    ]

    for applicant in samples:
        print(f"\n  ┌─ Applicant #{applicant['id']} "
              f"| {applicant['gender']}, {applicant['ethnicity']} ─────────────")

        for model_tag, w, b in [("BIASED", w_biased, b_biased),
                                 ("FAIR",   w_fair,   b_fair)]:
            pred, prob = predict(applicant, w, b, FEATURE_COLS, stats)
            expl       = explain_decision(applicant, w, b, FEATURE_COLS, stats)
            entry      = audit_decision(applicant["id"], pred, prob,
                                        expl, model_tag)
            section(f"  {model_tag} Model")
            print_explanation(expl, entry["decision"], entry["confidence"])

    # ── 8.6  Audit log summary ───────────────────────────────
    banner("AUDIT TRAIL (last 6 entries)")
    for entry in AUDIT_LOG[-6:]:
        icon = "✅" if entry["decision"] == "APPROVED" else "❌"
        print(f"  [{entry['model']:<6}] Applicant {entry['applicant_id']:>5} "
              f"→ {icon} {entry['decision']:<8}  conf: {entry['confidence']}")

    # ── 8.7  Bias improvement summary ───────────────────────
    banner("BIAS IMPROVEMENT SUMMARY")
    b_di = biased_fairness["disparate_impact"]
    f_di = fair_fairness["disparate_impact"]
    b_dp = biased_fairness["dp_difference"]
    f_dp = fair_fairness["dp_difference"]
    b_eo = biased_fairness["eo_difference"]
    f_eo = fair_fairness["eo_difference"]

    def delta(before, after, lower_better=True):
        arrow = "▼" if (after < before) == lower_better else "▲"
        good  = (after < before) == lower_better
        tag   = "✅" if good else "❌"
        return f"{before:.4f} → {after:.4f}  {arrow}  {tag}"

    print(f"  Disparate Impact        : {delta(b_di, f_di, lower_better=False)}")
    print(f"  Demographic Parity Diff : {delta(b_dp, f_dp)}")
    print(f"  Equalized Odds Diff     : {delta(b_eo, f_eo)}")

    print("""
  ┌──────────────────────────────────────────────────────┐
  │  Key Takeaways                                       │
  │  • Biased labels encode systemic discrimination.     │
  │  • Re-weighting corrects group imbalance in training.│
  │  • Disparate Impact < 0.80 triggers regulatory risk. │
  │  • Explainability ensures every denial is justified. │
  │  • Audit trails enable external review & compliance. │
  └──────────────────────────────────────────────────────┘
""")


if __name__ == "__main__":
    main()
