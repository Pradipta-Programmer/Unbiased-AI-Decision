"""
UNBIASED AI DECISION SYSTEM — Improved Version
Enhanced for stability, realism, and better fairness evaluation
"""

import random
import math
from collections import defaultdict

# ─────────────────────────────────────────────
# DATASET GENERATION (More realistic bias)
# ─────────────────────────────────────────────

def generate_dataset(n=600, seed=42):
    random.seed(seed)
    data = []

    genders = ["Male", "Female"]
    ethnicities = ["GroupA", "GroupB", "GroupC"]

    bias_factor = {"GroupA": 1.0, "GroupB": 0.85, "GroupC": 0.95}

    for i in range(n):
        gender = random.choice(genders)
        ethnicity = random.choice(ethnicities)

        income = int(random.gauss(55000 * bias_factor[ethnicity], 12000))
        income = max(15000, income)

        credit_score = max(300, min(850, int(random.gauss(680, 80))))
        debt_ratio = round(random.uniform(0.1, 0.7), 2)
        age = max(18, min(70, int(random.gauss(38, 10))))

        merit_score = (
            (credit_score - 300) / 550 * 0.5 +
            min(income, 100000) / 100000 * 0.3 +
            (1 - debt_ratio) * 0.2
        )

        true_label = 1 if merit_score > 0.45 else 0

        biased_label = true_label
        if ethnicity == "GroupB" and true_label == 1:
            if random.random() < 0.25:
                biased_label = 0

        data.append({
            "id": i,
            "credit_score": credit_score,
            "income": income,
            "debt_ratio": debt_ratio,
            "age": age,
            "gender": gender,
            "ethnicity": ethnicity,
            "true_label": true_label,
            "label": biased_label
        })

    return data


# ─────────────────────────────────────────────
# MODEL (Improved Logistic Regression)
# ─────────────────────────────────────────────

def sigmoid(z):
    return 1.0 / (1.0 + math.exp(-max(-500, min(500, z))))


def normalize(data, feature_cols):
    stats = {}
    for col in feature_cols:
        vals = [row[col] for row in data]
        mu = sum(vals) / len(vals)
        std = math.sqrt(sum((v - mu) ** 2 for v in vals) / len(vals)) or 1
        stats[col] = (mu, std)
    return stats


def featurize(row, feature_cols, stats):
    return [(row[c] - stats[c][0]) / stats[c][1] for c in feature_cols]


def train_logistic(data, feature_cols, label_col, stats,
                   epochs=300, lr=0.05, reg_lambda=0.01, weights_override=None):

    n_feat = len(feature_cols)
    w = [0.0] * n_feat
    b = 0.0
    weights = weights_override or [1.0] * len(data)

    for _ in range(epochs):
        dw = [0.0] * n_feat
        db = 0.0
        total_w = sum(weights)

        for row, wt in zip(data, weights):
            x = featurize(row, feature_cols, stats)
            y = row[label_col]

            z = sum(wi * xi for wi, xi in zip(w, x)) + b
            pred = sigmoid(z)

            err = (pred - y) * wt / total_w

            for j in range(n_feat):
                dw[j] += err * x[j] + reg_lambda * w[j]

            db += err

        for j in range(n_feat):
            w[j] -= lr * dw[j]
        b -= lr * db

    return w, b


def predict(row, w, b, feature_cols, stats, threshold=0.5):
    x = featurize(row, feature_cols, stats)
    prob = sigmoid(sum(wi * xi for wi, xi in zip(w, x)) + b)
    return int(prob >= threshold), prob


# ─────────────────────────────────────────────
# FAIRNESS METRICS + ACCURACY
# ─────────────────────────────────────────────

def compute_accuracy(data, predictions):
    correct = sum(1 for row, (pred, _) in zip(data, predictions) if pred == row["true_label"])
    return correct / len(data)


def compute_fairness(data, predictions, sensitive_attr):
    groups = defaultdict(lambda: {"pos_pred": 0, "total": 0})

    for row, (pred, _) in zip(data, predictions):
        g = row[sensitive_attr]
        groups[g]["total"] += 1
        groups[g]["pos_pred"] += pred

    rates = {g: v["pos_pred"] / v["total"] for g, v in groups.items()}
    di = min(rates.values()) / max(rates.values())

    return {"rates": rates, "disparate_impact": round(di, 4)}


# ─────────────────────────────────────────────
# REWEIGHTING (Improved)
# ─────────────────────────────────────────────

def compute_reweighting(data, sensitive_attr, label_col):
    cell_counts = defaultdict(int)
    group_counts = defaultdict(int)
    label_counts = defaultdict(int)

    for row in data:
        g, l = row[sensitive_attr], row[label_col]
        cell_counts[(g, l)] += 1
        group_counts[g] += 1
        label_counts[l] += 1

    n = len(data)
    weights = []

    for row in data:
        g, l = row[sensitive_attr], row[label_col]
        expected = (group_counts[g] * label_counts[l]) / n
        actual = cell_counts[(g, l)]
        weights.append(expected / max(actual, 1))

    total = sum(weights)
    return [w * n / total for w in weights]


# ─────────────────────────────────────────────
# EXPLAINABILITY (Improved)
# ─────────────────────────────────────────────

def explain_decision(row, w, b, feature_cols, stats):
    x = featurize(row, feature_cols, stats)
    raw = [w[j] * x[j] for j in range(len(w))]
    total = sum(abs(r) for r in raw) or 1

    explanation = []
    for j, feat in enumerate(feature_cols):
        contrib = raw[j] / total
        explanation.append({
            "feature": feat,
            "contribution": round(abs(contrib), 4),
            "direction": "↑ Approval" if contrib > 0 else "↓ Denial"
        })

    return sorted(explanation, key=lambda x: x["contribution"], reverse=True)


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────

def main():
    print("\n=== UNBIASED AI SYSTEM (IMPROVED) ===")

    data = generate_dataset()
    split = int(0.7 * len(data))
    train, test = data[:split], data[split:]

    FEATURES = ["credit_score", "income", "debt_ratio", "age"]
    stats = normalize(train, FEATURES)

    # Biased model
    w_b, b_b = train_logistic(train, FEATURES, "label", stats)
    biased_preds = [predict(r, w_b, b_b, FEATURES, stats) for r in test]

    # Fair model
    rw = compute_reweighting(train, "ethnicity", "true_label")
    w_f, b_f = train_logistic(train, FEATURES, "true_label", stats, weights_override=rw)
    fair_preds = [predict(r, w_f, b_f, FEATURES, stats) for r in test]

    # Metrics
    acc_b = compute_accuracy(test, biased_preds)
    acc_f = compute_accuracy(test, fair_preds)

    fair_b = compute_fairness(test, biased_preds, "ethnicity")
    fair_f = compute_fairness(test, fair_preds, "ethnicity")

    print("\nAccuracy:")
    print(f"Biased Model: {acc_b:.3f}")
    print(f"Fair Model:   {acc_f:.3f}")

    print("\nDisparate Impact:")
    print(f"Biased: {fair_b['disparate_impact']}")
    print(f"Fair:   {fair_f['disparate_impact']}")

    # Sample explanation
    sample = test[0]
    print("\nSample Explanation (Fair Model):")
    exp = explain_decision(sample, w_f, b_f, FEATURES, stats)
    for e in exp[:3]:
        print(e)

AUDIT_LOG = []

def audit_decision(applicant_id, pred, prob, explanation, model_type):
    entry = {
        "applicant_id": applicant_id,
        "model": model_type,
        "decision": "APPROVED" if pred == 1 else "DENIED",
        "confidence": f"{prob*100:.1f}%",
        "top_reasons": explanation[:3],
    }
    AUDIT_LOG.append(entry)
    return entry

if __name__ == "__main__":
    main()
