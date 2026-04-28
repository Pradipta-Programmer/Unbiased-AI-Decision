import streamlit as st
import pandas as pd
import plotly.express as px

# ✅ Import backend properly
from unbiased_ai_decision import (
    generate_dataset,
    normalize,
    train_logistic,
    predict,
    compute_fairness,
    compute_reweighting,
    explain_decision,
    audit_decision
)

# Global audit log
AUDIT_LOG = []

st.set_page_config(page_title="Unbiased AI Decision System", layout="wide")

# Sidebar
st.sidebar.title("🚀 Controls")
app_mode = st.sidebar.selectbox(
    "Choose Mode",
    ["Demo Dataset", "Custom Applicant", "Fairness Analysis", "Audit Trail"]
)

FEATURE_COLS = ["credit_score", "income", "debt_ratio", "age"]

# ✅ Fixed cache function
@st.cache_data
def load_data(n=600, seed=42):
    return generate_dataset(n=n, seed=seed)

# ✅ Ensure models always available
def ensure_models():
    if "demo_results" not in st.session_state:
        data = load_data()
        split = int(0.7 * len(data))
        train = data[:split]
        test = data[split:]

        stats = normalize(train, FEATURE_COLS)

        # Train models
        w_biased, b_biased = train_logistic(train, FEATURE_COLS, "label", stats)
        rw = compute_reweighting(train, "ethnicity", "true_label")
        w_fair, b_fair = train_logistic(train, FEATURE_COLS, "true_label", stats, weights_override=rw)

        st.session_state.demo_results = {
            "train": train,
            "test": test,
            "stats": stats,
            "w_biased": w_biased,
            "b_biased": b_biased,
            "w_fair": w_fair,
            "b_fair": b_fair
        }

# Title
st.title("🤖 Unbiased AI Decision System")
st.markdown("Detect bias and ensure fairness in AI-based loan approvals.")

# =============================================================================
# DEMO DATASET
# =============================================================================
if app_mode == "Demo Dataset":
    st.header("📊 Full Pipeline Demo")

    if st.button("🔄 Run Complete Analysis", type="primary"):
        with st.spinner("Training models..."):
            ensure_models()
            data = st.session_state.demo_results["test"]

            stats = st.session_state.demo_results["stats"]
            w_b = st.session_state.demo_results["w_biased"]
            b_b = st.session_state.demo_results["b_biased"]
            w_f = st.session_state.demo_results["w_fair"]
            b_f = st.session_state.demo_results["b_fair"]

            biased_preds = [predict(r, w_b, b_b, FEATURE_COLS, stats) for r in data]
            fair_preds = [predict(r, w_f, b_f, FEATURE_COLS, stats) for r in data]

            biased_eth = compute_fairness(data, biased_preds, "ethnicity")
            fair_eth = compute_fairness(data, fair_preds, "ethnicity")

            st.session_state.metrics = {
                "biased_eth": biased_eth,
                "fair_eth": fair_eth
            }

    if "metrics" in st.session_state:
        st.subheader("Ethnicity Fairness Comparison")

        df = pd.DataFrame({
            "Group": list(st.session_state.metrics["fair_eth"]["groups"].keys()),
            "Approval Rate": [
                v["approval_rate"]
                for v in st.session_state.metrics["fair_eth"]["groups"].values()
            ]
        })

        fig = px.bar(df, x="Group", y="Approval Rate", title="Fair Model Approval Rates")
        st.plotly_chart(fig)

        di = st.session_state.metrics["fair_eth"]["disparate_impact"]
        st.metric("Disparate Impact", f"{di:.3f}",
                  "✅ PASS" if di >= 0.8 else "❌ FAIL")

# =============================================================================
# CUSTOM APPLICANT
# =============================================================================
elif app_mode == "Custom Applicant":
    st.header("👤 Test Single Applicant")

    ensure_models()

    col1, col2, col3, col4 = st.columns(4)
    credit = col1.slider("Credit Score", 300, 850, 680)
    income = col2.slider("Income", 15000, 150000, 60000)
    debt = col3.slider("Debt Ratio", 0.1, 0.7, 0.3)
    age = col4.slider("Age", 18, 70, 30)

    gender = st.selectbox("Gender", ["Male", "Female"])
    ethnicity = st.selectbox("Ethnicity", ["GroupA", "GroupB", "GroupC"])

    if st.button("🔮 Predict"):
        applicant = {
            "id": 9999,
            "credit_score": credit,
            "income": income,
            "debt_ratio": debt,
            "age": age,
            "gender": gender,
            "ethnicity": ethnicity
        }

        stats = st.session_state.demo_results["stats"]

        # Biased model
        pred_b, prob_b = predict(
            applicant,
            st.session_state.demo_results["w_biased"],
            st.session_state.demo_results["b_biased"],
            FEATURE_COLS,
            stats
        )
        expl_b = explain_decision(
            applicant,
            st.session_state.demo_results["w_biased"],
            st.session_state.demo_results["b_biased"],
            FEATURE_COLS,
            stats
        )

        # Fair model
        pred_f, prob_f = predict(
            applicant,
            st.session_state.demo_results["w_fair"],
            st.session_state.demo_results["b_fair"],
            FEATURE_COLS,
            stats
        )
        expl_f = explain_decision(
            applicant,
            st.session_state.demo_results["w_fair"],
            st.session_state.demo_results["b_fair"],
            FEATURE_COLS,
            stats
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Biased Model")
            st.metric("Decision", "APPROVED" if pred_b else "DENIED", f"{prob_b:.1%}")
            for e in expl_b[:3]:
                st.write(f"{e['feature']}: {e['direction']} ({e['contribution']:.3f})")

        with col2:
            st.subheader("Fair Model")
            st.metric("Decision", "APPROVED" if pred_f else "DENIED", f"{prob_f:.1%}")
            for e in expl_f[:3]:
                st.write(f"{e['feature']}: {e['direction']} ({e['contribution']:.3f})")

# =============================================================================
# FAIRNESS ANALYSIS
# =============================================================================
elif app_mode == "Fairness Analysis":
    st.header("⚖️ Fairness Metrics")

    if "metrics" in st.session_state:
        df = pd.DataFrame({
            "Model": ["Biased", "Fair"],
            "Disparate Impact": [
                st.session_state.metrics["biased_eth"]["disparate_impact"],
                st.session_state.metrics["fair_eth"]["disparate_impact"]
            ]
        })

        fig = px.bar(df, x="Model", y="Disparate Impact", title="Bias Reduction")
        st.plotly_chart(fig)
    else:
        st.info("Run Demo Dataset first.")

# =============================================================================
# AUDIT TRAIL
# =============================================================================
elif app_mode == "Audit Trail":
    st.header("📋 Audit Log")

    if AUDIT_LOG:
        df = pd.DataFrame(AUDIT_LOG)
        st.dataframe(df)
    else:
        st.info("No decisions logged yet.")

# Footer
st.markdown("---")
st.markdown("Built with Streamlit | H2S Hackathon Prototype")
