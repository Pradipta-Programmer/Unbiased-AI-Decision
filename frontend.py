"""
Unbiased AI Decision System - Streamlit Frontend
H2S Hackathon Prototype Frontend

Install: pip install streamlit pandas numpy plotly
Run: streamlit run unbiased_ai_frontend.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import random
import math
from collections import defaultdict
import json

# =============================================================================
# INTEGRATE YOUR PROTOTYPE CODE HERE (copy all functions from unbiased_ai_decision.py)
# =============================================================================

# Paste ALL functions: generate_dataset, sigmoid, normalize, featurize, train_logistic,
# predict, compute_fairness, compute_reweighting, explain_decision, audit_decision

# For brevity, assuming functions are defined above...
# (In real file, copy the full code from file:1 here)

AUDIT_LOG = []

# =============================================================================
# STREAMLIT APP
# =============================================================================

st.set_page_config(page_title="Unbiased AI Decision System", layout="wide")

# Sidebar
st.sidebar.title("🚀 Controls")
app_mode = st.sidebar.selectbox("Choose Mode", ["Demo Dataset", "Custom Applicant", "Fairness Analysis", "Audit Trail"])

# Global variables
@st.cache_data
def load_data(n=600, seed=42):
    return generate_dataset(n=n, seed=42)

FEATURE_COLS = ["credit_score", "income", "debt_ratio", "age"]

# Main title
st.title("🤖 Unbiased AI Decision System")
st.markdown("**H2S Open Innovation Hackathon Prototype** - Detect bias, ensure fairness in loan approvals[file:1]")

if app_mode == "Demo Dataset":
    st.header("📊 Full Pipeline Demo")
    
    if st.button("🔄 Run Complete Analysis", type="primary"):
        with st.spinner("Running biased & fair models..."):
            data = load_data()
            split = int(0.7 * len(data))
            train, test = data[:split], data[split:]
            stats = normalize(train, FEATURE_COLS)
            
            # Train models
            w_biased, b_biased = train_logistic(train, FEATURE_COLS, "label", stats, epochs=300)
            rw = compute_reweighting(train, "ethnicity", "true_label")
            w_fair, b_fair = train_logistic(train, FEATURE_COLS, "true_label", stats, epochs=300, weights_override=rw)
            
            biased_preds = [predict(r, w_biased, b_biased, FEATURE_COLS, stats) for r in test]
            fair_preds = [predict(r, w_fair, b_fair, FEATURE_COLS, stats) for r in test]
            
            # Fairness metrics
            biased_eth = compute_fairness(test, biased_preds, "ethnicity")
            fair_eth = compute_fairness(test, fair_preds, "ethnicity")
            biased_gen = compute_fairness(test, biased_preds, "gender")
            fair_gen = compute_fairness(test, fair_preds, "gender")
            
            # Store for display
            st.session_state.demo_results = {
                "biased_eth": biased_eth, "fair_eth": fair_eth,
                "biased_gen": biased_gen, "fair_gen": fair_gen,
                "test": test, "stats": stats,
                "w_biased": w_biased, "b_biased": b_biased,
                "w_fair": w_fair, "b_fair": b_fair
            }
    
    if "demo_results" in st.session_state:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Ethnicity Fairness")
            df_eth = pd.DataFrame({
                "Group": list(st.session_state.demo_results["fair_eth"]["groups"].keys()),
                "Approval Rate": [v["approval_rate"] for v in st.session_state.demo_results["fair_eth"]["groups"].values()],
                "DI": st.session_state.demo_results["fair_eth"]["disparate_impact"],
                "DP Diff": st.session_state.demo_results["fair_eth"]["dp_difference"]
            })
            fig_eth = px.bar(df_eth, x="Group", y="Approval Rate", title="Fair Model Approval Rates")
            st.plotly_chart(fig_eth)
            st.metric("Disparate Impact", f"{st.session_state.demo_results['fair_eth']['disparate_impact']:.3f}", 
                     "✅ PASS" if st.session_state.demo_results['fair_eth']['di_passes_4_5_rule'] else "❌ FAIL")
        
        with col2:
            st.subheader("Gender Fairness")
            df_gen = pd.DataFrame({
                "Group": list(st.session_state.demo_results["fair_gen"]["groups"].keys()),
                "Approval Rate": [v["approval_rate"] for v in st.session_state.demo_results["fair_gen"]["groups"].values()]
            })
            fig_gen = px.bar(df_gen, x="Group", y="Approval Rate")
            st.plotly_chart(fig_gen)

elif app_mode == "Custom Applicant":
    st.header("👤 Test Single Applicant")
    
    col1, col2, col3, col4 = st.columns(4)
    credit_score = col1.slider("Credit Score", 300, 850, 680)
    income = col2.slider("Income ($)", 15000, 150000, 65000)
    debt_ratio = col3.slider("Debt Ratio", 0.1, 0.7, 0.25)
    age = col4.slider("Age", 18, 70, 34)
    
    gender = st.selectbox("Gender", ["Male", "Female"])
    ethnicity = st.selectbox("Ethnicity", ["GroupA", "GroupB", "GroupC"])
    
    if st.button("🔮 Predict with Both Models"):
        applicant = {
            "id": 9999, "credit_score": credit_score, "income": income,
            "debt_ratio": debt_ratio, "age": age, "gender": gender,
            "ethnicity": ethnicity
        }
        
        if "demo_results" in st.session_state:
            stats = st.session_state.demo_results["stats"]
            
            # Biased prediction
            pred_b, prob_b = predict(applicant, st.session_state.demo_results["w_biased"], 
                                   st.session_state.demo_results["b_biased"], FEATURE_COLS, stats)
            expl_b = explain_decision(applicant, st.session_state.demo_results["w_biased"], 
                                    st.session_state.demo_results["b_biased"], FEATURE_COLS, stats)
            audit_decision(applicant["id"], pred_b, prob_b, expl_b, "BIASED")
            
            # Fair prediction
            pred_f, prob_f = predict(applicant, st.session_state.demo_results["w_fair"], 
                                   st.session_state.demo_results["b_fair"], FEATURE_COLS, stats)
            expl_f = explain_decision(applicant, st.session_state.demo_results["w_fair"], 
                                    st.session_state.demo_results["b_fair"], FEATURE_COLS, stats)
            audit_decision(applicant["id"], pred_f, prob_f, expl_f, "FAIR")
            
            # Display results
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Biased Model")
                decision_b = "✅ APPROVED" if pred_b == 1 else "❌ DENIED"
                st.metric("Decision", decision_b, f"{prob_b:.1%}")
                st.subheader("Top Reasons")
                for e in expl_b[:3]:
                    st.write(f"**{e['feature']}** ({e['raw_value']}): {e['direction']} | weight: {e['contribution']:.3f}")
            
            with col2:
                st.subheader("Fair Model")
                decision_f = "✅ APPROVED" if pred_f == 1 else "❌ DENIED"
                st.metric("Decision", decision_f, f"{prob_f:.1%}")
                st.subheader("Top Reasons")
                for e in expl_f[:3]:
                    st.write(f"**{e['feature']}** ({e['raw_value']}): {e['direction']} | weight: {e['contribution']:.3f}")

elif app_mode == "Fairness Analysis":
    st.header("⚖️ Detailed Fairness Metrics")
    if "demo_results" in st.session_state:
        metrics_df = pd.DataFrame({
            "Attribute": ["Ethnicity", "Ethnicity", "Gender", "Gender"],
            "Model": ["Biased", "Fair", "Biased", "Fair"],
            "Disparate Impact": [
                st.session_state.demo_results["biased_eth"]["disparate_impact"],
                st.session_state.demo_results["fair_eth"]["disparate_impact"],
                st.session_state.demo_results["biased_gen"]["disparate_impact"],
                st.session_state.demo_results["fair_gen"]["disparate_impact"]
            ],
            "DP Difference": [
                st.session_state.demo_results["biased_eth"]["dp_difference"],
                st.session_state.demo_results["fair_eth"]["dp_difference"],
                st.session_state.demo_results["biased_gen"]["dp_difference"],
                st.session_state.demo_results["fair_gen"]["dp_difference"]
            ]
        })
        fig_metrics = px.bar(metrics_df, x="Model", y="Disparate Impact", color="Attribute", 
                           title="Fairness Improvement Across Models[file:1]")
        st.plotly_chart(fig_metrics)
    else:
        st.info("Run Demo Dataset first to generate metrics.")

elif app_mode == "Audit Trail":
    st.header("📋 Decision Audit Log")
    if AUDIT_LOG:
        df_audit = pd.DataFrame(AUDIT_LOG[-20:])
        st.dataframe(df_audit)
        
        fig_pie = px.pie(df_audit, names="decision", title="Recent Decisions")
        st.plotly_chart(fig_pie)
    else:
        st.info("Make some predictions first!")

# Footer
st.markdown("---")
st.markdown("*Built with Streamlit for the H2S Hackathon. Source: [unbiased_ai_decision.py][file:1] | Inspired by fairness dashboards[web:3][web:4][web:5]*")
