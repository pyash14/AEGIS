import streamlit as st
import uuid
import sys
import os
sys.path.insert(0, '/Users/yash014p/AEGIS/aegis')

from database.database import create_tables, save_session, get_recent_food_logs
from llm.explainer import explain_decision
from nutrition.food_logger import log_meal
from llm.safety_monitor import check_safety

st.set_page_config(
    page_title="AEGIS — T1D Decision Support",
    page_icon="🩺",
    layout="wide"
)

create_tables()

if 'session_id' not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if 'decision_json' not in st.session_state:
    st.session_state.decision_json = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'alerts' not in st.session_state:
    st.session_state.alerts = []

st.title("🩺 AEGIS — AI-Driven Glucose Prediction & Insulin Dosing")
st.caption("Type 1 Diabetes Clinical Decision Support System | IEEE CCWC 2026 Extension")
st.divider()

with st.sidebar:
    st.header("⚙️ System Controls")
    st.caption(f"Session: `{st.session_state.session_id[:8]}...`")

    st.subheader("🍽️ Log a Meal")
    meal_input = st.text_area("Describe your meal", placeholder="e.g. bowl of oatmeal with banana")
    if st.button("📊 Analyze Meal", use_container_width=True):
        if meal_input.strip():
            with st.spinner("Analyzing meal nutrition..."):
                result = log_meal(meal_input)
            if result.get('success'):
                st.success("Meal logged!")
                st.metric("Carbs", f"{result['totals']['carbs_g']:.1f}g")
                st.metric("Calories", f"{result['totals']['calories']:.0f} kcal")
                st.caption(f"Confidence: {result['confidence']}")
            else:
                st.error(f"Error: {result.get('error')}")
        else:
            st.warning("Please enter a meal description.")

    st.divider()
    st.subheader("📋 Recent Meals")
    recent_meals = get_recent_food_logs(limit=5)
    if recent_meals:
        for meal in recent_meals:
            with st.expander(f"🍴 {meal['meal_description'][:30]}..."):
                st.write(f"**Carbs:** {meal['total_carbs_g']}g")
                st.write(f"**Protein:** {meal['total_protein_g']}g")
                st.write(f"**Fat:** {meal['total_fat_g']}g")
                st.write(f"**Calories:** {meal['total_calories']} kcal")
                st.caption(meal['timestamp'])
    else:
        st.info("No meals logged yet.")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🔬 Run Clinical Scenario")
    scenario = st.selectbox(
        "Select Scenario",
        ["A — Meal Spike (Hyperglycemia)", "B — Hypoglycemia", "C — Hyperglycemia Alert"]
    )

    if st.button("▶️ Run Inference", use_container_width=True, type="primary"):
        with st.spinner("Running AEGIS inference..."):
            if "A" in scenario:
                st.session_state.decision_json = {
                    "scenario": "A - Meal Spike",
                    "current_state": {
                        "current_glucose_mmol_L": 8.5,
                        "predicted_glucose_10min": 10.2,
                        "trend": "rising",
                        "risk_zone": "Hyper",
                        "thresholds_mmol_L": {"hypo": 3.9, "hyper": 10.0, "target": 6.0}
                    },
                    "recent_inputs": {
                        "carbs_last_90min_g": 45.0,
                        "bolus_last_90min_u": 0.0,
                        "active_insulin_iob_u": 0.2,
                        "active_carbs_cob_g": 30.0,
                        "activity": {"met": 1.2}
                    },
                    "decision": {
                        "recommended_dose_u": 2.5,
                        "decision_type": "Meal bolus + correction",
                        "dose_based_on_horizon": "10min",
                        "safety_constraints_applied": ["dose_cap_10U"]
                    }
                }
            elif "B" in scenario:
                st.session_state.decision_json = {
                    "scenario": "B - Hypoglycemia",
                    "current_state": {
                        "current_glucose_mmol_L": 3.2,
                        "predicted_glucose_10min": 3.0,
                        "trend": "falling",
                        "risk_zone": "Hypo",
                        "thresholds_mmol_L": {"hypo": 3.9, "hyper": 10.0, "target": 6.0}
                    },
                    "recent_inputs": {
                        "carbs_last_90min_g": 0.0,
                        "bolus_last_90min_u": 0.0,
                        "active_insulin_iob_u": 0.5,
                        "active_carbs_cob_g": 0.0,
                        "activity": {"met": 1.0}
                    },
                    "decision": {
                        "recommended_dose_u": 0.0,
                        "decision_type": "No insulin — hypoglycemia risk",
                        "dose_based_on_horizon": "10min",
                        "safety_constraints_applied": ["hypo_block"]
                    }
                }
            else:
                st.session_state.decision_json = {
                    "scenario": "C - Hyperglycemia Alert",
                    "current_state": {
                        "current_glucose_mmol_L": 14.0,
                        "predicted_glucose_10min": 14.8,
                        "trend": "rising",
                        "risk_zone": "Hyper",
                        "thresholds_mmol_L": {"hypo": 3.9, "hyper": 10.0, "target": 6.0}
                    },
                    "recent_inputs": {
                        "carbs_last_90min_g": 0.0,
                        "bolus_last_90min_u": 0.0,
                        "active_insulin_iob_u": 0.1,
                        "active_carbs_cob_g": 0.0,
                        "activity": {"met": 1.0}
                    },
                    "decision": {
                        "recommended_dose_u": 2.7,
                        "decision_type": "Correction dose",
                        "dose_based_on_horizon": "10min",
                        "safety_constraints_applied": ["dose_cap_10U"]
                    }
                }

            save_session(st.session_state.decision_json)
            st.session_state.alerts = check_safety(st.session_state.session_id)
            st.success("Inference complete!")

with col2:
    st.subheader("📊 Current Decision")
    if st.session_state.decision_json:
        d = st.session_state.decision_json
        cs = d['current_state']
        dec = d['decision']
        ri = d['recent_inputs']

        risk = cs.get('risk_zone', 'Unknown')
        color = "🔴" if risk == "Hypo" else "🟡" if risk == "Hyper" else "🟢"

        st.markdown(f"### {color} {risk} — {d.get('scenario')}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Current Glucose", f"{cs.get('current_glucose_mmol_L')} mmol/L")
        m2.metric("Predicted (10min)", f"{cs.get('predicted_glucose_10min')} mmol/L")
        m3.metric("Recommended Dose", f"{dec.get('recommended_dose_u')} U")

        m4, m5, m6 = st.columns(3)
        m4.metric("IOB", f"{ri.get('active_insulin_iob_u')} U")
        m5.metric("COB", f"{ri.get('active_carbs_cob_g')} g")
        m6.metric("Trend", cs.get('trend', '-'))

        st.caption(f"Decision: {dec.get('decision_type')}")
        st.caption(f"Safety checks: {dec.get('safety_constraints_applied')}")
    else:
        st.info("Run a scenario to see the decision.")

st.divider()

st.subheader("🚨 Safety Alerts")
if st.session_state.alerts:
    for alert in st.session_state.alerts:
        if alert['severity'] == "HIGH":
            st.error(f"**{alert['pattern']}** ({alert['severity']})\n\n{alert['explanation']}")
        else:
            st.warning(f"**{alert['pattern']}** ({alert['severity']})\n\n{alert['explanation']}")
else:
    st.success("✅ No active safety alerts.")

st.divider()

st.subheader("💬 Ask AEGIS")

for msg in st.session_state.chat_history:
    with st.chat_message(msg['role']):
        st.write(msg['content'])

user_input = st.chat_input("Ask about the current decision...")
if user_input:
    if not st.session_state.decision_json:
        st.warning("Please run a scenario first.")
    else:
        st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        with st.chat_message('user'):
            st.write(user_input)

        with st.chat_message('assistant'):
            with st.spinner("Thinking..."):
                response = explain_decision(
                    decision_json=st.session_state.decision_json,
                    user_question=user_input,
                    session_id=st.session_state.session_id
                )
            st.write(response)
            st.session_state.chat_history.append({'role': 'assistant', 'content': response})