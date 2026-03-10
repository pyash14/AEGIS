import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from llm.claude_client import call_claude
from rag.retriever import retrieve_medical
from database.database import get_chat_history, save_chat_message

SYSTEM_PROMPT = """You are T1D-Explain AI, a clinical decision support assistant for Type 1 Diabetes management.

Your role is to explain insulin dosing decisions made by the AEGIS system. You do NOT make new dosing decisions.

Rules you must always follow:
- Only reference values present in the decision JSON provided to you
- Never give medical advice or suggest new insulin doses
- Always ground explanations in ADA thresholds: hypoglycemia below 3.9 mmol/L, hyperglycemia above 10.0 mmol/L
- For hypoglycemia situations: you may mention the 15g fast-acting carb guideline as informational only
- If asked something off-topic: politely redirect to the system data
- Never speculate beyond what the data shows
- End every response with: "Let me know if you need more details."

You have access to:
1. The current AEGIS decision JSON (glucose, risk zone, recommended dose, IOB, COB)
2. Relevant medical knowledge retrieved from clinical guidelines
3. Recent conversation history for context
"""

def explain_decision(decision_json: dict, user_question: str, session_id: str = 'default') -> str:
    try:
        # Step 1 - Retrieve relevant medical KB chunks
        kb_chunks = retrieve_medical(user_question, top_k=4)
        kb_text = '\n\n'.join(kb_chunks) if kb_chunks else 'No relevant KB chunks found.'

        # Step 2 - Load last 5 chat turns from SQLite
        history = get_chat_history(session_id, limit=5)

        # Step 3 - Build user message with context
        current_state = decision_json.get('current_state', {})
        decision = decision_json.get('decision', {})
        recent_inputs = decision_json.get('recent_inputs', {})

        context = f"""
=== CURRENT AEGIS DECISION ===
Scenario: {decision_json.get('scenario', 'Unknown')}
Current Glucose: {current_state.get('current_glucose_mmol_L')} mmol/L
Predicted Glucose (10 min): {current_state.get('predicted_glucose_10min')} mmol/L
Trend: {current_state.get('trend')}
Risk Zone: {current_state.get('risk_zone')}
Recommended Dose: {decision.get('recommended_dose_u')} U
Decision Type: {decision.get('decision_type')}
Safety Constraints Applied: {decision.get('safety_constraints_applied')}
IOB: {recent_inputs.get('active_insulin_iob_u')} U
COB: {recent_inputs.get('active_carbs_cob_g')} g
Carbs (last 90 min): {recent_inputs.get('carbs_last_90min_g')} g

=== RELEVANT MEDICAL KNOWLEDGE ===
{kb_text}

=== USER QUESTION ===
{user_question}
"""

        # Step 4 - Call Claude
        response = call_claude(
            system_prompt=SYSTEM_PROMPT,
            user_message=context,
            history=history
        )

        # Step 5 - Save to SQLite
        save_chat_message(session_id, 'user', user_question)
        save_chat_message(session_id, 'assistant', response)

        return response

    except Exception as e:
        return f'Explainer error: {str(e)}'


if __name__ == '__main__':
    # Test with a sample decision JSON
    test_decision = {
        'scenario': 'B - Hypoglycemia',
        'current_state': {
            'current_glucose_mmol_L': 3.2,
            'predicted_glucose_10min': 3.0,
            'trend': 'falling',
            'risk_zone': 'Hypo',
            'thresholds_mmol_L': {'hypo': 3.9, 'hyper': 10.0, 'target': 6.0}
        },
        'recent_inputs': {
            'carbs_last_90min_g': 0.0,
            'bolus_last_90min_u': 0.0,
            'active_insulin_iob_u': 0.5,
            'active_carbs_cob_g': 0.0
        },
        'decision': {
            'recommended_dose_u': 0.0,
            'decision_type': 'No insulin — hypoglycemia risk',
            'safety_constraints_applied': ['hypo_block']
        }
    }

    answer = explain_decision(
        decision_json=test_decision,
        user_question='Why was insulin withheld?',
        session_id='test_session'
    )
    print(answer)