import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json

def build_explainer_prompt(decision_json: dict, kb_chunks: list) -> str:
    current_state = decision_json.get('current_state', {})
    decision = decision_json.get('decision', {})
    recent_inputs = decision_json.get('recent_inputs', {})
    kb_text = '\n\n'.join(kb_chunks) if kb_chunks else 'No relevant KB chunks found.'

    return f"""
=== CURRENT AEGIS DECISION ===
Scenario: {decision_json.get('scenario', 'Unknown')}
Current Glucose: {current_state.get('current_glucose_mmol_L')} mmol/L
Predicted Glucose (10 min): {current_state.get('predicted_glucose_10min')} mmol/L
Trend: {current_state.get('trend')}
Risk Zone: {current_state.get('risk_zone')}
Recommended Dose: {decision.get('recommended_dose_u')} U
Decision Type: {decision.get('decision_type')}
Safety Constraints: {decision.get('safety_constraints_applied')}
IOB: {recent_inputs.get('active_insulin_iob_u')} U
COB: {recent_inputs.get('active_carbs_cob_g')} g
Carbs (last 90 min): {recent_inputs.get('carbs_last_90min_g')} g

=== RELEVANT MEDICAL KNOWLEDGE ===
{kb_text}
"""

def build_meal_parse_prompt(meal_text: str) -> str:
    return f"""Parse this meal into individual food items and quantities.
Respond with valid JSON only. No markdown, no backticks.
Format: {{"items": [{{"food_item": "name", "quantity_g": number}}]}}
Use common portion sizes if not specified.
Meal: {meal_text}"""

def build_safety_prompt(pattern: str, data: str) -> str:
    return f"""Explain this diabetes safety pattern in plain language.
Keep it under 100 words. Never suggest specific doses.
End with: "Please consult your healthcare provider."
Pattern: {pattern}
Data: {data}"""

if __name__ == '__main__':
    test = {'scenario': 'Test', 'current_state': {}, 'decision': {}, 'recent_inputs': {}}
    print(build_explainer_prompt(test, ['chunk1', 'chunk2']))
    print(build_meal_parse_prompt('bowl of oatmeal'))
    print(build_safety_prompt('Recurring Hypo', '3 events in 24h'))
