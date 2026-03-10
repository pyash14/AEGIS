import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import datetime
from llm.claude_client import call_claude
from database.database import get_recent_sessions, get_recent_food_logs
from config.settings import (
    HYPO_COUNT_24H, NOCTURNAL_START, NOCTURNAL_END,
    STACKING_IOB_THRESHOLD, RAPID_DROP_RATE
)

SAFETY_SYSTEM_PROMPT = """You are a diabetes safety monitor assistant.
Your role is to explain detected safety patterns in plain language.
Rules:
- Never give direct medical advice or suggest specific doses
- Explain what the pattern means and why it is concerning
- Keep explanations under 100 words
- Always end with: "Please consult your healthcare provider."
"""

def _explain_alert(pattern: str, data: str) -> str:
    try:
        response = call_claude(
            system_prompt=SAFETY_SYSTEM_PROMPT,
            user_message=f'Explain this safety pattern in plain language:\nPattern: {pattern}\nData: {data}'
        )
        return response
    except Exception as e:
        return f'Alert: {pattern}. Please consult your healthcare provider.'

def check_safety(session_id: str = 'default') -> list:
    alerts = []
    sessions = get_recent_sessions(hours=24)

    if not sessions:
        return alerts

    # Pattern 1 — Recurring Hypo
    hypo_events = [s for s in sessions if s.get('risk_zone') == 'Hypo']
    if len(hypo_events) >= HYPO_COUNT_24H:
        explanation = _explain_alert(
            'Recurring Hypoglycemia',
            f'{len(hypo_events)} hypoglycemic events detected in the last 24 hours'
        )
        alerts.append({
            'pattern': 'Recurring Hypoglycemia',
            'severity': 'HIGH',
            'explanation': explanation,
            'triggered_at': datetime.now().isoformat()
        })

    # Pattern 2 — Nocturnal Hypo
    nocturnal_hypos = [
        s for s in sessions
        if s.get('risk_zone') == 'Hypo' and
        NOCTURNAL_START <= datetime.fromisoformat(s['timestamp']).hour < NOCTURNAL_END
    ]
    if nocturnal_hypos:
        explanation = _explain_alert(
            'Nocturnal Hypoglycemia',
            f'{len(nocturnal_hypos)} hypo events detected between midnight and 6am'
        )
        alerts.append({
            'pattern': 'Nocturnal Hypoglycemia',
            'severity': 'HIGH',
            'explanation': explanation,
            'triggered_at': datetime.now().isoformat()
        })

    # Pattern 3 — Post-Meal Hyperglycemia
    food_logs = get_recent_food_logs(limit=5)
    if food_logs:
        for meal in food_logs:
            meal_time = datetime.fromisoformat(meal['timestamp'])
            post_meal_hyper = [
                s for s in sessions
                if s.get('risk_zone') == 'Hyper' and
                0 <= (datetime.fromisoformat(s['timestamp']) - meal_time).total_seconds() <= 5400
            ]
            if post_meal_hyper:
                explanation = _explain_alert(
                    'Post-Meal Hyperglycemia',
                    f'Glucose above 10.0 mmol/L within 90 minutes of meal: {meal.get("meal_description")}'
                )
                alerts.append({
                    'pattern': 'Post-Meal Hyperglycemia',
                    'severity': 'MEDIUM',
                    'explanation': explanation,
                    'triggered_at': datetime.now().isoformat()
                })
                break

    # Pattern 4 — Insulin Stacking
    latest = sessions[-1] if sessions else None
    if latest:
        iob = latest.get('iob') or 0
        dose = latest.get('recommended_dose') or 0
        if iob > STACKING_IOB_THRESHOLD and dose > 0:
            explanation = _explain_alert(
                'Insulin Stacking Risk',
                f'IOB is {iob:.2f} U and a new dose of {dose:.2f} U was recommended'
            )
            alerts.append({
                'pattern': 'Insulin Stacking Risk',
                'severity': 'HIGH',
                'explanation': explanation,
                'triggered_at': datetime.now().isoformat()
            })

    # Pattern 5 — Rapid Glucose Drop
    if len(sessions) >= 3:
        recent_3 = sessions[-3:]
        try:
            glucose_values = [s['current_glucose'] for s in recent_3 if s.get('current_glucose')]
            times = [datetime.fromisoformat(s['timestamp']) for s in recent_3]
            if len(glucose_values) == 3:
                drop = glucose_values[0] - glucose_values[-1]
                minutes = (times[-1] - times[0]).total_seconds() / 60
                if minutes > 0:
                    rate = drop / minutes
                    if rate > RAPID_DROP_RATE:
                        explanation = _explain_alert(
                            'Rapid Glucose Drop',
                            f'Glucose dropped {drop:.2f} mmol/L in {minutes:.1f} minutes (rate: {rate:.2f} mmol/L/min)'
                        )
                        alerts.append({
                            'pattern': 'Rapid Glucose Drop',
                            'severity': 'MEDIUM',
                            'explanation': explanation,
                            'triggered_at': datetime.now().isoformat()
                        })
        except Exception:
            pass

    return alerts


if __name__ == '__main__':
    alerts = check_safety()
    if alerts:
        for alert in alerts:
            print(f"\n[{alert['severity']}] {alert['pattern']}")
            print(alert['explanation'])
    else:
        print('No safety alerts detected.')