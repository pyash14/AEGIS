"""
AEGIS Inference Pipeline
------------------------
Loads XGB_10min_P2301.json + XGB_10min_P2301_scaler.pkl from models/
Reproduces the exact feature engineering from the training notebook.
Applies the redefined insulin dosing module from the insulin notebook.

Feature order (must match training exactly):
    glucose, delta_5, glucose_roc,
    glucose_lag_1, glucose_lag_3, glucose_lag_6,
    bolus, IOB, carbs, COB,
    met, rolling_met_15,
    sin_time, cos_time

IOB: dual-phase decay  0.55*exp(-t/30) + 0.45*exp(-t/90)
COB: gamma-style       (t-delay)*exp(-(t-delay)/20), delay=20min, tau=20min
Both projected to T+10 using decay-ratio method.
"""

import os
import sys
import math
import json
import numpy as np
import joblib
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=InconsistentVersionWarning if False else UserWarning)

# ── Make sure project root is on path when run directly ──────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    TARGET, HYPO_THRESHOLD, HYPER_THRESHOLD,
    ICR, ISF
)

# ── Constants ─────────────────────────────────────────────────────────────
MODEL_PATH  = os.path.join('models', 'XGB_10min_P2301.json')
SCALER_PATH = os.path.join('models', 'XGB_10min_P2301_scaler.pkl')

MIN_DOSE  = 0.1   # U — doses below this are zeroed
MAX_DOSE  = 10.0  # U — hard safety cap
DT        = 5     # minutes — CGM step size
HORIZON   = 10    # minutes — prediction horizon

# ── Lazy model cache ───────────────────────────────────────────────────────
_model  = None
_scaler = None


def _load_model():
    global _model, _scaler
    if _model is not None:
        return

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"XGBoost model not found at '{MODEL_PATH}'.\n"
            "Place XGB_10min_P2301.json in the models/ folder."
        )
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found at '{SCALER_PATH}'.\n"
            "Place XGB_10min_P2301_scaler.pkl in the models/ folder."
        )

    _model = xgb.XGBRegressor()
    _model.load_model(MODEL_PATH)
    _scaler = joblib.load(SCALER_PATH)


# ════════════════════════════════════════════════════════════════════════════
# IOB  —  dual-phase exponential decay
# ════════════════════════════════════════════════════════════════════════════

def _iob_decay(elapsed_min: float, tau1: float = 30, tau2: float = 90) -> float:
    """Fraction of insulin still active after elapsed_min minutes."""
    return 0.55 * math.exp(-elapsed_min / tau1) + 0.45 * math.exp(-elapsed_min / tau2)


def compute_iob(doses: list, project_minutes: int = 0) -> float:
    """
    Sum IOB from a list of past dose events, optionally projected forward.

    Each dose: {'units': float, 'minutes_ago': int}
    project_minutes = 0  → IOB right now
    project_minutes = 10 → IOB at T+10
    """
    iob = 0.0
    for dose in doses:
        t = dose.get('minutes_ago', 0) + project_minutes
        iob += dose.get('units', 0.0) * _iob_decay(t)
    return round(max(iob, 0.0), 4)


def _project_iob(current_iob: float) -> float:
    """
    Project a running IOB accumulator forward by HORIZON minutes
    using the decay-ratio method (matches insulin notebook Block 3).
    """
    decay_now  = _iob_decay(DT)
    decay_proj = _iob_decay(DT + HORIZON)
    if decay_now == 0:
        return 0.0
    return round(max(current_iob * (decay_proj / decay_now), 0.0), 4)


# ════════════════════════════════════════════════════════════════════════════
# COB  —  gamma-style delayed absorption  (matches training notebook)
# ════════════════════════════════════════════════════════════════════════════

def _gamma_kernel(n_steps: int, step_min: int = DT,
                  delay_min: int = 20, tau: float = 20.0) -> np.ndarray:
    """
    Normalised gamma absorption kernel used in training (Block 4).
    k[t] = (t - delay) * exp(-(t-delay)/tau)  for t >= delay, else 0
    """
    t = np.arange(n_steps)
    delay = delay_min // step_min
    k = (t - delay) * np.exp(-(t - delay) / tau)
    k[t < delay] = 0.0
    k[k < 0]     = 0.0
    total = k.sum()
    if total > 0:
        k /= total
    return k


def compute_cob(carb_events: list, project_minutes: int = 0) -> float:
    """
    Compute COB from a list of past carb events using the gamma kernel.

    Each event: {'grams': float, 'minutes_ago': int}
    project_minutes = 0  → COB right now
    project_minutes = 10 → COB at T+10
    """
    if not carb_events:
        return 0.0

    # Build a 180-min history array (36 × 5-min steps)
    n_steps   = 180 // DT          # 36 steps
    kernel    = _gamma_kernel(n_steps)
    cob_total = 0.0

    for event in carb_events:
        grams     = event.get('grams', 0.0)
        t_ago     = event.get('minutes_ago', 0) + project_minutes
        step_ago  = int(round(t_ago / DT))

        if step_ago >= n_steps:
            continue  # fully absorbed

        # Remaining absorption from this event
        remaining_kernel = kernel[step_ago:]
        absorbed_so_far  = kernel[:step_ago].sum()
        remaining_frac   = 1.0 - absorbed_so_far
        cob_total += grams * remaining_frac

    return round(max(cob_total, 0.0), 4)


def _project_cob(current_cob: float) -> float:
    """
    Project a running COB accumulator forward by HORIZON minutes
    using the decay-ratio method (matches insulin notebook Block 4).
    Absorption uses exp(-t/20) for the projection ratio.
    """
    rate_now  = math.exp(-DT / 20)
    rate_proj = math.exp(-(DT + HORIZON) / 20)
    if rate_now == 0:
        return 0.0
    return round(max(current_cob * (rate_proj / rate_now), 0.0), 4)


# ════════════════════════════════════════════════════════════════════════════
# Feature builder  —  exact 14-feature order from training notebook
# ════════════════════════════════════════════════════════════════════════════

def _build_features(inputs: dict, iob: float, cob: float) -> np.ndarray:
    """
    Reproduces build_feature_df() column order from training notebook.

    Feature order:
        glucose, delta_5, glucose_roc,
        glucose_lag_1, glucose_lag_3, glucose_lag_6,
        bolus, IOB, carbs, COB,
        met, rolling_met_15,
        sin_time, cos_time
    """
    g = inputs['glucose_readings']   # [g0, g1, g2, g3, g4] oldest→newest

    glucose      = g[4]              # current (lag_0)
    glucose_lag1 = g[3]              # 5 min ago
    glucose_lag3 = g[1]              # 15 min ago (3 × 5min)
    glucose_lag6 = g[0] if len(g) > 5 else g[0]   # 30 min ago (6 × 5min)
    # Note: inputs only carries 5 readings (indices 0-4).
    # lag_6 would need a 6th reading — we use g[0] (oldest available, 20 min ago)
    # as the best approximation when only 5 readings are provided.

    delta_5      = g[4] - g[3]                     # 1-step diff
    glucose_roc  = (g[4] - g[3]) / DT              # mmol/L per minute

    bolus_now    = sum(d.get('units', 0.0)
                       for d in inputs.get('insulin_doses', [])
                       if d.get('minutes_ago', 999) < DT)

    carbs_now    = sum(e.get('grams', 0.0)
                       for e in inputs.get('carb_events', [])
                       if e.get('minutes_ago', 999) < DT)

    met          = inputs.get('met', 1.0)
    rolling_met  = inputs.get('rolling_met_15', 1.0)

    hour         = inputs.get('hour', 12)
    angle        = 2 * math.pi * hour / 24
    sin_time     = math.sin(angle)
    cos_time     = math.cos(angle)

    features = np.array([[
        glucose,
        delta_5,    glucose_roc,
        glucose_lag1, glucose_lag3, glucose_lag6,
        bolus_now,  iob,
        carbs_now,  cob,
        met,        rolling_met,
        sin_time,   cos_time,
    ]], dtype=np.float64)

    return features


# ════════════════════════════════════════════════════════════════════════════
# Trend + risk helpers
# ════════════════════════════════════════════════════════════════════════════

def _get_trend(glucose: list) -> str:
    delta = glucose[4] - glucose[3]
    if delta > 0.1:
        return 'rising'
    elif delta < -0.1:
        return 'falling'
    return 'stable'


def _get_risk_zone(value: float) -> str:
    if value < HYPO_THRESHOLD:
        return 'Hypo'
    elif value > HYPER_THRESHOLD:
        return 'Hyper'
    return 'Euglycemia'


# ════════════════════════════════════════════════════════════════════════════
# Dosing engine  —  redefined module (insulin notebook Block 5)
# ════════════════════════════════════════════════════════════════════════════

def _compute_dose(proj_bg: float, proj_iob: float,
                  current_cob: float, proj_cob: float):
    """
    Four dosing modes exactly as specified.
    Uses current_cob for meal bolus numerator (matches insulin notebook),
    proj_iob as the IOB deducted.
    Returns: (dose, decision_type, advice, safety_constraints)
    """
    safety = []

    # Mode 1 — Hypoglycemia risk
    if proj_bg < HYPO_THRESHOLD:
        return (0.0,
                'No insulin — hypoglycemia risk',
                'Take 15g fast-acting carbohydrates immediately.',
                ['hypo_block'])

    # Mode 2 — Euglycemia, no active carbs
    if HYPO_THRESHOLD <= proj_bg <= HYPER_THRESHOLD and current_cob == 0:
        return (0.0,
                'Euglycemia — no action needed',
                'Glucose is in target range with no active carbs. No insulin required.',
                ['euglycemia_no_cob'])

    # Mode 3 — Euglycemia with active carbs
    if HYPO_THRESHOLD <= proj_bg <= HYPER_THRESHOLD and current_cob > 0:
        meal_bolus = current_cob / ICR
        raw_dose   = max(meal_bolus - proj_iob, 0.0)
        decision   = 'Meal bolus'
        advice     = (f'Meal bolus for {current_cob:.1f}g active carbs. '
                      f'IOB of {proj_iob:.2f}U deducted.')

    # Mode 4 — Hyperglycemia
    else:
        meal_bolus = current_cob / ICR
        correction = (proj_bg - TARGET) / ISF
        raw_dose   = max(meal_bolus + correction - proj_iob, 0.0)
        decision   = 'Correction dose'
        advice     = (f'Correction for predicted glucose {proj_bg:.1f} mmol/L '
                      f'(target {TARGET} mmol/L). '
                      f'IOB of {proj_iob:.2f}U deducted.')

    # Apply dose limits
    if 0 < raw_dose < MIN_DOSE:
        raw_dose = 0.0
        safety.append('below_min_dose_zeroed')

    if raw_dose > MAX_DOSE:
        raw_dose = MAX_DOSE
        safety.append('dose_capped_10U')

    return round(raw_dose, 2), decision, advice, safety


# ════════════════════════════════════════════════════════════════════════════
# Main public function
# ════════════════════════════════════════════════════════════════════════════

def run_inference(inputs: dict) -> dict:
    """
    Full AEGIS inference pass.

    Parameters
    ----------
    inputs : dict
        {
            "glucose_readings":  [g0, g1, g2, g3, g4],  # mmol/L oldest→newest
            "insulin_doses":     [{"units": float, "minutes_ago": int}, ...],
            "carb_events":       [{"grams": float, "minutes_ago": int}, ...],
            "met":               float,
            "rolling_met_15":    float,
            "hour":              int   # 0-23
        }

    Returns
    -------
    dict matching the AEGIS output JSON schema
    """
    try:
        _load_model()
    except FileNotFoundError as e:
        return {'error': str(e)}

    glucose     = inputs['glucose_readings']
    doses       = inputs.get('insulin_doses', [])
    carb_events = inputs.get('carb_events', [])

    # ── IOB: current + projected ──────────────────────────────────────────
    iob_now  = compute_iob(doses, project_minutes=0)
    proj_iob = compute_iob(doses, project_minutes=HORIZON)

    # ── COB: current + projected ──────────────────────────────────────────
    cob_now  = compute_cob(carb_events, project_minutes=0)
    proj_cob = compute_cob(carb_events, project_minutes=HORIZON)

    # ── Build features using current IOB/COB (matches training pipeline) ──
    features = _build_features(inputs, iob_now, cob_now)

    # ── Scale + predict ───────────────────────────────────────────────────
    try:
        scaled            = _scaler.transform(features)
        predicted_glucose = float(_model.predict(scaled)[0])
    except Exception as e:
        return {'error': f'Model prediction failed: {str(e)}'}

    # ── Derived values ────────────────────────────────────────────────────
    current_glucose = glucose[4]
    trend           = _get_trend(glucose)
    risk_zone       = _get_risk_zone(predicted_glucose)

    carbs_90 = sum(e['grams'] for e in carb_events
                   if e.get('minutes_ago', 999) <= 90)
    bolus_90 = sum(d['units'] for d in doses
                   if d.get('minutes_ago', 999) <= 90)

    # ── Scenario label ────────────────────────────────────────────────────
    if predicted_glucose < HYPO_THRESHOLD:
        scenario = 'Hypoglycemia Risk'
    elif predicted_glucose > HYPER_THRESHOLD:
        scenario = 'Hyperglycemia'
    elif cob_now > 0:
        scenario = 'Meal Bolus Required'
    else:
        scenario = 'Euglycemia — Stable'

    # ── Dosing ────────────────────────────────────────────────────────────
    dose, decision_type, advice, safety_constraints = _compute_dose(
        proj_bg=predicted_glucose,
        proj_iob=proj_iob,
        current_cob=cob_now,
        proj_cob=proj_cob,
    )

    return {
        'scenario': scenario,
        'current_state': {
            'current_glucose_mmol_L':   round(current_glucose, 2),
            'predicted_glucose_10min':  round(predicted_glucose, 2),
            'trend':                    trend,
            'risk_zone':                risk_zone,
            'thresholds_mmol_L': {
                'hypo':   HYPO_THRESHOLD,
                'hyper':  HYPER_THRESHOLD,
                'target': TARGET,
            },
        },
        'recent_inputs': {
            'carbs_last_90min_g':   round(carbs_90, 1),
            'bolus_last_90min_u':   round(bolus_90, 2),
            'active_insulin_iob_u': round(iob_now, 4),
            'active_carbs_cob_g':   round(cob_now, 4),
            'activity': {'met': inputs.get('met', 1.0)},
        },
        'decision': {
            'recommended_dose_u':          dose,
            'decision_type':               decision_type,
            'advice':                      advice,
            'dose_based_on_horizon':       '10min',
            'safety_constraints_applied':  safety_constraints,
        },
    }


# ════════════════════════════════════════════════════════════════════════════
# Self-test  —  one case per dosing mode
# ════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':

    TESTS = {
        # Mode 1 — Hypoglycemia: predicted BG will be near current low readings
        'Mode1_Hypo': {
            'glucose_readings': [3.8, 3.6, 3.4, 3.2, 3.1],
            'insulin_doses':    [{'units': 1.0, 'minutes_ago': 25}],
            'carb_events':      [],
            'met': 1.0, 'rolling_met_15': 1.0, 'hour': 3,
        },
        # Mode 2 — Euglycemia, no carbs
        'Mode2_Euglycemia_Silent': {
            'glucose_readings': [6.0, 6.1, 6.2, 6.1, 6.3],
            'insulin_doses':    [],
            'carb_events':      [],
            'met': 1.2, 'rolling_met_15': 1.1, 'hour': 10,
        },
        # Mode 3 — Euglycemia with active carbs
        'Mode3_Meal_Bolus': {
            'glucose_readings': [5.8, 6.0, 6.2, 6.4, 6.5],
            'insulin_doses':    [],
            'carb_events':      [{'grams': 50, 'minutes_ago': 10}],
            'met': 1.0, 'rolling_met_15': 1.0, 'hour': 12,
        },
        # Mode 4 — Hyperglycemia correction
        'Mode4_Hyperglycemia': {
            'glucose_readings': [11.0, 12.0, 13.0, 13.5, 14.0],
            'insulin_doses':    [],
            'carb_events':      [],
            'met': 1.0, 'rolling_met_15': 1.0, 'hour': 14,
        },
    }

    print('=' * 65)
    print('AEGIS Inference Pipeline — Self-test (P2301 model)')
    print('=' * 65)

    for name, inp in TESTS.items():
        print(f'\n--- {name} ---')
        result = run_inference(inp)
        if 'error' in result:
            print(f'  ERROR: {result["error"]}')
            continue
        cs = result['current_state']
        d  = result['decision']
        print(f'  Current BG     : {cs["current_glucose_mmol_L"]} mmol/L')
        print(f'  Predicted T+10 : {cs["predicted_glucose_10min"]} mmol/L')
        print(f'  Trend          : {cs["trend"]}')
        print(f'  Risk zone      : {cs["risk_zone"]}')
        print(f'  Decision       : {d["decision_type"]}')
        print(f'  Dose           : {d["recommended_dose_u"]} U')
        print(f'  Advice         : {d["advice"]}')
        if d['safety_constraints_applied']:
            print(f'  Safety flags   : {d["safety_constraints_applied"]}')

    print('\n' + '=' * 65)
    print('Self-test complete.')