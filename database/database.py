import sqlite3
import json
from datetime import datetime

DB_PATH = 'database/patient_logs.db'

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def create_tables():
    try:
        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS session_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                scenario TEXT,
                current_glucose REAL,
                predicted_10min REAL,
                risk_zone TEXT,
                recommended_dose REAL,
                decision_type TEXT,
                iob REAL,
                cob REAL,
                full_json TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS food_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                meal_description TEXT,
                total_carbs_g REAL,
                total_protein_g REAL,
                total_fat_g REAL,
                total_calories REAL,
                items_json TEXT,
                confidence TEXT
            )
        ''')

        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                timestamp TEXT NOT NULL,
                role TEXT,
                content TEXT
            )
        ''')

        conn.commit()
        conn.close()
        print('Tables created successfully')

    except Exception as e:
        print(f'Database error: {str(e)}')

def save_session(result: dict):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO session_history
            (timestamp, scenario, current_glucose, predicted_10min,
             risk_zone, recommended_dose, decision_type, iob, cob, full_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            result.get('scenario'),
            result.get('current_state', {}).get('current_glucose_mmol_L'),
            result.get('current_state', {}).get('predicted_glucose_10min'),
            result.get('current_state', {}).get('risk_zone'),
            result.get('decision', {}).get('recommended_dose_u'),
            result.get('decision', {}).get('decision_type'),
            result.get('recent_inputs', {}).get('active_insulin_iob_u'),
            result.get('recent_inputs', {}).get('active_carbs_cob_g'),
            json.dumps(result)
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f'Save session error: {str(e)}')

def save_food_log(meal: dict):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO food_logs
            (timestamp, meal_description, total_carbs_g, total_protein_g,
             total_fat_g, total_calories, items_json, confidence)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now().isoformat(),
            meal.get('meal_description'),
            meal.get('totals', {}).get('carbs_g'),
            meal.get('totals', {}).get('protein_g'),
            meal.get('totals', {}).get('fat_g'),
            meal.get('totals', {}).get('calories'),
            json.dumps(meal.get('items', [])),
            meal.get('confidence')
        ))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f'Save food log error: {str(e)}')

def get_chat_history(session_id: str, limit: int = 5) -> list:
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT role, content FROM chat_history
            WHERE session_id = ?
            ORDER BY timestamp DESC LIMIT ?
        ''', (session_id, limit))
        rows = cursor.fetchall()
        conn.close()
        return [{'role': r['role'], 'content': r['content']} for r in reversed(rows)]
    except Exception as e:
        print(f'Get chat history error: {str(e)}')
        return []

def save_chat_message(session_id: str, role: str, content: str):
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO chat_history (session_id, timestamp, role, content)
            VALUES (?, ?, ?, ?)
        ''', (session_id, datetime.now().isoformat(), role, content))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f'Save chat message error: {str(e)}')

def get_recent_sessions(hours: int = 24) -> list:
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM session_history
            WHERE timestamp >= datetime('now', ?)
            ORDER BY timestamp ASC
        ''', (f'-{hours} hours',))
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f'Get recent sessions error: {str(e)}')
        return []

def get_recent_food_logs(limit: int = 5) -> list:
    try:
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM food_logs
            ORDER BY timestamp DESC LIMIT ?
        ''', (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [dict(r) for r in rows]
    except Exception as e:
        print(f'Get recent food logs error: {str(e)}')
        return []

if __name__ == '__main__':
    create_tables()