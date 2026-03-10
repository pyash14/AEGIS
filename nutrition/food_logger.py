import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.database import save_food_log, get_recent_food_logs
from llm.meal_parser import parse_meal

def log_meal(meal_text: str) -> dict:
    try:
        result = parse_meal(meal_text)
        
        if 'error' in result:
            return {'success': False, 'error': result['error']}
        
        save_food_log(result)
        
        return {
            'success': True,
            'meal_description': result['meal_description'],
            'totals': result['totals'],
            'confidence': result['confidence'],
            'items_count': len(result['items'])
        }

    except Exception as e:
        return {'success': False, 'error': str(e)}

def get_meal_history(limit: int = 5) -> list:
    try:
        return get_recent_food_logs(limit)
    except Exception as e:
        print(f'Error getting meal history: {str(e)}')
        return []

if __name__ == '__main__':
    print('Logging meal...')
    result = log_meal('2 eggs with toast')
    print('Result:', result)
    
    print('\nRecent meals:')
    history = get_meal_history()
    for meal in history:
        print(f"- {meal['meal_description']} | carbs: {meal['total_carbs_g']}g")