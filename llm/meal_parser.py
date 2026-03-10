import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from llm.claude_client import call_claude
from rag.retriever import retrieve_nutrition

PARSE_SYSTEM_PROMPT = """You are a nutrition parser. Your job is to extract individual food items and their quantities from meal descriptions.

Always respond with valid JSON only. No extra text, no markdown, no backticks.

Format:
{"items": [{"food_item": "food name", "quantity_g": number}, ...]}

Rules:
- Estimate quantity in grams if not specified
- Split combined dishes into individual ingredients
- Use common portion sizes if needed (1 cup rice = 200g, 1 banana = 120g, 1 bowl = 250g)"""

AGGREGATE_SYSTEM_PROMPT = """You are a nutrition calculator. Given a list of food items with their nutrition data, calculate the total nutrition for the meal.

Always respond with valid JSON only. No extra text, no markdown, no backticks.

Format:
{"total_carbs_g": number, "total_protein_g": number, "total_fat_g": number, "total_calories": number, "confidence": "high|medium|low"}"""

def parse_meal(meal_text: str) -> dict:
    try:
        # Step 1 - Parse meal text into food items
        parse_response = call_claude(
            system_prompt=PARSE_SYSTEM_PROMPT,
            user_message=f'Parse this meal: {meal_text}'
        )

        # Clean response
        clean = parse_response.strip()
        if '```' in clean:
            clean = clean.split('```')[1]
            if clean.startswith('json'):
                clean = clean[4:]
        
        parsed = json.loads(clean)
        items = parsed.get('items', [])

        # Step 2 - FAISS lookup for each item
        enriched_items = []
        for item in items:
            food_name = item.get('food_item', '')
            quantity_g = item.get('quantity_g', 100)

            matches = retrieve_nutrition(food_name, top_k=1)
            if matches:
                match = matches[0]
                scale = quantity_g / 100.0
                enriched_items.append({
                    'food': food_name,
                    'matched_to': match['food_name'],
                    'quantity_g': quantity_g,
                    'carbs_g': round(match['carbs_g'] * scale, 2),
                    'protein_g': round(match['protein_g'] * scale, 2),
                    'fat_g': round(match['fat_g'] * scale, 2),
                    'calories': round(match['calories'] * scale, 2)
                })

        # Step 3 - Aggregate totals
        items_summary = json.dumps(enriched_items)
        agg_response = call_claude(
            system_prompt=AGGREGATE_SYSTEM_PROMPT,
            user_message=f'Calculate totals for: {items_summary}'
        )

        clean_agg = agg_response.strip()
        if '```' in clean_agg:
            clean_agg = clean_agg.split('```')[1]
            if clean_agg.startswith('json'):
                clean_agg = clean_agg[4:]

        totals = json.loads(clean_agg)

        return {
            'meal_description': meal_text,
            'items': enriched_items,
            'totals': {
                'carbs_g': totals.get('total_carbs_g', 0),
                'protein_g': totals.get('total_protein_g', 0),
                'fat_g': totals.get('total_fat_g', 0),
                'calories': totals.get('total_calories', 0)
            },
            'confidence': totals.get('confidence', 'medium')
        }

    except Exception as e:
        return {'error': str(e), 'meal_description': meal_text}


if __name__ == '__main__':
    result = parse_meal('bowl of oatmeal with banana')
    print(json.dumps(result, indent=2))