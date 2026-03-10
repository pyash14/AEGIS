import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
import numpy as np
import pandas as pd
from datasets import load_dataset

from llm.meal_parser import parse_meal

def run_evaluation():
    print('Loading NutriBench v2 dataset...')
    try:
        dataset = load_dataset('dongx1997/NutriBench', 'v2')
        df = dataset['train'].to_pandas()
        print(f'Loaded {len(df)} samples')
    except Exception as e:
        print(f'Failed to load dataset: {str(e)}')
        return

    # Sample meals
    sample_size = int(os.environ.get('NUTRIBENCH_SAMPLE', 300))
    df_sample = df.sample(n=sample_size, random_state=42).reset_index(drop=True)
    print(f'Sampled {sample_size} meals for evaluation')

    os.makedirs('outputs', exist_ok=True)
    results = []
    failed = []

    for i, row in df_sample.iterrows():
        meal_text     = row['meal_description']
        true_carbs    = float(row['carb']    or 0)
        true_protein  = float(row['protein'] or 0)
        true_fat      = float(row['fat']     or 0)
        true_calories = float(row['energy']  or 0)

        try:
            result = parse_meal(meal_text)

            if 'error' in result:
                raise ValueError(result['error'])

            pred_carbs    = result['totals'].get('carbs_g',   0) or 0
            pred_protein  = result['totals'].get('protein_g', 0) or 0
            pred_fat      = result['totals'].get('fat_g',     0) or 0
            pred_calories = result['totals'].get('calories',  0) or 0

            results.append({
                'meal_description': meal_text,
                'true_carbs':    true_carbs,
                'pred_carbs':    pred_carbs,
                'true_protein':  true_protein,
                'pred_protein':  pred_protein,
                'true_fat':      true_fat,
                'pred_fat':      pred_fat,
                'true_calories': true_calories,
                'pred_calories': pred_calories
            })

        except Exception as e:
            failed.append({'meal': meal_text, 'error': str(e)})
            with open('outputs/failed_meals.txt', 'a') as f:
                f.write(f'Meal: {meal_text}\nError: {str(e)}\n\n')

        if (i + 1) % 5 == 0:
            print(f'Progress: {i+1}/{sample_size}')

        time.sleep(2)

    print(f'\nCompleted: {len(results)} success, {len(failed)} failed')

    if not results:
        print('No results to evaluate!')
        return

    df_results = pd.DataFrame(results)
    df_results.to_csv('outputs/nutribench_raw_results.csv', index=False)

    def mae(true, pred):
        return float(np.mean(np.abs(np.array(true) - np.array(pred))))

    def rmse(true, pred):
        return float(np.sqrt(np.mean((np.array(true) - np.array(pred))**2)))

    def within_n(true, pred, n):
        return float(np.mean(np.abs(np.array(true) - np.array(pred)) <= n) * 100)

    metrics = {
        'n_evaluated': len(results),
        'n_failed': len(failed),
        'carbs': {
            'MAE':        round(mae(df_results.true_carbs, df_results.pred_carbs), 2),
            'RMSE':       round(rmse(df_results.true_carbs, df_results.pred_carbs), 2),
            'Within_10g': round(within_n(df_results.true_carbs, df_results.pred_carbs, 10), 1)
        },
        'protein': {
            'MAE':  round(mae(df_results.true_protein, df_results.pred_protein), 2),
            'RMSE': round(rmse(df_results.true_protein, df_results.pred_protein), 2)
        },
        'fat': {
            'MAE':  round(mae(df_results.true_fat, df_results.pred_fat), 2),
            'RMSE': round(rmse(df_results.true_fat, df_results.pred_fat), 2)
        },
        'calories': {
            'MAE':           round(mae(df_results.true_calories, df_results.pred_calories), 2),
            'RMSE':          round(rmse(df_results.true_calories, df_results.pred_calories), 2),
            'Within_50kcal': round(within_n(df_results.true_calories, df_results.pred_calories, 50), 1)
        }
    }

    with open('outputs/nutribench_summary.json', 'w') as f:
        json.dump(metrics, f, indent=2)

    print('\n=== NUTRIBENCH EVALUATION RESULTS ===')
    print(f"{'Metric':<20} {'Carbs':>10} {'Protein':>10} {'Fat':>10} {'Calories':>10}")
    print('-' * 62)
    print(f"{'MAE':<20} {metrics['carbs']['MAE']:>10} {metrics['protein']['MAE']:>10} {metrics['fat']['MAE']:>10} {metrics['calories']['MAE']:>10}")
    print(f"{'RMSE':<20} {metrics['carbs']['RMSE']:>10} {metrics['protein']['RMSE']:>10} {metrics['fat']['RMSE']:>10} {metrics['calories']['RMSE']:>10}")
    print(f"{'Within_10g':<20} {str(metrics['carbs']['Within_10g'])+'%':>10} {'-':>10} {'-':>10} {'-':>10}")
    print(f"{'Within_50kcal':<20} {'-':>10} {'-':>10} {'-':>10} {str(metrics['calories']['Within_50kcal'])+'%':>10}")
    print('\nResults saved to outputs/')

if __name__ == '__main__':
    run_evaluation()