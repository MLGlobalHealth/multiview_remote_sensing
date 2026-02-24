import os
import subprocess
import random
import csv
import argparse
import re

def run_command(command, capture_output=False):
    """Runs a terminal command. Can capture output to extract scores."""
    print(f"Running: {command}")
    
    if capture_output:
        # Run and capture the printed text
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr)
            raise Exception(f"Command failed: {command}")
        return result.stdout
    else:
        # Just run it normally (for fine-tuning so you can see the progress bar)
        result = subprocess.run(command, shell=True)
        if result.returncode != 0:
            raise Exception(f"Command failed: {command}")
        return None

def extract_score(output_text):
    """Hunts through the terminal output from bottom to top to find the MAE float."""
    lines = output_text.strip().split('\n')
    # Read from the bottom up to find the last printed number (which is usually the final MAE)
    for line in reversed(lines):
        # Look for a decimal number in the text
        match = re.search(r'([0-9]+\.[0-9]+)', line)
        if match:
            return float(match.group(1))
    return "Error"

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run full fine-tune and eval pipeline for random bands.')
    parser.add_argument('--imagery_path', type=str, required=True, help='Path to imagery')
    parser.add_argument('--num_random_tests', type=int, default=1, help='How many random band combos to test')
    args = parser.parse_args()

    available_bands = list(range(1, 14))
    tested_combos = set()
    csv_filename = 'modelling/dino/random_bands_full_pipeline_results.csv'

    # Initialize CSV with the new individual fold columns
    if not os.path.exists(csv_filename):
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Band_1', 'Band_2', 'Band_3', 'Fold_1', 'Fold_2', 'Fold_3', 'Fold_4', 'Fold_5', 'Avg_MAE', 'Status'])

    while len(tested_combos) < args.num_random_tests:
        combo = tuple(sorted(random.sample(available_bands, 3)))
        if combo in tested_combos:
            continue
            
        tested_combos.add(combo)
        b1, b2, b3 = combo
        band_str = f"{b1} {b2} {b3}"
        
        print(f"\n=======================================================")
        print(f"STARTING FULL PIPELINE FOR BANDS: [{band_str}]")
        print(f"=======================================================\n")

        try:
            # STEP 1: Fine-tune all 5 folds
            for fold in range(1, 6):
                print(f"\n--- Fine-tuning Fold {fold} for bands [{band_str}] ---")
                ft_cmd = f"python modelling/dino/finetune_spatial.py --fold {fold} --model_name dinov2_vitb14 --imagery_path {args.imagery_path} --batch_size 1 --imagery_source S --num_epochs 20 --grouped_bands {band_str}"
                run_command(ft_cmd, capture_output=False)

            # STEP 2: Evaluate folds individually to get exact MAEs
            fold_maes = []
            for fold in range(1, 6):
                print(f"\n--- Evaluating Fold {fold} for bands [{band_str}] ---")
                # Notice we added --fold {fold} here!
                eval_cmd = f"python modelling/dino/evaluate.py --fold {fold} --use_checkpoint --imagery_path {args.imagery_path} --imagery_source S --mode spatial --grouped_bands {band_str}"
                eval_output = run_command(eval_cmd, capture_output=True)
                
                score = extract_score(eval_output)
                print(f"Extracted Score for Fold {fold}: {score}")
                fold_maes.append(score)

            # STEP 3: Calculate the average MAE ourselves
            valid_scores = [s for s in fold_maes if isinstance(s, float)]
            if len(valid_scores) == 5:
                avg_mae = sum(valid_scores) / 5.0
                status = 'Success'
            else:
                avg_mae = "Error"
                status = 'Partial Failure'

            # STEP 4: Log everything to the CSV
            with open(csv_filename, mode='a', newline='') as f:
                row_data = [b1, b2, b3] + fold_maes + [avg_mae, status]
                csv.writer(f).writerow(row_data)
                
        except Exception as e:
            print(f"Pipeline failed for bands [{band_str}]: {e}")
            with open(csv_filename, mode='a', newline='') as f:
                csv.writer(f).writerow([b1, b2, b3, 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'N/A', 'Failed'])

        print(f"Finished pipeline for [{band_str}]")