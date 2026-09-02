'''
May 13th
- Add a check from hydra configuration to ensure the dim matches
- Aplit into dimension studies and other studies
'''
import os
import pickle
import re
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
import logging
from datetime import datetime # Needed for timestamp comparison if converting

# Configure basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
log = logging.getLogger(__name__)

# --- Configuration ---
# Directory where the .pkl result files are stored
RESULTS_DIR = "./"  # As per your script

# Directory where the generated plots will be saved
FIGURES_DIR = "./"

# --- Method and Objective Definitions ---
KNOWN_METHODS = ["bo", "dro", "turbo", 'pfns4bo', 'scorebo', "dro_ucb", "dro_ei", "dro_rotate", 'dro_mes']
KNOWN_OBJECIVES = ["ackley_20d", "ackley_10d", "ackley_5d", "ackley_2d",]

# --- Color Mapping for Methods ---
METHOD_COLORS = {
    "bo": "#1f77b4",      # Blue
    "dro": "#d62728",     # Red
    "turbo": "#2ca02c",   # Green
    "pfns4bo": "#9467bd", # Purple
    'scorebo': "#ff7f0e",  # Orange
    "dro_ucb": "#8c564b", # Brown
    "dro_ei": "#e377c2",  # Pink
    "dro_rotate": "#7f7f7f", # Gray
    "dro_mes": "#bcbd22", # Olive
    
}

DEFAULT_PLOT_COLOR = "#7f7f7f" # Gray

# --- Helper Functions ---
def parse_filename(filename):
    """
    Parses the filename to extract method name, test function identifier,
    and timestamp string.
    Returns (method_name, test_function_id, timestamp_str) or ('na', 'na', None).
    """
    method_name_found, test_function_id_found = 'na', 'na'
    timestamp_str = None

    # Regex to capture the core name part and the timestamp
    # Example: bo_ackley_2d_20230101_120000_all_trials.pkl
    match = re.match(r"^(.*?)_(\d{8}_\d{6})_all_trials\.pkl$", filename)
    if not match:
        log.debug(f"Filename {filename} does not match expected timestamp pattern. Skipping.")
        return 'na', 'na', None

    core_part = match.group(1) # e.g., "bo_ackley_2d"
    timestamp_str = match.group(2) # e.g., "20230101_120000"

    # Use lowercase for matching known methods/objectives
    core_lower = core_part.lower()

    # Sort known names by length descending for more robust matching
    sorted_known_methods = sorted(KNOWN_METHODS, key=len, reverse=True)
    sorted_known_objectives = sorted(KNOWN_OBJECIVES, key=len, reverse=True)

    # Find Method
    for method in sorted_known_methods:
        # Check if the core part starts or ends with the method name + underscore
        # This is slightly more robust than just 'in'
        if core_lower.startswith(method + "_") or core_lower.endswith("_" + method):
             method_name_found = method
             break
        elif method in core_lower: # Fallback to simple substring check
             method_name_found = method
             break # Take the first (longest) match

    # Find Objective
    for objective in sorted_known_objectives:
        if objective in core_lower:
            test_function_id_found = objective
            break # Take the first (longest) match

    if method_name_found == 'na' or test_function_id_found == 'na':
        log.warning(f"Could not determine method or test function from core part '{core_part}' of filename: {filename}. Skipping.")
        return 'na', 'na', None # Return 'na' strings and None for timestamp

    # Basic validation: Ensure the identified method and objective are actually in the core part
    # This helps prevent accidental matches if names are substrings of each other in weird ways
    # (Though the length sorting helps mitigate this)
    if method_name_found not in core_lower or test_function_id_found not in core_lower:
         log.warning(f"Potential mismatch in parsing '{core_part}'. Found method '{method_name_found}', objective '{test_function_id_found}'. Skipping {filename}.")
         return 'na', 'na', None


    return method_name_found, test_function_id_found, timestamp_str

# --- Main Script Logic ---

def main():
    log.info(f"Starting visualization script. Reading from: {RESULTS_DIR}, Saving to: {FIGURES_DIR}")
    log.info(f"Known methods for parsing: {KNOWN_METHODS}")
    log.info(f"Color mapping: {METHOD_COLORS}")

    if not os.path.isdir(RESULTS_DIR):
        log.error(f"Results directory not found: {RESULTS_DIR}")
        print(f"Error: Results directory '{RESULTS_DIR}' not found. Please create it or check the path.")
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)
    log.info(f"Ensured figure directory exists: {FIGURES_DIR}")

    # --- 1. Scan and Select Latest Files ---
    log.info("Scanning for .pkl files and selecting the latest for each experiment...")
    files_by_experiment = defaultdict(list) # key: (method, test_func_id), value: list of (timestamp_str, filepath)

    all_pkl_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".pkl")]
    log.info(f"Found {len(all_pkl_files)} total .pkl files.")

    parsed_count = 0
    for filename in all_pkl_files:
        method, test_func_id, timestamp = parse_filename(filename)
        if method != 'na' and test_func_id != 'na' and timestamp is not None:
            filepath = os.path.join(RESULTS_DIR, filename)
            files_by_experiment[(method, test_func_id)].append((timestamp, filepath))
            parsed_count += 1
        # Warnings for failed parses are handled within parse_filename

    log.info(f"Successfully parsed {parsed_count} files to identify experiments.")

    latest_files_to_process = []
    for experiment_key, file_list in files_by_experiment.items():
        if not file_list:
            continue
        # Sort by timestamp string descending (lexicographical comparison works for YYYYMMDD_HHMMSS)
        file_list.sort(key=lambda x: x[0], reverse=True)
        latest_timestamp, latest_filepath = file_list[0]
        latest_files_to_process.append(latest_filepath)
        if len(file_list) > 1:
            log.info(f"  Experiment {experiment_key}: Found {len(file_list)} files. Using latest: {os.path.basename(latest_filepath)} (Timestamp: {latest_timestamp})")

    log.info(f"Selected {len(latest_files_to_process)} unique latest files to process.")


    # --- 2. Load and Process Data from Latest Files ---
    results_by_function = defaultdict(lambda: defaultdict(list))
    max_len_per_function = defaultdict(int)
    processed_files = 0

    if not latest_files_to_process:
         log.warning("No valid latest files selected. Exiting.")
         print("No valid .pkl files matching the expected format and naming conventions were found.")
         return

    log.info("Loading data from selected latest files...")
    for filepath in latest_files_to_process:
        filename = os.path.basename(filepath) # Get filename for parsing again (or store parsed info earlier)
        method, test_func_id, _ = parse_filename(filename) # We know this should succeed now

        if method == 'na' or test_func_id == 'na': # Should not happen based on selection logic, but safe check
             log.error(f"Error re-parsing selected file: {filename}. This should not happen.")
             continue

        try:
            with open(filepath, 'rb') as f:
                all_trials_data = pickle.load(f)
            log.debug(f"Successfully loaded {filepath} ({method} on {test_func_id}). Contains {len(all_trials_data)} trials.")

            trial_curves = [] # Collect curves for this file/experiment
            current_max_len = 0

            for trial_idx, trial_data in enumerate(all_trials_data):
                if 'all_y' not in trial_data:
                    log.warning(f"  'all_y' not found in trial {trial_idx} in {filename}. Skipping trial.")
                    continue

                all_y = np.asarray(trial_data['all_y'])
                if all_y.ndim == 0 or len(all_y) == 0:
                    log.warning(f"  'all_y' is empty or scalar in trial {trial_idx} in {filename}. Skipping trial.")
                    continue

                cumulative_max_y = np.maximum.accumulate(all_y)
                trial_curves.append(cumulative_max_y)

                # Track max length within this specific experiment's trials
                if len(cumulative_max_y) > current_max_len:
                    current_max_len = len(cumulative_max_y)

            # Store all curves for this experiment together
            if trial_curves:
                 results_by_function[test_func_id][method].extend(trial_curves) # Use extend to add all trial curves
                 # Update the overall max length for this test function across all methods
                 if current_max_len > max_len_per_function[test_func_id]:
                      max_len_per_function[test_func_id] = current_max_len
                 processed_files += 1
            else:
                 log.warning(f"No valid trials found in {filename}.")


        except pickle.UnpicklingError as e:
            log.error(f"Error unpickling {filepath}: {e}")
        except Exception as e:
            log.error(f"An unexpected error occurred processing {filepath}: {e}", exc_info=True)

    log.info(f"Successfully processed data from {processed_files} latest files.")

    if not results_by_function:
        log.warning("No data loaded after filtering for latest files. Exiting.")
        print("No data could be extracted from the selected latest .pkl files.")
        return

    # --- 3. Plotting (Identical to previous version) ---
    log.info("Generating plots...")
    plot_count = 0
    for test_func_id, methods_data in results_by_function.items():
        log.info(f"Plotting for test function: {test_func_id}")
        plt.figure(figsize=(12, 8))

        target_len = max_len_per_function[test_func_id]
        if target_len == 0:
            log.warning(f"  No valid data length found for {test_func_id}, skipping plot.")
            continue

        iterations = np.arange(target_len)

        # Sort methods alphabetically for consistent legend order
        sorted_method_names = sorted(methods_data.keys())

        for method_name in sorted_method_names:
            curves_list = methods_data[method_name]
            if not curves_list:
                # This shouldn't happen if data was added correctly, but good check
                log.warning(f"  No curves found for method {method_name} on {test_func_id} during plotting. Skipping.")
                continue

            padded_curves = []
            for curve in curves_list:
                # Pad shorter curves
                if len(curve) < target_len:
                    # Check if curve is empty before accessing last element
                    if len(curve) > 0:
                        padding_val = curve[-1]
                        padding = np.full(target_len - len(curve), padding_val)
                        padded_curve = np.concatenate((curve, padding))
                    else: # Handle empty curve case - pad with NaN
                        padded_curve = np.full(target_len, np.nan)

                # Truncate longer curves (shouldn't happen often with max_len logic)
                elif len(curve) > target_len:
                    padded_curve = curve[:target_len]
                else:
                    padded_curve = curve

                padded_curves.append(padded_curve)

            if not padded_curves: continue # Skip if all curves were empty

            curves_array = np.array(padded_curves)

            # Filter out trials that might be all NaNs after padding
            valid_trials_mask = ~np.all(np.isnan(curves_array), axis=1)
            if not np.any(valid_trials_mask):
                log.warning(f"  All trials for method {method_name} on {test_func_id} are NaN after padding. Skipping.")
                continue
            curves_array = curves_array[valid_trials_mask]

            mean_curve = np.nanmean(curves_array, axis=0)

            counts_per_iteration = np.sum(~np.isnan(curves_array), axis=0)
            std_dev_curve = np.nanstd(curves_array, axis=0)
            stderr_curve = np.full_like(mean_curve, np.nan)
            valid_counts_mask = counts_per_iteration > 0
            stderr_curve[valid_counts_mask] = std_dev_curve[valid_counts_mask] / np.sqrt(counts_per_iteration[valid_counts_mask])

            current_method_color = METHOD_COLORS.get(method_name, DEFAULT_PLOT_COLOR)

            plt.plot(iterations, mean_curve, label=f"{method_name.upper()} (Trials: {curves_array.shape[0]})", color=current_method_color)
            plt.fill_between(iterations, mean_curve - stderr_curve, mean_curve + stderr_curve, alpha=0.2, color=current_method_color)

        plt.title(f"Comparison of Methods on {test_func_id.replace('_', ' ').title()}\nAverage Cumulative Maximum Objective Value")
        plt.xlabel("Iteration (incl. initial points)")
        plt.ylabel("Best Objective Value Found")
        plt.legend(title="Methods")
        plt.grid(True, which="both", linestyle='--', linewidth=0.5)
        plt.tight_layout()

        safe_test_func_id = re.sub(r'[^\w\-_.]', '_', test_func_id)
        plot_filename = f"comparison_{safe_test_func_id}.pdf"
        plot_filepath = os.path.join(FIGURES_DIR, plot_filename)

        try:
            plt.savefig(plot_filepath)
            log.info(f"  Saved plot: {plot_filepath}")
            plot_count +=1
        except Exception as e:
            log.error(f"  Failed to save plot {plot_filepath}: {e}")
        plt.close()

    log.info(f"Generated {plot_count} plots.")
    log.info("Visualization script finished.")

if __name__ == "__main__":
    main()
