'''
May 13th
- Add a check from hydra configuration to ensure the dim matches
- Split into dimension studies and other studies
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
RESULTS_DIR = "./"
FIGURES_DIR = "./"

# font size
fontsize = 24

# --- Method and Objective Definitions ---
KNOWN_METHODS = ["bo", "dro_1", "dro_2", "dro_3", "dro_5", "dro_10", "dro", "turbo", 'pfns4bo', 'scorebo']
METHOD_LABELS = {"bo":"BO", "dro":"DRO", "turbo":"TuRBO", 'pfns4bo': "PFNs4BO", 'scorebo': "SCoreBO"}
KNOWN_OBJECIVES = ["ackley_20d", "ackley_10d", "ackley_5d", "ackley_2d",]
ACKLEY_DIMS_ORDERED = ["ackley_2d", "ackley_5d", "ackley_10d", "ackley_20d"]

# --- Color Mapping for Methods ---
METHOD_COLORS = {
    "bo": "#1f77b4", "dro": "#d62728", "turbo": "#2ca02c",
    "pfns4bo": "#9467bd", 'scorebo': "#ff7f0e", "dro_1": "#8c564b",
    "dro_2": "#e377c2", "dro_3": "#7f7f7f", "dro_5": "#bcbd22", "dro_10": "#17becf",
}
DEFAULT_PLOT_COLOR = "#7f7f7f"
Y_AXIS_LABEL = "Best Observed Value"

# --- Helper Functions ---
def parse_filename(filename):
    method_name_found, test_function_id_found, timestamp_str = 'na', 'na', None
    match = re.match(r"^(.*?)_(\d{8}_\d{6})_all_trials\.pkl$", filename)
    if not match:
        log.debug(f"Filename {filename} does not match expected timestamp pattern. Skipping.")
        return 'na', 'na', None
    core_part, timestamp_str = match.group(1), match.group(2)
    core_lower = core_part.lower()
    sorted_known_methods = sorted(KNOWN_METHODS, key=len, reverse=True)
    sorted_known_objectives = sorted(KNOWN_OBJECIVES, key=len, reverse=True)
    for method in sorted_known_methods:
        if core_lower.startswith(method + "_") or core_lower.endswith("_" + method) or method in core_lower:
            method_name_found = method
            break
    for objective in sorted_known_objectives:
        if objective in core_lower:
            test_function_id_found = objective
            break
    if method_name_found == 'na' or test_function_id_found == 'na' or \
       method_name_found not in core_lower or test_function_id_found not in core_lower:
        log.warning(f"Could not accurately parse method/objective from '{core_part}' in {filename}. Skipping.")
        return 'na', 'na', None
    return method_name_found, test_function_id_found, timestamp_str

def get_plotting_data_for_group(methods_data_group, target_len_group):
    """
    Pre-calculates mean, stderr, and other info for a group of methods.
    This is used to determine the alignment baseline before actual plotting.
    """
    all_methods_plot_info = []
    if not methods_data_group or target_len_group == 0:
        return [], np.nan

    sorted_method_names = sorted(methods_data_group.keys())

    for method_name in sorted_method_names:
        curves_list = methods_data_group[method_name]
        if not curves_list: continue

        padded_curves = []
        for curve in curves_list:
            if len(curve) < target_len_group:
                padding_val = curve[-1] if len(curve) > 0 else np.nan
                padding = np.full(target_len_group - len(curve), padding_val)
                padded_curve = np.concatenate((curve, padding))
            elif len(curve) > target_len_group:
                padded_curve = curve[:target_len_group]
            else:
                padded_curve = curve
            padded_curves.append(padded_curve)

        if not padded_curves: continue
        curves_array = np.array(padded_curves)
        valid_trials_mask = ~np.all(np.isnan(curves_array), axis=1)
        if not np.any(valid_trials_mask): continue
        curves_array = curves_array[valid_trials_mask]

        original_mean_curve = np.nanmean(curves_array, axis=0)
        counts_per_iteration = np.sum(~np.isnan(curves_array), axis=0)
        std_dev_curve = np.nanstd(curves_array, axis=0)
        stderr_curve = np.full_like(original_mean_curve, np.nan)
        valid_counts_mask = counts_per_iteration > 0
        stderr_curve[valid_counts_mask] = std_dev_curve[valid_counts_mask] / np.sqrt(counts_per_iteration[valid_counts_mask])
        
        all_methods_plot_info.append({
            'name': method_name,
            'original_mean': original_mean_curve,
            'stderr': stderr_curve,
            'trials': curves_array.shape[0]
        })

    initial_values = []
    for item in all_methods_plot_info:
        if not np.all(np.isnan(item['original_mean'])) and not np.isnan(item['original_mean'][0]):
            initial_values.append(item['original_mean'][0])
    
    alignment_baseline_y = np.nanmin(initial_values) if initial_values else np.nan
    return all_methods_plot_info, alignment_baseline_y


def plot_ackley_dimension_study(ackley_data, max_len_per_function, method_colors, figures_dir):
    log.info("Generating Ackley dimension study plot...")
    num_dims = len(ACKLEY_DIMS_ORDERED)
    fig, axes = plt.subplots(1, num_dims, figsize=(20, 5.5), sharey=True)
    if num_dims == 1: axes = [axes]

    legend_handles = {}
    all_method_names_in_study = set()
    for test_func_id in ackley_data:
        for method_name in ackley_data[test_func_id]:
            all_method_names_in_study.add(method_name)
    sorted_all_method_names = sorted(list(all_method_names_in_study))

    for i, test_func_id in enumerate(ACKLEY_DIMS_ORDERED):
        ax = axes[i]
        methods_data_subplot = ackley_data.get(test_func_id)
        target_len_subplot = max_len_per_function.get(test_func_id, 0)
        
        ax.set_title(f"{test_func_id.replace('_', ' ').title()}", fontsize=fontsize)
        ax.set_xlabel("Iteration", fontsize=fontsize)
        if i == 0: ax.set_ylabel(Y_AXIS_LABEL, fontsize=fontsize)
        ax.grid(True, which="both", linestyle='--', linewidth=0.5)

        if not methods_data_subplot or target_len_subplot == 0:
            log.warning(f"No data or zero target length for {test_func_id} in Ackley study. Subplot will be mostly empty.")
            ax.text(0.5, 0.5, "No data", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            continue

        iterations = np.arange(target_len_subplot)
        
        # Pre-calculate means and determine alignment baseline for this subplot
        subplot_plot_info, alignment_baseline_y = get_plotting_data_for_group(methods_data_subplot, target_len_subplot)

        if not subplot_plot_info:
            log.warning(f"No valid plot info generated for {test_func_id}. Subplot empty.")
            ax.text(0.5, 0.5, "Processing error", ha='center', va='center', transform=ax.transAxes)
            continue

        for item in subplot_plot_info:
            method_name = item['name']
            original_mean_curve = item['original_mean']
            stderr_curve = item['stderr']
            
            displayed_mean_curve = original_mean_curve # Default
            if not np.all(np.isnan(original_mean_curve)) and \
               not np.isnan(original_mean_curve[0]) and \
               not np.isnan(alignment_baseline_y):
                shift = alignment_baseline_y - original_mean_curve[0]
                displayed_mean_curve[0] = original_mean_curve[0] + shift
                displayed_mean_curve = np.maximum.accumulate(displayed_mean_curve)
            
            current_method_color = method_colors.get(method_name, DEFAULT_PLOT_COLOR)
            line, = ax.plot(iterations, displayed_mean_curve, color=current_method_color, label=method_name.upper())
            ax.fill_between(iterations, displayed_mean_curve - stderr_curve, displayed_mean_curve + stderr_curve, alpha=0.2, color=current_method_color)
            if method_name not in legend_handles:
                 legend_handles[method_name] = line

    ordered_handles = [legend_handles[m] for m in sorted_all_method_names if m in legend_handles]
    ordered_labels = [METHOD_LABELS.get(m, m.upper()) for m in sorted_all_method_names if m in legend_handles]
    if ordered_handles:
        num_legend_cols = min(len(ordered_handles), 5)
        fig.legend(ordered_handles, ordered_labels, loc='upper center', ncol=num_legend_cols, bbox_to_anchor=(0.5, 0.98), fontsize=fontsize)
    
    # fig.suptitle("Comparison of Methods on Ackley Function (Varying Dimensions)\n"
                #  "Mean curves in each subplot are vertically aligned to the lowest initial mean for that subplot.",
                #  fontsize=14, y=1.05) # Adjusted font size and y
    plt.tight_layout(rect=[0, 0.03, 1, 0.8]) # Adjusted rect

    plot_filename = "comparison_ackley_dim_study.pdf"
    plot_filepath = os.path.join(figures_dir, plot_filename)
    try:
        plt.savefig(plot_filepath)
        log.info(f"  Saved Ackley dimension study plot: {plot_filepath}")
    except Exception as e:
        log.error(f"  Failed to save Ackley dimension study plot {plot_filepath}: {e}")
    plt.close(fig)

def main():
    log.info(f"Starting visualization script. Reading from: {RESULTS_DIR}, Saving to: {FIGURES_DIR}")
    # ... (rest of setup and file scanning logic remains the same) ...
    if not os.path.isdir(RESULTS_DIR):
        log.error(f"Results directory not found: {RESULTS_DIR}")
        print(f"Error: Results directory '{RESULTS_DIR}' not found. Please create it or check the path.")
        return

    os.makedirs(FIGURES_DIR, exist_ok=True)
    log.info(f"Ensured figure directory exists: {FIGURES_DIR}")

    # --- 1. Scan and Select Latest Files ---
    log.info("Scanning for .pkl files and selecting the latest for each experiment...")
    files_by_experiment = defaultdict(list) 

    all_pkl_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".pkl")]
    log.info(f"Found {len(all_pkl_files)} total .pkl files.")

    parsed_count = 0
    for filename in all_pkl_files:
        method, test_func_id, timestamp = parse_filename(filename)
        if method != 'na' and test_func_id != 'na' and timestamp is not None:
            filepath = os.path.join(RESULTS_DIR, filename)
            files_by_experiment[(method, test_func_id)].append((timestamp, filepath))
            parsed_count += 1
        
    log.info(f"Successfully parsed {parsed_count} files to identify experiments.")

    latest_files_to_process = []
    for experiment_key, file_list in files_by_experiment.items():
        if not file_list:
            continue
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
        filename = os.path.basename(filepath) 
        method, test_func_id, _ = parse_filename(filename) 

        if method == 'na' or test_func_id == 'na': 
             log.error(f"Error re-parsing selected file: {filename}. This should not happen.")
             continue

        try:
            with open(filepath, 'rb') as f:
                all_trials_data = pickle.load(f)
            log.debug(f"Successfully loaded {filepath} ({method} on {test_func_id}). Contains {len(all_trials_data)} trials.")

            trial_curves = [] 
            current_max_len_for_file = 0

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

                if len(cumulative_max_y) > current_max_len_for_file:
                    current_max_len_for_file = len(cumulative_max_y)
            
            if trial_curves:
                 results_by_function[test_func_id][method].extend(trial_curves) 
                 if current_max_len_for_file > max_len_per_function[test_func_id]:
                      max_len_per_function[test_func_id] = current_max_len_for_file
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

    # --- Separate Ackley studies from other studies ---
    ackley_studies_data = defaultdict(lambda: defaultdict(list))
    other_studies_data = defaultdict(lambda: defaultdict(list))

    for test_func_id, methods_data in results_by_function.items():
        if test_func_id in ACKLEY_DIMS_ORDERED:
            ackley_studies_data[test_func_id] = methods_data
        else:
            other_studies_data[test_func_id] = methods_data
    
    # --- 3a. Plot Ackley Dimension Study ---
    if ackley_studies_data:
        plot_ackley_dimension_study(ackley_studies_data, max_len_per_function, METHOD_COLORS, FIGURES_DIR)
    else:
        log.info("No Ackley dimension study data found to plot.")

    # --- 3b. Plotting for Other Studies ---
    log.info("Generating plots for other studies...")
    plot_count = 0
    if not other_studies_data:
        log.info("No data for other studies to plot.")
    
    for test_func_id, methods_data_plot in other_studies_data.items():
        log.info(f"Plotting for test function: {test_func_id}")
        plt.figure(figsize=(12, 8))

        target_len_plot = max_len_per_function[test_func_id]
        if target_len_plot == 0:
            log.warning(f"  No valid data length found for {test_func_id}, skipping plot.")
            plt.close() # Close the figure if skipping
            continue

        iterations = np.arange(target_len_plot)
        
        # Pre-calculate means and determine alignment baseline for this plot
        plot_info_list, alignment_baseline_y = get_plotting_data_for_group(methods_data_plot, target_len_plot)

        if not plot_info_list:
            log.warning(f"No plottable data for {test_func_id} after processing. Skipping plot.")
            plt.title(f"Comparison of Methods on {test_func_id.replace('_', ' ').title()}\n(No plottable data)")
            plt.xlabel("Iteration (incl. initial points)")
            plt.ylabel(Y_AXIS_LABEL)
            # Save or close empty plot appropriately
            safe_test_func_id = re.sub(r'[^\w\-_.]', '_', test_func_id)
            plot_filename = f"comparison_{safe_test_func_id}_nodata.pdf" # Indicate no data
            plot_filepath = os.path.join(FIGURES_DIR, plot_filename)
            try:
                plt.savefig(plot_filepath)
                log.info(f"  Saved empty plot placeholder: {plot_filepath}")
            except Exception as e:
                log.error(f"  Failed to save empty plot {plot_filepath}: {e}")
            plt.close()
            continue


        for item in plot_info_list:
            method_name = item['name']
            original_mean_curve = item['original_mean']
            stderr_curve = item['stderr']
            num_trials = item['trials']
            
            displayed_mean_curve = original_mean_curve # Default
            if not np.all(np.isnan(original_mean_curve)) and \
               not np.isnan(original_mean_curve[0]) and \
               not np.isnan(alignment_baseline_y):
                shift = alignment_baseline_y - original_mean_curve[0]
                displayed_mean_curve = original_mean_curve + shift
            
            current_method_color = METHOD_COLORS.get(method_name, DEFAULT_PLOT_COLOR)
            plt.plot(iterations, displayed_mean_curve, label=f"{method_name.upper()} (Trials: {num_trials})", color=current_method_color)
            plt.fill_between(iterations, displayed_mean_curve - stderr_curve, displayed_mean_curve + stderr_curve, alpha=0.2, color=current_method_color)

        plt.title(f"Comparison of Methods on {test_func_id.replace('_', ' ').title()}\n"
                  f"Mean curves vertically aligned to the lowest initial mean for this plot.")
        plt.xlabel("Iteration (incl. initial points)")
        plt.ylabel(Y_AXIS_LABEL)
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

    log.info(f"Generated {plot_count} plots for other studies.")
    log.info("Visualization script finished.")

if __name__ == "__main__":
    main()