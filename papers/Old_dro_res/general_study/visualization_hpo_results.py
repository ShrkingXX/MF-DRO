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
RESULTS_DIR = "."  # As per your script

# Directory where the generated plots will be saved
FIGURES_DIR = "."

# Font size for plots
fontsize = 24

# --- Method and Objective Definitions ---
KNOWN_METHODS = ["bo", "dro", "turbo", 'pfns4bo', 'scorebo']
METHOD_LABELS = {"bo":"BO", "dro":"DRO", "turbo":"TuRBO", 'pfns4bo': "PFNs4BO", 'scorebo': "SCoreBO"}
KNOWN_OBJECIVES = ["ackley", # Note: "ackley" without dim
                   'xgboost',
                   'lunarlander',
                   "adam_wine_acc",
                   "adam_breast_acc",
                   "adam_iris_acc",
                   'adam_wine_acc',]

# Objectives for the new subplot group
REGRET_SUBPLOT_OBJECTIVES = ['xgboost', 'adam_breast_acc', 'adam_iris_acc', 'adam_wine_acc']
REGRET_SUBPLOT_FILENAME = "comparison_hpo_regret_studies.pdf"


# --- Color Mapping for Methods ---
METHOD_COLORS = {
    "bo": "#1f77b4",      # Blue
    "dro": "#d62728",     # Red
    "turbo": "#2ca02c",   # Green
    "pfns4bo": "#9467bd", # Purple
    'scorebo': "#ff7f0e",  # Orange
}
DEFAULT_PLOT_COLOR = "#7f7f7f" # Gray
Y_AXIS_LABEL_ORIGINAL = "Best Objective Value Found"
Y_AXIS_LABEL_REGRET = "Simple Regret"


# --- Helper Functions ---
def parse_filename(filename):
    method_name_found, test_function_id_found = 'na', 'na'
    timestamp_str = None
    match = re.match(r"^(.*?)_(\d{8}_\d{6})_all_trials\.pkl$", filename)
    if not match:
        log.debug(f"Filename {filename} does not match expected timestamp pattern. Skipping.")
        return 'na', 'na', None
    core_part = match.group(1)
    timestamp_str = match.group(2)
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
    if method_name_found == 'na' or test_function_id_found == 'na':
        # log.warning(f"Could not determine method or test function from core part '{core_part}' of filename: {filename}. Skipping.")
        return 'na', 'na', None
    if method_name_found not in core_lower or test_function_id_found not in core_lower:
        #  log.warning(f"Potential mismatch in parsing '{core_part}'. Found method '{method_name_found}', objective '{test_function_id_found}'. Skipping {filename}.")
         return 'na', 'na', None
    return method_name_found, test_function_id_found, timestamp_str


def get_method_plot_details(curves_list, target_len):
    """Calculates mean, stderr from a list of curves for a single method."""
    if not curves_list:
        return None, None, 0

    padded_curves = []
    for curve in curves_list:
        if len(curve) < target_len:
            padding_val = curve[-1] if len(curve) > 0 else np.nan
            padding = np.full(target_len - len(curve), padding_val)
            padded_curve = np.concatenate((curve, padding))
        elif len(curve) > target_len:
            padded_curve = curve[:target_len]
        else:
            padded_curve = curve
        padded_curves.append(padded_curve)

    if not padded_curves: return None, None, 0
    curves_array = np.array(padded_curves)
    valid_trials_mask = ~np.all(np.isnan(curves_array), axis=1)
    if not np.any(valid_trials_mask): return None, None, 0
    curves_array = curves_array[valid_trials_mask]

    original_mean_curve = np.nanmean(curves_array, axis=0)
    counts_per_iteration = np.sum(~np.isnan(curves_array), axis=0)
    std_dev_curve = np.nanstd(curves_array, axis=0)
    stderr_curve = np.full_like(original_mean_curve, np.nan)
    valid_counts_mask = counts_per_iteration > 0
    stderr_curve[valid_counts_mask] = std_dev_curve[valid_counts_mask] / np.sqrt(counts_per_iteration[valid_counts_mask])
    
    return original_mean_curve, stderr_curve, curves_array.shape[0]


def plot_selected_subplot_figure(subplot_data, subplot_objectives_order, filename_suffix,
                                 max_len_per_function, method_colors, figures_dir, is_regret_plot=False):
    log.info(f"Generating subplot figure for: {subplot_objectives_order}")
    num_subplots = len(subplot_objectives_order)
    # Using sharey=False because regret plots use log scale and might have very different ranges
    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5.5), sharey=False)
    if num_subplots == 1: axes = [axes]

    legend_handles = {}
    all_method_names_in_figure = set()
    for test_func_id in subplot_data:
        for method_name in subplot_data[test_func_id]:
            all_method_names_in_figure.add(method_name)
    sorted_all_method_names = sorted(list(all_method_names_in_figure))

    for i, test_func_id in enumerate(subplot_objectives_order):
        ax = axes[i]
        methods_data_for_subplot = subplot_data.get(test_func_id)
        target_len_subplot = max_len_per_function.get(test_func_id, 0)
        
        
        if 'acc' in test_func_id:
            ax.set_title(f"{test_func_id.replace('_', ' ').title()[:-3]}", fontsize=fontsize)
        else:
            ax.set_title(f"{test_func_id.replace('_', ' ').title()}", fontsize=fontsize)
        ax.set_xlabel("Iteration", fontsize=fontsize)
        ax.grid(True, which="both", linestyle='--', linewidth=0.5)

        if not methods_data_for_subplot or target_len_subplot == 0:
            log.warning(f"No data or zero target length for {test_func_id}. Subplot empty.")
            ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
            if is_regret_plot: ax.set_ylabel(Y_AXIS_LABEL_REGRET, fontsize=fontsize)
            else: ax.set_ylabel(Y_AXIS_LABEL_ORIGINAL, fontsize=fontsize)
            continue

        iterations = np.arange(target_len_subplot)
        
        # Store details for alignment pass
        subplot_method_details = []
        sorted_method_names_subplot = sorted(methods_data_for_subplot.keys())

        for method_name in sorted_method_names_subplot:
            curves_list = methods_data_for_subplot[method_name]
            original_mean, stderr, num_trials = get_method_plot_details(curves_list, target_len_subplot)
            
            if original_mean is None: continue

            plotted_y_values = original_mean
            if is_regret_plot:
                # For regret, the transformation is 1 - mean (assuming mean is accuracy/objective)
                plotted_y_values = 1 - original_mean
            
            subplot_method_details.append({
                'name': method_name,
                'plotted_y_values': plotted_y_values, # These are the values (original or transformed) before alignment
                'stderr': stderr, # Stderr magnitude is same for X and 1-X
                'trials': num_trials
            })
        
        if not subplot_method_details:
            log.warning(f"No valid method details for {test_func_id}. Subplot empty.")
            ax.text(0.5, 0.5, "Processing error", ha='center', va='center', transform=ax.transAxes)
            if is_regret_plot: ax.set_ylabel(Y_AXIS_LABEL_REGRET, fontsize=fontsize)
            else: ax.set_ylabel(Y_AXIS_LABEL_ORIGINAL, fontsize=fontsize)
            continue

        # Determine alignment baseline for this subplot based on initial points of plotted_y_values
        initial_plotted_values_for_alignment = []
        for item in subplot_method_details:
            if not np.all(np.isnan(item['plotted_y_values'])) and not np.isnan(item['plotted_y_values'][0]):
                initial_plotted_values_for_alignment.append(item['plotted_y_values'][0])
        
        alignment_baseline_y = np.nanmin(initial_plotted_values_for_alignment) if initial_plotted_values_for_alignment else np.nan

        # Second pass for plotting with alignment
        for item in subplot_method_details:
            method_name = item['name']
            plotted_y_values_unaligned = item['plotted_y_values']
            stderr_curve = item['stderr']
            
            displayed_y_values = plotted_y_values_unaligned # Default
            if not np.all(np.isnan(plotted_y_values_unaligned)) and \
               not np.isnan(plotted_y_values_unaligned[0]) and \
               not np.isnan(alignment_baseline_y):
                shift = alignment_baseline_y - plotted_y_values_unaligned[0]
                displayed_y_values = plotted_y_values_unaligned + shift
            
            current_method_color = method_colors.get(method_name, DEFAULT_PLOT_COLOR)
            
            xlim_val = None
            if is_regret_plot:
                xlim_val = 30 
                ax.set_ylabel(Y_AXIS_LABEL_REGRET, fontsize=fontsize)
                ax.set_yscale("log")
            else: # For other generic subplots if this function is reused
                ax.set_ylabel(Y_AXIS_LABEL_ORIGINAL, fontsize=fontsize)


            plot_iterations = iterations
            plot_displayed_y = displayed_y_values
            plot_stderr = stderr_curve

            if xlim_val and target_len_subplot > xlim_val : # Ensure xlim doesn't exceed data length
                plot_iterations = iterations[:xlim_val]
                plot_displayed_y = displayed_y_values[:xlim_val]
                plot_stderr = stderr_curve[:xlim_val]
            elif xlim_val and target_len_subplot < xlim_val: # xlim is larger than data, use all data
                pass # use full plot_iterations, etc.
            elif xlim_val: # xlim_val == target_len_subplot
                 pass


            line, = ax.plot(plot_iterations, plot_displayed_y, color=current_method_color, label=METHOD_LABELS.get(method_name.lower(), method_name.upper()))
            # Ensure fill_between also respects xlim if applied
            ax.fill_between(plot_iterations, plot_displayed_y - plot_stderr, 
                            plot_displayed_y + plot_stderr, alpha=0.2, color=current_method_color)

            if method_name not in legend_handles:
                 legend_handles[method_name] = line
    
    # Common Legend
    ordered_handles = [legend_handles[m] for m in sorted_all_method_names if m in legend_handles]
    ordered_labels = [METHOD_LABELS.get(m, m.upper()) for m in sorted_all_method_names if m in legend_handles]
    if ordered_handles:
        num_legend_cols = min(len(ordered_handles), 5) # Adjust num_legend_cols
        fig.legend(ordered_handles, ordered_labels, loc='upper center', ncol=num_legend_cols, bbox_to_anchor=(0.5, 1), fontsize=fontsize)
    
    plt.tight_layout(rect=[0, 0, 1, .8]) # rect=[left, bottom, right, top]

    plot_filename = f"comparison_{filename_suffix}.pdf"
    plot_filepath = os.path.join(figures_dir, plot_filename)
    try:
        plt.savefig(plot_filepath)
        log.info(f"  Saved subplot figure: {plot_filepath}")
    except Exception as e:
        log.error(f"  Failed to save subplot figure {plot_filepath}: {e}")
    plt.close(fig)


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
            current_max_len = 0 # Renamed from current_max_len_for_file

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

                if len(cumulative_max_y) > current_max_len:
                    current_max_len = len(cumulative_max_y)
            
            if trial_curves:
                 results_by_function[test_func_id][method].extend(trial_curves) 
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

    # --- Separate data for different plot groups ---
    regret_subplot_actual_data = defaultdict(lambda: defaultdict(list))
    individual_plot_data = defaultdict(lambda: defaultdict(list)) 

    for test_func_id, methods_data in results_by_function.items():
        if test_func_id in REGRET_SUBPLOT_OBJECTIVES:
            regret_subplot_actual_data[test_func_id] = methods_data
        else:
            individual_plot_data[test_func_id] = methods_data
    
    # --- 3a. Plot Regret Subplot Figure ---
    if regret_subplot_actual_data:
        plot_selected_subplot_figure(regret_subplot_actual_data, REGRET_SUBPLOT_OBJECTIVES, 
                                     "hpo_regret_studies", max_len_per_function, 
                                     METHOD_COLORS, FIGURES_DIR, is_regret_plot=True)
    else:
        log.info(f"No data found for regret subplot objectives: {REGRET_SUBPLOT_OBJECTIVES}")

    # --- 3b. Plotting for Other Studies (Individually) ---
    log.info("Generating individual plots for other studies...")
    plot_count = 0
    if not individual_plot_data:
        log.info("No data for other individual studies to plot.")
    
    for test_func_id, methods_data_for_plot in individual_plot_data.items():
        log.info(f"Plotting individually for test function: {test_func_id}")
        plt.figure(figsize=(12, 8))

        target_len = max_len_per_function[test_func_id]
        if target_len == 0:
            log.warning(f"  No valid data length for {test_func_id}, skipping plot.")
            plt.close()
            continue

        iterations = np.arange(target_len)
        
        # This part is for individual plots, so alignment is within this single plot
        single_plot_method_details = []
        sorted_method_names_plot = sorted(methods_data_for_plot.keys())

        for method_name in sorted_method_names_plot:
            curves_list = methods_data_for_plot[method_name]
            original_mean, stderr, num_trials = get_method_plot_details(curves_list, target_len)
            if original_mean is None: continue
            single_plot_method_details.append({
                'name': method_name, 'original_mean': original_mean, 
                'stderr': stderr, 'trials': num_trials
            })

        if not single_plot_method_details:
            log.warning(f"No plottable method data for {test_func_id}. Skipping.")
            plt.close()
            continue
            
        initial_values_for_alignment = [
            item['original_mean'][0] for item in single_plot_method_details 
            if not np.all(np.isnan(item['original_mean'])) and not np.isnan(item['original_mean'][0])
        ]
        alignment_baseline_y = np.nanmin(initial_values_for_alignment) if initial_values_for_alignment else np.nan

        current_y_label = Y_AXIS_LABEL_ORIGINAL
        apply_log_scale = False
        xlim_val = None

        if 'lunarlander' in test_func_id.lower() : # Example of non-regret special handling
            xlim_val = 200

        for item in single_plot_method_details:
            method_name = item['name']
            original_mean_curve = item['original_mean']
            stderr_curve = item['stderr']
            num_trials = item['trials']

            plotted_y_values = original_mean_curve

            displayed_y_values = plotted_y_values
            if not np.all(np.isnan(plotted_y_values)) and \
               not np.isnan(plotted_y_values[0]) and \
               not np.isnan(alignment_baseline_y):
                shift = alignment_baseline_y - plotted_y_values[0]
                displayed_y_values = plotted_y_values + shift
            
            current_method_color = METHOD_COLORS.get(method_name, DEFAULT_PLOT_COLOR)
            
            # Apply xlim for individual plots if defined
            plot_iterations_indiv = iterations
            plot_displayed_y_indiv = displayed_y_values
            plot_stderr_indiv = stderr_curve

            # The original script had conditional plotting logic here directly modifying mean_curve etc.
            # Re-integrating that for individual plots:
            if 'adam' in test_func_id or ('xgboost' in test_func_id and test_func_id not in REGRET_SUBPLOT_OBJECTIVES) or ('acc' in test_func_id and test_func_id not in REGRET_SUBPLOT_OBJECTIVES):
                xlim_val = 30 
                plot_displayed_y_indiv = 1 - original_mean_curve # Apply transform here for these individual cases
                # Re-align based on this transformed value for these specific cases
                temp_initial_transformed_values = [
                    (1 - itm['original_mean'][0]) for itm in single_plot_method_details 
                     if not np.all(np.isnan(itm['original_mean'])) and not np.isnan(itm['original_mean'][0])
                ]
                temp_alignment_baseline_y = np.nanmin(temp_initial_transformed_values) if temp_initial_transformed_values else np.nan
                
                if not np.all(np.isnan(plot_displayed_y_indiv)) and \
                   not np.isnan(plot_displayed_y_indiv[0]) and \
                   not np.isnan(temp_alignment_baseline_y):
                    shift = temp_alignment_baseline_y - plot_displayed_y_indiv[0]
                    plot_displayed_y_indiv = plot_displayed_y_indiv + shift
                
                current_y_label = Y_AXIS_LABEL_REGRET
                apply_log_scale = True
                plot_title = f"Comparison of Methods on {test_func_id.replace('_', ' ').title()}\nMean curves (simple regret) vertically aligned."


            if xlim_val and target_len > xlim_val:
                plot_iterations_indiv = iterations[:xlim_val]
                plot_displayed_y_indiv = plot_displayed_y_indiv[:xlim_val]
                plot_stderr_indiv = stderr_curve[:xlim_val] # stderr_curve is on original_mean scale
            
            plt.plot(plot_iterations_indiv, plot_displayed_y_indiv, label=f"{METHOD_LABELS.get(method_name, method_name.upper())}", color=current_method_color)
            plt.fill_between(plot_iterations_indiv, plot_displayed_y_indiv - plot_stderr_indiv, 
                             plot_displayed_y_indiv + plot_stderr_indiv, alpha=0.2, color=current_method_color)

        # plt.title(plot_title)
        plt.xlabel("Iteration", fontsize=fontsize)
        plt.ylabel(current_y_label,  fontsize=fontsize)
        if apply_log_scale:
            plt.yscale("log")
        plt.legend(title="Methods",  fontsize=fontsize, title_fontsize=fontsize)
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

    log.info(f"Generated {plot_count} individual plots for other studies.")
    log.info("Visualization script finished.")

if __name__ == "__main__":
    main()