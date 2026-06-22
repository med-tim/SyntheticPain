import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             confusion_matrix, f1_score, roc_curve, auc)
from fairlearn.metrics import MetricFrame, count, selection_rate
from fairlearn.metrics import demographic_parity_difference, demographic_parity_ratio
from fairlearn.metrics import equalized_odds_difference, equalized_odds_ratio
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Input CSV must contain the columns named below: true labels, predicted labels,
# and one column per sensitive attribute (here: 'monk' and 'fitz' skin-tone scales).
CSV_FILE_PATH = 'path/to/fairness_raw_data.csv'
OUTPUT_REPORT_PATH = 'fairness_report_with_visuals.txt'
PLOT_DIR_NAME = "fairness_visualizations_no_prob"

TRUE_LABEL_COL = 'true_pain_level'
PREDICTED_LABEL_COL = 'predicted_pain_level'

SENSITIVE_ATTR_COLS = ['monk', 'fitz']
POSITIVE_LABEL_VALUE = 1
ALL_LABELS = [0, 1]
SMALL_GROUP_THRESHOLD = 5

def log_print(message, file_handle):
    print(message)
    file_handle.write(str(message) + '\n')

def false_positive_rate(y_true, y_pred):
    y_true_arr = pd.Series(y_true).values; y_pred_arr = pd.Series(y_pred).values
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=ALL_LABELS); tn, fp, fn, tp = cm.ravel()
    return 0.0 if (fp + tn) == 0 else fp / (fp + tn)

def true_negative_rate(y_true, y_pred):
    y_true_arr = pd.Series(y_true).values; y_pred_arr = pd.Series(y_pred).values
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=ALL_LABELS); tn, fp, fn, tp = cm.ravel()
    return 0.0 if (tn + fp) == 0 else tn / (tn + fp)

def false_negative_rate(y_true, y_pred): 
    """Calculates the False Negative Rate."""
    y_true_arr = pd.Series(y_true).values
    y_pred_arr = pd.Series(y_pred).values
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=ALL_LABELS)
    tn, fp, fn, tp = cm.ravel()
    denominator = fn + tp
    return 0.0 if denominator == 0 else fn / denominator

try:
    df = pd.read_csv(CSV_FILE_PATH)
    print(f"Successfully loaded data from {CSV_FILE_PATH}")
    print(f"Data shape: {df.shape}")
    required_cols = [TRUE_LABEL_COL, PREDICTED_LABEL_COL] + SENSITIVE_ATTR_COLS
    if not all(col in df.columns for col in required_cols):
        missing = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns for fairness metrics: {missing}")
    
except Exception as e:
    print(f"Error loading or validating CSV: {e}"); exit()

metrics = {
    'count': count, 'accuracy': accuracy_score,
    'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, labels=ALL_LABELS, pos_label=POSITIVE_LABEL_VALUE, zero_division=0),
    'recall (TPR)': lambda y_true, y_pred: recall_score(y_true, y_pred, labels=ALL_LABELS, pos_label=POSITIVE_LABEL_VALUE, zero_division=0),
    'specificity (TNR)': true_negative_rate,
    'fpr': false_positive_rate,
    'fnr': false_negative_rate, 
    'f1_score': lambda y_true, y_pred: f1_score(y_true, y_pred, labels=ALL_LABELS, pos_label=POSITIVE_LABEL_VALUE, zero_division=0),
    'selection_rate': selection_rate
}
y_true = df[TRUE_LABEL_COL]
y_pred = df[PREDICTED_LABEL_COL]

output_base_dir = os.path.dirname(OUTPUT_REPORT_PATH)
plot_dir = os.path.join(output_base_dir, PLOT_DIR_NAME)
if output_base_dir: os.makedirs(output_base_dir, exist_ok=True)
os.makedirs(plot_dir, exist_ok=True)

try:
    with open(OUTPUT_REPORT_PATH, 'w') as outfile:
        log_print("="*80, outfile); log_print("Fairness Evaluation Report (Binary Predictions Focus)", outfile); log_print("="*80, outfile)
        log_print(f"Data Source: {os.path.basename(CSV_FILE_PATH)}", outfile)
        log_print(f"Report Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", outfile)
        log_print(f"Plots saved to: {plot_dir}", outfile)
        log_print("="*80, outfile)

        log_print("\n\n--- Overall Model Performance ---", outfile)
        log_print("Skipping Overall ROC AUC chart: Analysis based on binary predictions.", outfile)

        all_disparity_summaries = []

        for sensitive_col in SENSITIVE_ATTR_COLS:
            log_print(f"\n\n{'#'*20} Fairness Metrics for Sensitive Attribute: '{sensitive_col}' {'#'*20}\n", outfile)
            sensitive_features = df[sensitive_col]
            group_counts = sensitive_features.value_counts().sort_index()

            try:
                grouped_on_sens = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive_features)
                by_group_df = grouped_on_sens.by_group.copy()
                overall_metrics_vals = grouped_on_sens.overall

                log_print("\nMetrics By Group:", outfile); outfile.write(by_group_df.to_string() + '\n')
                log_print("\nOverall Metrics:", outfile); outfile.write(overall_metrics_vals.to_string() + '\n')

                current_disparities_summary = {'sensitive_attribute': sensitive_col}
                if len(group_counts) > 1:
                    diff_metrics = grouped_on_sens.difference(method='between_groups')
                    log_print("\nDisparity Metrics (Difference: Max - Min):", outfile); outfile.write(diff_metrics.to_string() + '\n')

                    stat_parity_diff = demographic_parity_difference(y_true, y_pred, sensitive_features=sensitive_features)
                    current_disparities_summary['Statistical Parity Diff.'] = stat_parity_diff
                    log_print(f"  - Stat. Parity Diff.: {stat_parity_diff:.4f}", outfile)

                    tpr_diff = diff_metrics.get('recall (TPR)', np.nan)
                    current_disparities_summary['Equal Opportunity Diff. (TPR)'] = tpr_diff
                    log_print(f"  - Equal Opp. Diff. (TPR): {tpr_diff:.4f}", outfile)

                    fpr_diff = diff_metrics.get('fpr', np.nan)
                    current_disparities_summary['Predictive Equality Diff. (FPR)'] = fpr_diff
                    log_print(f"  - Pred. Equality Diff. (FPR): {fpr_diff:.4f}", outfile)
                    
                    fnr_diff = diff_metrics.get('fnr', np.nan) 
                    current_disparities_summary['False Negative Rate Diff.'] = fnr_diff

                    if not (np.isnan(tpr_diff) or np.isnan(fpr_diff)):
                        eq_odds_diff_manual = max(abs(tpr_diff), abs(fpr_diff))
                        current_disparities_summary['Equalized Odds Diff. (Max|TPR,FPR|)'] = eq_odds_diff_manual
                        log_print(f"  - Equalized Odds Diff. (Max): {eq_odds_diff_manual:.4f}", outfile)
                    all_disparity_summaries.append(current_disparities_summary)
                else:
                    log_print("\nOnly one group found. Disparity metrics not applicable.", outfile)

          
                metrics_to_plot_bar = { 
                    'recall (TPR)': {'name': 'Recall (True Positive Rate)', 'ylim': [0, 1.05]},
                    'fpr': {'name': 'False Positive Rate', 'ylim': [0, 1.05]},
                    'fnr': {'name': 'False Negative Rate', 'ylim': [0, 1.05]},
                    'selection_rate': {'name': 'Selection Rate (P(Pred=1))', 'ylim': [0, 1.05]},
                    'accuracy': {'name': 'Accuracy', 'ylim': [0, 1.05]},
                }
                group_counts_series = by_group_df['count']

                if sensitive_col == 'monk':
                    plot_order_str = [str(x) for x in sorted(by_group_df.index.astype(int).unique())]
                    by_group_df.index = by_group_df.index.astype(str)
                    group_counts_series.index = group_counts_series.index.astype(str)
                else:
                    plot_order_str = sorted(by_group_df.index.astype(str).unique())
                    by_group_df.index = by_group_df.index.astype(str)
                    group_counts_series.index = group_counts_series.index.astype(str)

                for metric_key, plot_info in metrics_to_plot_bar.items():
                    if metric_key not in by_group_df: continue
                    metric_name = plot_info['name']; y_limit = plot_info['ylim']
                    plot_filename = f"barchart_{metric_key.replace(' ', '').replace('(', '').replace(')', '')}_vs_{sensitive_col}.png"
                    plot_filepath = os.path.join(plot_dir, plot_filename)
                    plt.figure(figsize=(max(8, len(plot_order_str) * 0.8), 6))
                    plot_data = by_group_df[[metric_key]].reset_index().rename(columns={'index': sensitive_col})
                    bar_plot = sns.barplot(x=sensitive_col, y=metric_key, data=plot_data, palette="viridis", order=plot_order_str)
                    if metric_key in overall_metrics_vals:
                        overall_val = overall_metrics_vals[metric_key]
                        plt.axhline(overall_val, color='r', linestyle='--', label=f'Overall ({overall_val:.3f})')
                    for i, bar_group_name in enumerate(plot_order_str):
                        bar = bar_plot.patches[i]
                        count_val = group_counts_series.get(bar_group_name, 'N/A')
                        height = bar.get_height()
                        plt.text(bar.get_x() + bar.get_width() / 2., height + 0.01, f'n={count_val}', ha='center', va='bottom', fontsize=9)
                    plt.title(f"{metric_name} by {sensitive_col.capitalize()}", fontsize=16)
                    plt.xlabel(sensitive_col.capitalize(), fontsize=12); plt.ylabel(metric_name, fontsize=12)
                    plt.ylim(y_limit); plt.xticks(rotation=45, ha='right')
                    plt.legend(); plt.grid(axis='y', linestyle='--', alpha=0.7); plt.tight_layout()
                    plt.savefig(plot_filepath); plt.close()

                metrics_for_heatmap = ['accuracy', 'recall (TPR)', 'fpr', 'fnr', 'precision', 'selection_rate']
                heatmap_data = by_group_df[metrics_for_heatmap].copy()
                if sensitive_col == 'monk':
                     heatmap_data = heatmap_data.reindex(plot_order_str)


                plt.figure(figsize=(max(10, len(heatmap_data.columns) * 1.5) , max(6, len(heatmap_data.index) * 0.5)))
                sns.heatmap(heatmap_data.T, annot=True, fmt=".3f", cmap="viridis_r", linewidths=.5) 
                plt.title(f'Heatmap of Metrics by {sensitive_col.capitalize()} Group', fontsize=16)
                plt.xlabel(sensitive_col.capitalize() + " Group", fontsize=12)
                plt.ylabel("Metric", fontsize=12)
                plt.xticks(rotation=45, ha="right")
                plt.yticks(rotation=0)
                plt.tight_layout()
                heatmap_plot_path = os.path.join(plot_dir, f"heatmap_metrics_vs_{sensitive_col}.png")
                plt.savefig(heatmap_plot_path); plt.close()
                log_print(f"Metrics heatmap saved to: {heatmap_plot_path}", outfile)


                log_print(f"\nVisualizing Confusion Matrices for top groups in '{sensitive_col}':", outfile)
                significant_groups = group_counts[group_counts >= SMALL_GROUP_THRESHOLD].index.astype(str)
                if len(significant_groups) > 5 : 
                    log_print(f"  (Showing top 5 by count due to many significant groups)", outfile)
                    significant_groups = group_counts.nlargest(5).index.astype(str)

                if not significant_groups.empty:
                    num_groups_to_plot = len(significant_groups)
                    ncols = int(np.ceil(np.sqrt(num_groups_to_plot)))
                    nrows = int(np.ceil(num_groups_to_plot / ncols))

                    fig_cm, axes_cm = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4.5), squeeze=False)
                    axes_cm = axes_cm.flatten() 

                    for i, group_val in enumerate(significant_groups):
                        group_df = df[df[sensitive_col].astype(str) == group_val]
                        if group_df.empty:
                            log_print(f"  Skipping CM for group '{group_val}': No data after filtering (this shouldn't happen if group_counts is accurate).", outfile)
                            if i < len(axes_cm): axes_cm[i].set_visible(False) 

                        cm_group = confusion_matrix(group_df[TRUE_LABEL_COL], group_df[PREDICTED_LABEL_COL], labels=ALL_LABELS)

                        sns.heatmap(cm_group, annot=True, fmt='d', cmap='Blues', ax=axes_cm[i], cbar=False,
                                    xticklabels=['Predicted No Pain', 'Predicted Pain'],
                                    yticklabels=['Actual No Pain', 'Actual Pain'])
                        axes_cm[i].set_title(f'{sensitive_col.capitalize()} = {group_val} (n={len(group_df)})', fontsize=12)
                        axes_cm[i].set_ylabel('Actual Label')
                        axes_cm[i].set_xlabel('Predicted Label')

                    for j in range(num_groups_to_plot, len(axes_cm)):
                        axes_cm[j].set_visible(False)

                    fig_cm.suptitle(f'Confusion Matrices by {sensitive_col.capitalize()} Group', fontsize=16, y=1.02 if nrows > 1 else 1.05)
                    fig_cm.tight_layout(rect=[0, 0, 1, 0.96 if nrows > 1 else 0.95])
                    cm_plot_path = os.path.join(plot_dir, f"confusion_matrices_by_{sensitive_col}.png")
                    plt.savefig(cm_plot_path); plt.close(fig_cm)
                    log_print(f"Per-group confusion matrix plot saved to: {cm_plot_path}", outfile)
                else:
                    log_print(f"  No groups in '{sensitive_col}' met the threshold for individual CM visualization.", outfile)


            except Exception as e_inner:
                log_print(f"\n!!!!!! Error processing attribute '{sensitive_col}' !!!!!!\n{e_inner}", outfile)
            log_print(f"\n{'#'*20} End Metrics for '{sensitive_col}' {'#'*20}", outfile)

        if all_disparity_summaries:
            summary_df = pd.DataFrame(all_disparity_summaries).set_index('sensitive_attribute')
            summary_df = summary_df.melt(ignore_index=False, var_name='Disparity Metric', value_name='Max Difference').reset_index()
            plt.figure(figsize=(10, max(6, len(summary_df) * 0.3)))
            sns.barplot(x='Max Difference', y='Disparity Metric', hue='sensitive_attribute', data=summary_df, palette="muted")
            plt.title('Summary of Fairness Disparity Metrics', fontsize=16)
            plt.xlabel('Maximum Difference Between Groups', fontsize=12)
            plt.ylabel(''); plt.axvline(0, color='grey', linestyle='-', lw=0.7)
            plt.legend(title='Sensitive Attribute', loc='center left', bbox_to_anchor=(1, 0.5))
            plt.tight_layout(rect=[0, 0, 0.85, 1])
            summary_plot_path = os.path.join(plot_dir, "summary_disparity_metrics.png")
            plt.savefig(summary_plot_path); plt.close()
            log_print(f"\nSummary disparity plot saved to: {summary_plot_path}", outfile)

        log_print("\n\n" + "="*80, outfile); log_print("--- Fairness Evaluation Script Finished ---", outfile); log_print("="*80, outfile)

    print(f"\n\n--- Fairness Evaluation Script Finished ---")
    print(f"Full text report saved to: {OUTPUT_REPORT_PATH}")
    print(f"Visualizations saved to folder: {plot_dir}")

except Exception as e_outer:
    print(f"\n!!!!!! A critical error occurred !!!!!!\n{e_outer}")
    import traceback
    traceback.print_exc()