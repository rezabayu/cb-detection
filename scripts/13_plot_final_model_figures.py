import warnings

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve

from _config import METRIC_DIR, FIGURE_DIR

warnings.filterwarnings("ignore")


FINAL_MODEL_FILE = METRIC_DIR / "final_model_comparison_selected.csv"


PREDICTION_FILES = {
    "Dummy Stratified": METRIC_DIR / "baseline_dummy_stratified_test_predictions.csv",
    "Logistic Regression": METRIC_DIR / "baseline_logistic_regression_test_predictions.csv",
    "Random Forest": METRIC_DIR / "baseline_random_forest_test_predictions.csv",
    "DNN": METRIC_DIR / "dnn_test_predictions.csv",
    "1D-CNN_w7": METRIC_DIR / "cnn1d_w7_test_predictions.csv",
    "TCN_w5": METRIC_DIR / "tcn_w5_test_predictions.csv",
}


DISPLAY_NAMES = {
    "Dummy Stratified": "Dummy Stratified",
    "Logistic Regression": "Logistic Regression",
    "Random Forest": "Random Forest",
    "DNN": "DNN",
    "1D-CNN_w7": "1D-CNN W7",
    "TCN_w5": "TCN W5",
}


def safe_name(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("-", "")
        .replace("/", "_")
    )


def plot_confusion_matrix_final(y_true, y_prob, model_name, threshold):
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()

    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - {DISPLAY_NAMES.get(model_name, model_name)}\nThreshold = {threshold:.2f}")

    plt.xticks([0, 1], ["Pred 0", "Pred 1"])
    plt.yticks([0, 1], ["True 0", "True 1"])

    for i in range(2):
        for j in range(2):
            plt.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=14,
                color="black",
            )

    plt.colorbar()
    plt.tight_layout()

    output_file = FIGURE_DIR / f"final_confusion_matrix_{safe_name(model_name)}.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    return output_file, tn, fp, fn, tp


def plot_roc_curve_final(y_true, y_prob, model_name):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")

    plt.title(f"ROC Curve - {DISPLAY_NAMES.get(model_name, model_name)}")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate / POD")
    plt.legend()
    plt.tight_layout()

    output_file = FIGURE_DIR / f"final_roc_curve_{safe_name(model_name)}.png"
    plt.savefig(output_file, dpi=300)
    plt.close()

    return output_file, auc


def main():
    print("=" * 80)
    print("PLOT FINAL MODEL FIGURES USING VALIDATION-SELECTED THRESHOLDS")
    print("=" * 80)

    final_df = pd.read_csv(FINAL_MODEL_FILE)

    summary_rows = []

    for _, row in final_df.iterrows():
        model_name = row["model"]
        threshold = float(row["threshold"])

        pred_file = PREDICTION_FILES.get(model_name)

        if pred_file is None:
            print(f"SKIP: no prediction file mapping for {model_name}")
            continue

        if not pred_file.exists():
            raise FileNotFoundError(f"Prediction file not found: {pred_file}")

        pred_df = pd.read_csv(pred_file)

        y_true = pred_df["y_true"].astype(int).to_numpy()
        y_prob = pred_df["y_prob"].astype(float).to_numpy()

        cm_file, tn, fp, fn, tp = plot_confusion_matrix_final(
            y_true=y_true,
            y_prob=y_prob,
            model_name=model_name,
            threshold=threshold,
        )

        roc_file, auc = plot_roc_curve_final(
            y_true=y_true,
            y_prob=y_prob,
            model_name=model_name,
        )

        summary_rows.append(
            {
                "model": model_name,
                "threshold": threshold,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
                "auc": auc,
                "confusion_matrix_file": str(cm_file),
                "roc_curve_file": str(roc_file),
            }
        )

        print(f"\n{model_name}")
        print(f"Threshold: {threshold:.2f}")
        print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        print("Saved:", cm_file)
        print("Saved:", roc_file)

    summary_df = pd.DataFrame(summary_rows)

    output_csv = METRIC_DIR / "final_model_figures_summary.csv"
    output_xlsx = METRIC_DIR / "final_model_figures_summary.xlsx"

    summary_df.to_csv(output_csv, index=False)
    summary_df.to_excel(output_xlsx, index=False)

    print("\nSaved:")
    print(output_csv)
    print(output_xlsx)


if __name__ == "__main__":
    main()