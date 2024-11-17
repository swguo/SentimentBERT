# eval.py

import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_results(inference_results_path, class_names):
    # 讀取測試資料和推論結果
    df_results = pd.read_csv(inference_results_path)
    
    # 計算混淆矩陣
    cm = confusion_matrix(df_results['targets'], df_results['prediction'])
    print("Confusion Matrix:")
    print(cm)

    # 繪製混淆矩陣
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.title("Confusion Matrix")
    plt.show()

    # 顯示分類報告
    report = classification_report(df_results['targets'], df_results['prediction'], target_names=class_names)
    print("Classification Report:")
    print(report)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate inference results.")    
    parser.add_argument('--inference_results_path', type=str, required=True, help="Path to CSV with inference results.")
    parser.add_argument('--class_names', nargs='+', default=['negative', 'neutral', 'positive'], help="List of class names.")

    args = parser.parse_args()
    evaluate_results(
        inference_results_path=args.inference_results_path,
        class_names=args.class_names
    )
