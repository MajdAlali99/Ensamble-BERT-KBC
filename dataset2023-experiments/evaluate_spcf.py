import sys
import json
import argparse
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from file_io import read_lm_kbc_jsonl_to_df

def evaluate_instances(test_file, predictions_file, key_name):
    test_df = read_lm_kbc_jsonl_to_df(test_file)
    pred_df = read_lm_kbc_jsonl_to_df(predictions_file)

    if len(test_df) != len(pred_df):
        raise ValueError("Test and predictions files have different lengths")

    test_labels = test_df[key_name].tolist()
    pred_labels = pred_df[key_name].tolist()
    print("TEST:\n", test_labels)
    print("TEST:\n", pred_labels)

    return test_labels, pred_labels



def main(args):
    test_labels, pred_labels = evaluate_instances(args.ground_truth, args.predictions, args.key_name)
    precision, recall, f1_score, _ = precision_recall_fscore_support(test_labels, pred_labels, average='weighted')

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1_score:.2f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Precision, Recall and F1-score of predictions")
    parser.add_argument("-p", "--predictions", type=str, required=True, help="Path to the predictions file (required)")
    parser.add_argument("-g", "--ground_truth", type=str, required=True, help="Path to the ground truth file (required)")
    parser.add_argument("-k", "--key_name", type=str, required=True, help="Key name to compare in the JSON objects (required)")

    args = parser.parse_args()
    main(args)