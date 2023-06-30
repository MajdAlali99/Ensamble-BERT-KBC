import argparse
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
import torch
from transformers import BertForMaskedLM, BertTokenizerFast
import json


def read_lm_kbc_jsonl(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()

    rows = []
    for line in lines:
        rows.append(json.loads(line.strip()))

    return rows


def evaluate(gt_rows, pred_rows):
    gt_set = set([(row["SubjectEntityID"], row["Relation"], oid) for row in gt_rows for oid in row["ObjectEntitiesID"]])
    pred_set = set([(row["SubjectEntityID"], row["Relation"], oid) for row in pred_rows for oid in row["ObjectEntitiesID"]])

    precision = precision_score(list(gt_set), list(pred_set), average='micro')
    recall = recall_score(list(gt_set), list(pred_set), average='micro')
    f1 = f1_score(list(gt_set), list(pred_set), average='micro')

    return {"precision": precision, "recall": recall, "f1": f1}


def generate_predictions(model, tokenizer, test_df, output_file):
    model.eval()

    with open(output_file, "w") as out_f:
        for _, row in test_df.iterrows():
            prompt = row["Prompt"]
            masked_prompt = prompt.replace("{mask_token}", tokenizer.mask_token)
            input_ids = tokenizer.encode(masked_prompt, return_tensors="pt")
            mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

            with torch.no_grad():
                outputs = model(input_ids)
                logits = outputs.logits

            softmax_logits = torch.nn.functional.softmax(logits, dim=-1)
            mask_token_logits = softmax_logits[0, mask_token_index, :]
            top_token_ids = torch.topk(mask_token_logits, 5, dim=-1).indices.tolist()

            predicted_tokens = tokenizer.convert_ids_to_tokens(top_token_ids[0])

            # Write the predicted tokens to the output file
            out_f.write(json.dumps({
                "SubjectEntityID": row["SubjectEntityID"],
                "SubjectEntity": row["SubjectEntity"],
                "Relation": row["Relation"],
                "ObjectEntities": predicted_tokens
            }) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned BERT model")
    parser.add_argument("-m", "--model", type=str, required=True, help="Path to the fine-tuned BERT model")
    parser.add_argument("-t", "--test_data", type=str, required=True, help="Path to the test data")
    parser.add_argument("-g", "--ground_truth", type=str, required=True, help="Path to the ground truth file")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output predictions file")
    
    args = parser.parse_args()

    model = BertForMaskedLM.from_pretrained(args.model)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    test_df = pd.read_csv(args.test_data)
    gt_rows = read_lm_kbc_jsonl(args.ground_truth)

    # Generate predictions
    generate_predictions(model, tokenizer, test_df, args.output)

    # Read predictions
    pred_rows = read_lm_kbc_jsonl(args.output)

    # Evaluate predictions
    scores = evaluate(gt_rows, pred_rows)

    print("Precision: {:.3f}".format(scores["precision"]))
    print("Recall: {:.3f}".format(scores["recall"]))
    print("F1-score: {:.3f}".format(scores["f1"]))


if __name__ == "__main__":
    main()
