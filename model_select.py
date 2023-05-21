import json
from transformers import pipeline, AutoTokenizer, logging
from tqdm import tqdm
from collections import defaultdict
import argparse
from file_io import read_lm_kbc_jsonl, write_lm_kbc_jsonl
from baseline import create_prompt
from evaluate import evaluate_per_sr_pair

logging.set_verbosity_error()  # avoid irritating transformers warnings

def main(args):
    # For each model
    best_model_per_relation = defaultdict(lambda: {'model': None, 'f1': 0})

    for model_name in args.model_names:
        print(f"Processing with model: {model_name}")
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        mask_token = tokenizer.mask_token

        # Initialize pipeline
        pipe = pipeline(args.task, model=model_name, tokenizer=tokenizer)

        # Load development data
        dev_data = read_lm_kbc_jsonl(args.dev_data)

        # Create prompts for development data
        dev_prompts = [create_prompt(item["SubjectEntity"], item["Relation"], mask_token) for item in dev_data]

        # Probe the model with the development prompts
        dev_outputs = []
        for out in tqdm(pipe(dev_prompts, batch_size=8), total=len(dev_prompts)):
            dev_outputs.append(out)

        # Save predictions
        dev_predictions = []
        for row, prompt, output in zip(dev_data, dev_prompts, dev_outputs):
            dev_predictions.append({
                "SubjectEntity": row["SubjectEntity"],
                "Relation": row["Relation"],
                "Prompt": prompt,
                "ObjectEntities": [seq["token_str"] for seq in output if seq["score"] > 0.5],  # replace 0.5 with your threshold
            })

        write_lm_kbc_jsonl(dev_predictions, args.dev_pred)

        # Evaluate the model's predictions and calculate the F1 score for each relation
        for i, item in enumerate(dev_data):
            predictions_fp = {'SubjectEntity': item["SubjectEntity"], 'Relation': item["Relation"], 'ObjectEntities': dev_predictions[i]}
            scores_per_sr_pair = evaluate_per_sr_pair(args.dev_pred, args.dev_data)

            for relation_scores in scores_per_sr_pair:
                relation = relation_scores['Relation']
                f1 = relation_scores['f1']

                # Check if this model has a better F1 score for this relation
                if f1 > best_model_per_relation[relation]['f1']:
                    best_model_per_relation[relation] = {'model': model_name, 'f1': f1}

    # Print the best model for each relation
    for relation, model_f1 in best_model_per_relation.items():
        print(f"Best model for relation {relation}: {model_f1['model']} with F1 score of {model_f1['f1']}")

    # Calculate and print the average F1 score over all relations
    average_f1_score = sum(model_f1['f1'] for model_f1 in best_model_per_relation.values()) / len(best_model_per_relation)
    print(f"Average F1 score across all relations: {average_f1_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dev_data', type=str, required=True, help='Path to the development data')
    parser.add_argument('-o', '--dev_pred', type=str, required=True, help='Path to the predictions output')
    parser.add_argument('-m', '--model_names', nargs='+', required=True, help='List of model names')
    args = parser.parse_args()
    main(args)
