import json
from transformers import pipeline, AutoTokenizer, logging
from tqdm import tqdm
from collections import defaultdict
import argparse
from file_io import read_lm_kbc_jsonl, write_lm_kbc_jsonl
from baseline import create_prompt
from evaluate import evaluate_per_sr_pair, combine_scores_per_relation

logging.set_verbosity_error()  # avoid irritating transformers warnings

def main(args):
    best_model_per_relation = defaultdict(lambda: {'model': None, 'f1': 0})

    for model_name in args.model_names:
        print(f"Processing with model: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        mask_token = tokenizer.mask_token

        pipe = pipeline(args.task, model=model_name, tokenizer=tokenizer)

        dev_data = read_lm_kbc_jsonl(args.dev_data)

        dev_prompts = [create_prompt(item["SubjectEntity"], item["Relation"], mask_token) for item in dev_data]

        # Probe the model with the development prompts
        dev_outputs = []
        for out in tqdm(pipe(dev_prompts, batch_size=32), total=len(dev_prompts)):
            dev_outputs.append(out)

        dev_predictions = []
        for row, prompt, output in zip(dev_data, dev_prompts, dev_outputs):
            dev_predictions.append({
                "SubjectEntity": row["SubjectEntity"],
                "Relation": row["Relation"],
                "Prompt": prompt,
                "ObjectEntities": [seq["token_str"] for seq in output if seq["score"] > 0.5],  # replace 0.5 with your threshold
            })

        write_lm_kbc_jsonl(dev_predictions, args.dev_pred)

    # Evaluate the predictions
    scores_per_sr_pair = evaluate_per_sr_pair(args.dev_pred, args.dev_data)
    scores_per_relation = combine_scores_per_relation(scores_per_sr_pair)

    # Update best model for each relation
    for relation, scores in scores_per_relation.items():
        if scores['f1'] > best_model_per_relation[relation]['f1']:
            best_model_per_relation[relation] = {'model': model_name, 'f1': scores['f1']}

    # Print the model with the highest F1 score for each relation
    for relation, model_info in best_model_per_relation.items():
        print(f"For relation '{relation}', the model with the highest F1 score is {model_info['model']} with F1 score {model_info['f1']}")


    average_f1_score = sum(model_f1['f1'] for model_f1 in best_model_per_relation.values()) / len(best_model_per_relation)
    print(f"Average F1 score across all relations: {average_f1_score}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--dev_data', type=str, required=True, help='Path to the development data')
    parser.add_argument('-o', '--dev_pred', type=str, required=True, help='Path to the predictions output')
    parser.add_argument('-m', '--model_names', nargs='+', required=True, help='List of model names')
    args = parser.parse_args()
    main(args)
