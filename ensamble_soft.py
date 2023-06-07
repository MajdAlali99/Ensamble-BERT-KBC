import argparse
import json
import logging
from collections import Counter

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

from file_io import read_lm_kbc_jsonl

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class PromptSet(Dataset):
    def __init__(self, prompts):
        self.prompts = prompts

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index) -> T_co:
        return self.prompts[index]


def create_prompt(subject_entity: str, relation: str, mask_token: str) -> str:
    prompt = f"{subject_entity}, {relation}, {mask_token}."
    return prompt

def run(args):
    # Load the models
    models = []
    model_names = args.models.split(',')

    logger.info("Loading the models...")
    for model_name in model_names:
        tokenizer = AutoTokenizer.from_pretrained('bert-large-cased')
        model = AutoModelForMaskedLM.from_pretrained(model_name)
        models.append((tokenizer, model))

    mask_token = models[0][0].mask_token

    # Load the input file
    logger.info(f"Loading the input file \"{args.input}\"...")
    input_rows = read_lm_kbc_jsonl(args.input)
    logger.info(f"Loaded {len(input_rows):,} rows.")

    # Create prompts
    logger.info(f"Creating prompts...")
    prompts = PromptSet([create_prompt(
        subject_entity=row["SubjectEntity"],
        relation=row["Relation"],
        mask_token=mask_token,
    ) for row in input_rows])

    # Run the ensemble model
    logger.info("Running the ensemble model...")
    results = []
    for row, prompt in zip(input_rows, prompts):
        ensemble_predictions = []
        for model in models:
            tokenizer, model = model
            pipe = pipeline(
                task="fill-mask",
                model=model,
                tokenizer=tokenizer,
                top_k=args.top_k,
                device=args.gpu
            )
            output = pipe(prompt)
            ensemble_predictions.append(output)

        # Combine predictions using soft voting
        combined_predictions = []
        for i in range(len(ensemble_predictions[0])):
            token_counts = Counter()
            for j, _ in enumerate(models):
                token = ensemble_predictions[j][i]["token_str"]
                # print(token)
                token_counts[token] += 1
            combined_predictions.append(token_counts)
            # print("TOKEN CONTS:", token_counts)

        # Choose the most frequent token as the ensemble prediction
        object_entities = []
        for token_counts in combined_predictions:
            if len(token_counts) == len(models):
                # All tokens are unique, choose the token with highest individual probability
                best_token = max(token_counts.items(), key=lambda x: x[1])[0]
            else:
                # Some tokens are repeated, choose the most frequent token
                best_token = max(token_counts, key=token_counts.get)
            object_entities.append(best_token)
            # print("BEST TOKEN:", best_token)

        result = {
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "Prompt": prompt,
            "ObjectEntities": [
                seq["token_str"]
                for seq in output if seq["score"] > args.threshold],
        }
        results.append(result)

    # Save the results
    logger.info(f"Saving the results to \"{args.output}\"...")
    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Probe a Language Model and "
                    "Run the Baseline Method on Prompt Outputs"
    )

    parser.add_argument(
        "-m",
        "--models",
        type=str,
        required=True,
        help="Comma-separated list of HuggingFace model names",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=str,
        required=True,
        help="Input test file (required)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        required=True,
        help="Output file (required)",
    )
    parser.add_argument(
        "-k",
        "--top_k",
        type=int,
        default=100,
        help="Top k prompt outputs (default: 100)",
    )
    parser.add_argument(
        "-t",
        "--threshold",
        type=float,
        default=0.5,
        help="Probability threshold (default: 0.5)",
    )
    parser.add_argument(
        "-g",
        "--gpu",
        type=int,
        default=-1,
        help="GPU ID, (default: -1, i.e., using CPU)"
    )

    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
