import argparse
import json
import logging

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, pipeline

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


def create_prompt(subject_entity: str, relation: str) -> str:
    """
    Depending on the relation, we fix the prompt
    """

    if relation == "CountryBordersWithCountry":
        prompt = f"Which countries does {subject_entity} share a border with?"
    elif relation == "CountryOfficialLanguage":
        prompt = f"What is the official language of {subject_entity}?"
    elif relation == "StateSharesBorderState":
        prompt = f"Which states does {subject_entity} share a border with?"
    elif relation == "RiverBasinsCountry":
        prompt = f"In which countries are the river basins of {subject_entity}?"
    elif relation == "ChemicalCompoundElement":
        prompt = f"What elements does {subject_entity} consist of?"
    elif relation == "PersonLanguage":
        prompt = f"What language does {subject_entity} speak?"
    elif relation == "PersonProfession":
        prompt = f"What is the profession of {subject_entity}?"
    elif relation == "PersonInstrument":
        prompt = f"What instrument does {subject_entity} play?"
    elif relation == "PersonEmployer":
        prompt = f"Where does {subject_entity} work?"
    elif relation == "PersonPlaceOfDeath":
        prompt = f"Where did {subject_entity} die?"
    elif relation == "PersonCauseOfDeath":
        prompt = f"What was the cause of death of {subject_entity}?"
    elif relation == "CompanyParentOrganization":
        prompt = f"Who is the parent organization of {subject_entity}?"

    return prompt


def run(args):
    # Load the model
    model_type = args.model
    logger.info(f"Loading the model \"{model_type}\"...")

    tokenizer = AutoTokenizer.from_pretrained(model_type)
    model = AutoModelForCausalLM.from_pretrained(model_type)

    pipe = pipeline(
        model=model,
        tokenizer=tokenizer,
        top_k=args.top_k,
        device=args.gpu
    )

    # Load the input file
    logger.info(f"Loading the input file \"{args.input}\"...")
    input_rows = read_lm_kbc_jsonl(args.input)
    logger.info(f"Loaded {len(input_rows):,} rows.")

    # Create prompts
    # The code doesn't fit in one message, so I'm continuing in this one.

    logger.info(f"Creating prompts...")
    prompts = PromptSet([create_prompt(
        subject_entity=row["SubjectEntity"],
        relation=row["Relation"],
    ) for row in input_rows])

    # Run the model
    logger.info(f"Running the model...")
    outputs = []
    for out in tqdm(pipe(prompts, batch_size=8), total=len(prompts)):
        outputs.append(out)
    results = []
    for row, prompt, output in zip(input_rows, prompts, outputs):
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
        "--model",
        type=str,
        default="declare-lab/flan-alpaca-large",  # Change the default model
        help="HuggingFace model name (default: declare-lab/flan-alpaca-large)",
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
