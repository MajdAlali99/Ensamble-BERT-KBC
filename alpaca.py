import argparse
import csv
import json
import logging
import requests
import random

from transformers import AutoModelForMaskedLM, AutoModelForCausalLM, AutoTokenizer, pipeline, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from typing import List

# Configure logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Disambiguation baseline
def disambiguation_baseline(item):
    try:
        url = f"https://www.wikidata.org/w/api.php?action=wbsearchentities&search={item}&language=en&format=json"
        data = requests.get(url).json()
        # Return the first id (Could upgrade this in the future)
        return data['search'][0]['id']
    except:
        return item

#Create a prompt using the provided data
def create_prompt(subject_entity: str, relation: str, prompt_templates: dict, instantiated_templates: List[str], tokenizer, few_shot: int = 0, task: str = "question-answering") -> str:
    prompt_template = prompt_templates[relation]
    if task == "question-answering":
        if few_shot > 0:
            random_examples = random.sample(instantiated_templates, min(few_shot, len(instantiated_templates)))
        else:
            random_examples = []
        few_shot_examples = "\n".join(random_examples)
        prompt = f"{few_shot_examples}\n{prompt_template.format(subject_entity=subject_entity)}"
    else:
        prompt = prompt_template.format(subject_entity=subject_entity, mask_token=tokenizer.mask_token)
    return prompt

def run(args):
    # Load the model
    model_type = args.model
    logger.info(f"Loading the model \"{model_type}\"...")
    tokenizer = AutoTokenizer.from_pretrained(model_type, padding_side='left')
    model = AutoModelForMaskedLM.from_pretrained(model_type)  if "bert" in model_type.lower() else AutoModelForSeq2SeqLM.from_pretrained("declare-lab/flan-alpaca-large", pad_token_id=tokenizer.eos_token_id) #AutoModelForCausalLM.from_pretrained(model_type) #
    task = "fill-mask" if "bert" in model_type.lower() else "question-answering"    
    #pipe = pipeline(task=task, model=model, tokenizer=tokenizer, top_k=args.top_k, device=args.gpu, fp16=args.fp16)  
    pipe = pipeline(model=model_type, tokenizer=tokenizer, top_k=args.top_k, device=args.gpu)  

    # Read the prompt templates and train data from CSV files
    if task == "question-answering":
        logger.info(f"Reading question prompt templates from \"{args.question_prompts}\"...")
        prompt_templates = read_prompt_templates_from_csv(args.question_prompts)
    else:
        logger.info(f"Reading fill-mask prompt templates from \"{args.fill_mask_prompts}\"...")
        prompt_templates = read_prompt_templates_from_csv(args.fill_mask_prompts)
    # Instantiate templates with train data
    instantiated_templates = []
    if task == "question-answering":
        logger.info(f"Reading train data from \"{args.train_data}\"...")
        train_data = read_train_data_from_csv(args.train_data)
        logger.info("Instantiating templates with train data...")
        for row in train_data:
            relation = row['Relation']
            prompt_template = prompt_templates[relation]
            object_entities = row['ObjectEntities']
            answers = ', '.join(object_entities)
            instantiated_example = prompt_template.format(subject_entity=row["SubjectEntity"]) + f" {answers}"
            instantiated_templates.append(instantiated_example)
        # print("THE TAMPLET ===> ", instantiated_templates[0])
        logger.info(f"example of the initiated tamplet:{ instantiated_templates[0]} ...")


    # Load the input file
    # logger.info(f"Loading the input file \"{args.input}\"...")
    # input_rows = [json.loads(line) for line in open(args.input, "r")]
    # logger.info(f"Loaded {len(input_rows):,} rows.")
    logger.info(f"Loading the input file \"{args.input}\"...")
    with open(args.input, "r") as f:
        if args.val_size > 0:
            input_rows = [json.loads(line) for _, line in zip(range(args.val_size), f)]
        else:
            input_rows = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(input_rows):,} rows.")

    # Create prompts
    logger.info(f"Creating prompts...")
    prompts = [create_prompt(
    subject_entity=row["SubjectEntity"],
    relation=row["Relation"],
    prompt_templates=prompt_templates,
    instantiated_templates=instantiated_templates,
    tokenizer=tokenizer,
    few_shot=args.few_shot,
    task=task,
    ) for row in input_rows]
    logger.info(f"example of the prompt tamplet: {prompt_templates} ...")
    print(prompts[0])
    #print(prompt_templates[0])

    # Run the model
    logger.info(f"Running the model...")
    if task == 'fill-mask':
        outputs = pipe(prompts, batch_size=args.batch_size)
    else:
        outputs = pipe(prompts, batch_size=args.batch_size, max_length=256)

    results = []
    for row, output, prompt in zip(input_rows, outputs, prompts):
        object_entities_with_wikidata_id = []  # Changed to set
        object_entities_text = [] # Changed to set

        if task == "fill-mask":
            for seq in output:
                if seq["score"] > args.threshold:
                    wikidata_id = disambiguation_baseline(seq["token_str"])
                    object_entities_with_wikidata_id.append(wikidata_id)  # Changed to add
                    object_entities_text.append(seq["token_str"])  # Changed to add

        else:
            qa_answer = output['generated_text'].split(prompt)[-1].strip()
            qa_entities = qa_answer.split(", ")
            print("PROMPT: ", prompt)
            print("OUTPUT: ", qa_entities)
            for entity in qa_entities:
                wikidata_id = disambiguation_baseline(entity)
                object_entities_with_wikidata_id.append(wikidata_id)  # Changed to add
                object_entities_text.append(entity)  # Changed to add

        result_row = {
            "SubjectEntityID": row["SubjectEntityID"],
            "SubjectEntity": row["SubjectEntity"],
            "ObjectEntitiesID": object_entities_with_wikidata_id,  # Convert back to list
            "Relation": row["Relation"],
            "ObjectEntities": object_entities_text  # Convert back to list
        }
        results.append(result_row)


    # Save the results
    logger.info(f"Saving the results to \"{args.output}\"...")
    with open(args.output, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

# Parsing command line arguments
parser = argparse.ArgumentParser(description="Model training and testing script")
parser.add_argument("--model", default="declare-lab/flan-alpaca-large", help="Model name or path")
parser.add_argument("--input", default="/content/drive/MyDrive/dataset2023-main-v0/data/val.jsonl", help="Input data path")
parser.add_argument("--output", default="/content/drive/MyDrive/dataset2023-main-v0/data/alpaca-out-2000.jsonl", help="Output path")
parser.add_argument("--top_k", type=int, default=10, help="Top K")
parser.add_argument("--threshold", type=float, default=0.8, help="Threshold value")
parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
parser.add_argument("--question_prompts", default="/content/drive/MyDrive/dataset2023-main-v0/question-prompts.csv", help="Question prompts path")
parser.add_argument("--fill_mask_prompts", default="/content/drive/MyDrive/dataset2023-main-v0/prompts.csv", help="Fill mask prompts path")
parser.add_argument("--few_shot", type=int, default=3, help="Few-shot learning value")
parser.add_argument("--train_data", default="/content/drive/MyDrive/dataset2023-main-v0/data/train_tiny.jsonl", help="Training data path")
parser.add_argument("--batch_size", type=int, default=25, help="Batch size")
parser.add_argument("--fp16", type=bool, default=False, help="16-bit floating point precision")
parser.add_argument("--val_size", type=int, default=500, help="Validation set size")

args = parser.parse_args()

# Removed class Args

run(args)