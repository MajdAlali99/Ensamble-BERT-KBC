import json
import ast
import time
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
from file_io import *
from evaluate import *

import logging
logging.getLogger("transformers").setLevel(logging.ERROR)

def LLaMA7B_response(prompt):
    response = None
    while response is None:
        try:
            input_tokens = tokenizer.encode(prompt, return_tensors="pt")
            output_tokens = model.generate(input_tokens, max_length=200)#, max_new_tokens=100)
            response = tokenizer.decode(output_tokens[0])
            response = ast.literal_eval(response)
        except Exception as err:
            print("Following error occurred in LLaMA 7B prediction \"{}\". Force stop and run again if error persists. Running again .....".format(err))
            response = None
            time.sleep(10)
    return response

if __name__ == '__main__':
    train_filepath = Path("./data/train.jsonl")
    output_filepath = Path("./data/output-LLAMA.jsonl")

    model = AutoModelForCausalLM.from_pretrained("decapoda-research/llama-7b-hf")
    tokenizer = AutoTokenizer.from_pretrained("decapoda-research/llama-7b-hf")



    prefix = '''State of Palestine, country-borders-country, ["Q801"]
    Paraguay, country-borders-country, ["Q155", "Q414", "Q750"]
    Lithuania, country-borders-country, ["Q34", "Q36", "Q159", "Q184", "Q211"]
    '''

    print('Starting probing LLaMA 7B ................')

    train_df = read_lm_kbc_jsonl_to_df(train_filepath)
    train_df = train_df[train_df['Relation'] == 'country-borders-country'].sample(10).reset_index(drop=True)
    print(train_df)

    results = []
    for idx, row in train_df.iterrows():
        prompt = prefix + row["SubjectEntity"] + ", " + row["Relation"] + ", "
        print("Prompt is \"{}\"".format(prompt))
        result = {
            "SubjectEntityID": row["SubjectEntityID"],
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "ObjectEntities": LLaMA7B_response(prompt),
        }
        results.append(result)

    save_df_to_jsonl(output_filepath, results)

    results = pd.DataFrame(results, columns=['SubjectEntityID', 'SubjectEntity', 'Relation', 'ObjectEntities'])
    train_df = train_df.drop(columns={'ObjectEntities'}).rename(columns={'ObjectEntitiesID': 'ObjectEntities'})

    scores_per_sr_pair = evaluate_per_sr_pair(json.loads(results.to_json(orient='records')), json.loads(train_df.to_json(orient='records')))
    scores_per_relation = combine_scores_per_relation(scores_per_sr_pair)

    scores_per_relation["*** Average ***"] = {
        "p": sum([x["p"] for x in scores_per_relation.values()]) / len(scores_per_relation),
        "r": sum([x["r"] for x in scores_per_relation.values()]) / len(scores_per_relation),
        "f1": sum([x["f1"] for x in scores_per_relation.values()]) / len(scores_per_relation),
    }

    print(pd.DataFrame(scores_per_relation).transpose().round(3))

    print('Finished probing LLaMA 7B ................')


## https://github.com/huggingface/transformers/issues/22222
## facebook/opt-1.3b other model