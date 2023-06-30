import json
import ast
from file_io import *
from evaluate import *
import time
from pathlib import Path
import torch
from transformers import BertTokenizer, BertForMaskedLM

def BERTresponse(q, model, tokenizer):
    response = None
    while response is None:
        try:
            input_text = q + tokenizer.mask_token
            input_ids = tokenizer.encode(input_text, return_tensors="pt")
            mask_idx = torch.where(input_ids == tokenizer.mask_token_id)[1]

            outputs = model(input_ids)
            logits = outputs.logits
            softmax = torch.nn.functional.softmax(logits, dim=-1)
            mask_word_probs = softmax[0, mask_idx, :].squeeze()
            topk = torch.topk(mask_word_probs, 5)

            response = tokenizer.convert_ids_to_tokens(topk.indices)
            response = [x for x in response if len(x) > 1 and x[0] != "[" and x[-1] != "]"]
        except Exception as err:
            print("Following error occurred in BERT-large prediction \"{}\". Force stop and run again if error persists. Running again .....".format(err))
            response = None
            time.sleep(10)
    return response


if __name__ == '__main__':
    train_filepath = Path("./data/train.jsonl")
    output_filepath = Path("./data/output.jsonl")
    
    model_name = "bert-large-uncased"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForMaskedLM.from_pretrained(model_name)

    prefix = '''State of Palestine, country-borders-country, ["Q801"]
    Paraguay, country-borders-country, ["Q155", "Q414", "Q750"]
    Lithuania, country-borders-country, ["Q34", "Q36", "Q159", "Q184", "Q211"]
    ''' 

    print('Starting probing BERT-large ................')

    train_df = read_lm_kbc_jsonl_to_df(train_filepath)
    
    # for monetary and test purposes, we take a sample from the dataframe for country-borders-country relation
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
            "ObjectEntities": BERTresponse(prompt, model, tokenizer),  ## naming prediction IDs directly as objectEntities here 
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
    print('Finished probing BERT-large ................')