import json
import ast
import time
import pandas as pd
from file_io import *
from evaluate import *
from pathlib import Path
from transformers import pipeline, AutoTokenizer

# def OPT_response(q):
#     response = None
#     while response is None:
#         try:
#             response = generator(q, max_length=100, do_sample=False, temperature=0)
#             generated_text = response[0]['generated_text']
#             print("###TEXT###: ", generated_text)
#             # tokens = tokenizer.tokenize(generated_text)
#             # valid_string = " ".join(tokens)
#             response = ast.literal_eval(generated_text) # valid_string
#         except Exception as err:
#             print ("Following error occurred in OPT prediction \"{}\". Force stop and run again if error persists. Running again .....".format(err))
#             response = None
#             time.sleep(10)
#     return response

def OPT_response(q):
    response = None
    while response is None:
        try:
            response = generator(q, max_length=100, do_sample=False, temperature=0)
            generated_text = response[0]['generated_text']
            print('###TEXT###: ', generated_text)
            response = [item.strip() for item in generated_text.split(",") if item.startswith("Q")]
        except Exception as err:
            print ("Following error occurred in OPT prediction \"{}\". Force stop and run again if error persists. Running again .....".format(err))
            response = None
            time.sleep(10)
    return response

if __name__ == '__main__':
    train_filepath = Path("./data/train.jsonl")
    output_filepath = Path("./data/output.jsonl")

    model_name = 'facebook/opt-1.3b'
    generator = pipeline('text-generation', model=model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    prefix = '''State of Palestine, country-borders-country, ["Q801"]
    Paraguay, country-borders-country, ["Q155", "Q414", "Q750"]
    Lithuania, country-borders-country, ["Q34", "Q36", "Q159", "Q184", "Q211"]
    ''' 

    # prefix = '''the country Palestine have borders with the country: Israel.
    # the country Paraguay have borders with the country: Bolivia & Brazil & Argentina."
    # the country Paraguay have borders with the country: Sweden & Belarus & Latvia & Poland & Russia.
    # '''

    print('Starting probing OPT-1.3b ................')

    train_df = read_lm_kbc_jsonl_to_df(train_filepath)

    train_df = train_df[train_df['Relation'] == 'country-borders-country'].sample(10).reset_index(drop=True)
    print (train_df)

    results = []
    for idx, row in train_df.iterrows():
        prompt = prefix + row["SubjectEntity"] + ", " + row["Relation"] + ", "
        # prompt = prefix + "the country " + row["SubjectEntity"] + "have borders with the country: "
        print("Prompt is \"{}\"".format(prompt))
        result = {
            "SubjectEntityID": row["SubjectEntityID"],
            "SubjectEntity": row["SubjectEntity"],
            "Relation": row["Relation"],
            "ObjectEntities": OPT_response(prompt),
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

    print('Finished probing OPT-1.3b ................')
