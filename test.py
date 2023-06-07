import re

# Regular expressions to match entities in the text
# entity_regex = re.compile(r'\b[A-Z][\w\s-]*\b')
entity_regex = re.compile(r'\b[A-Z][\w-]*(?:\s+[A-Z][\w-]*)*\b')
compound_regex = re.compile(r'\b\w+(?:ium|ate|ide|gen|on)\b', re.IGNORECASE)
language_regex = re.compile(r'\bis (\w+)(?: language)?\b')
# compound_regex = re.compile(r'\b\w+(?=(?:ium|ate|ide|gen|on)\b)', re.IGNORECASE)


# sentc = "['Zimbabwe shares a border with Zambia', 'Zimbabwe', 'Zimbabwean Democratic Republic]"

# extracted_entities = entity_regex.findall(str(sentc))

# print(extracted_entities)

import json
import ast
import itertools
from file_io import read_lm_kbc_jsonl_to_df

data = read_lm_kbc_jsonl_to_df("./data/alpaca-v1.jsonl")
test = read_lm_kbc_jsonl_to_df("./data/dev.jsonl")


# data['ObjectEntities'] = data['ObjectEntities'].apply(lambda x: x[0])
# data['ObjectEntities'] = data['ObjectEntities'].apply(lambda x: x + "]" if x[-1] != "]" else x)
# # This line has been removed
# # data['ObjectEntities'] = data['ObjectEntities'].apply(lambda x: ast.literal_eval(x) if x[-1] == "]" else ast.literal_eval(x + "]"))
# data['ObjectEntities'] = data['ObjectEntities'].apply(clean_list_string)

for i in data.iterrows():
    for j in test.iterrows():
    # if i[1].Relation == "ChemicalCompoundElement":
    #     print(set(compound_regex.findall(str(i[1].ObjectEntities))))
        if i[1].Relation == "StateSharesBorderState" and i[1].SubjectEntity == j[1].SubjectEntity:
            print("SUBJECT ENTITY: ", i[1].SubjectEntity)
            # print("TRUE OBJECTS: \n", j[1].ObjectEntities)
            # print("EXTRACTED OBJECTS: \n", set(entity_regex.findall(str(i[1].ObjectEntities))))

# import argparse
# import json
# import logging

# from torch.utils.data import Dataset
# from torch.utils.data.dataset import T_co
# from tqdm.auto import tqdm
# from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, pipeline, AutoModelForSeq2SeqLM

# from file_io import read_lm_kbc_jsonl

# logging.basicConfig(
#     format="%(asctime)s - %(levelname)s - %(name)s -  %(message)s",
#     datefmt="%m/%d/%Y %H:%M:%S",
#     level=logging.INFO,
# )
# logger = logging.getLogger(__name__)



# class PromptSet(Dataset):
#     def __init__(self, prompts):
#         self.prompts = prompts

#     def __len__(self):
#         return len(self.prompts)

#     def __getitem__(self, index) -> T_co:
#         return self.prompts[index]


# """def create_prompt(subject_entity: str, relation: str) -> str:

#     if relation == "CountryBordersWithCountry":
#         prompt = f"Which countries does {subject_entity} share a border with?"
#     elif relation == "CountryOfficialLanguage":
#         prompt = f"What is the official language of {subject_entity}?"
#     elif relation == "StateSharesBorderState":
#         prompt = f"Which states does {subject_entity} share a border with?"
#     elif relation == "RiverBasinsCountry":
#         prompt = f"In which countries are the river basins of {subject_entity}?"
#     elif relation == "ChemicalCompoundElement":
#         prompt = f"What elements does {subject_entity} consist of?"
#     elif relation == "PersonLanguage":
#         prompt = f"What language does {subject_entity} speak?"
#     elif relation == "PersonProfession":
#         prompt = f"What is the profession of {subject_entity}?"
#     elif relation == "PersonInstrument":
#         prompt = f"What instrument does {subject_entity} play?"
#     elif relation == "PersonEmployer":
#         prompt = f"Where does {subject_entity} work?"
#     elif relation == "PersonPlaceOfDeath":
#         prompt = f"Where did {subject_entity} die?"
#     elif relation == "PersonCauseOfDeath":
#         prompt = f"What was the cause of death of {subject_entity}?"
#     elif relation == "CompanyParentOrganization":
#         prompt = f"Who is the parent organization of {subject_entity}?"

#     return prompt"""

# def create_prompt(subject_entity, relation):
#     ### depending on the relation, we fix the prompt
#     if relation == "CountryBordersWithCountry":
#         prompt = f"""
# Which countries share boraders with Dominica? 1-Venezuela
# Which countries share boraders with North Korea? 1-South Korea 2-China 3-Russia
# Which countries share boraders with Serbia? 1-Montenegro 2-Kosovo 3-Bosnia and Herzegovina 4-Hungary 5-Croatia 6-Bulgaria  7-Macedonia 8-Albania 9-Romania
# Which countries share boraders with Fiji? NONE
# Which countries share boraders with {subject_entity}?
# """


# """Which countries share boraders with Dominica? ['Venezuela']
# Which countries share boraders with North Korea? ['South Korea', 'China', 'Russia']
# Which countries share boraders with Serbia? ['Montenegro', 'Kosovo', 'Bosnia and Herzegovina', 'Hungary', 'Croatia', 'Bulgaria',  'Macedonia', 'Albania', 'Romania']
# Which countries share boraders with Fiji? ['NONE']
# Which countries share boraders with {subject_entity}?"""

#     elif relation == "CountryOfficialLanguage":
#         prompt = f"""
# Which are the official languages of Suriname? 1-Dutch
# Which are the official languages of Canada? 1-English 2-French
# Which are the official languages of Singapore? 1-English 2-Malay 3-Mandarin 4-Tamil
# Which are the official languages of Sri Lanka? 1-Sinhala 2-Tamil
# Which are the official languages of {subject_entity}?       
# """

# """
# Which are the official languages of Suriname? ['Dutch']
# Which are the official languages of Canada? ['English', 'French']
# Which are the official languages of Singapore? ['English', 'Malay', 'Mandarin', 'Tamil']
# Which are the official languages of Sri Lanka? ['Sinhala', 'Tamil']
# Which are the official languages of {subject_entity}?        
# """
#     elif relation == "StateSharesBorderState":
#         prompt = f"""
# What states border San Marino? 1-San Leo 2-Acquaviva 3-Borgo Maggiore 4-Chiesanuova 5-Fiorentino
# What states border Whales? 1-England
# What states border Liguria? 1-Tuscany 2-Auvergne-Rhoone-Alpes 3-Piedmont 4-Emilia-Romagna
# What states border Mecklenberg-Western Pomerania? 1-Brandenburg 2-Pomeranian 3-Schleswig-Holstein 4-Lower Saxony
# What states border {subject_entity}?
# """

# """
# What states border San Marino? ['San Leo', 'Acquaviva', 'Borgo Maggiore', 'Chiesanuova', 'Fiorentino']
# What states border Whales? ['England']
# What states border Liguria? ['Tuscany', 'Auvergne-Rhoone-Alpes', 'Piedmont', 'Emilia-Romagna']
# What states border Mecklenberg-Western Pomerania? ['Brandenburg', 'Pomeranian', 'Schleswig-Holstein', 'Lower Saxony']
# What states border {subject_entity}?
# """

#     elif relation == "RiverBasinsCountry":
#         prompt = f"""
# What countries does the river Drava cross? 1-Hungary 2-Italy 3-Austria 4-Slovenia 5-Croatia
# What countries does the river Huai river cross? 1-China
# What countries does the river Paraná river cross? 1-Bolivia 2-Paraguay 3-Argentina 4-Brazil
# What countries does the river Oise cross? 1-Belgium 2-France
# What countries does the river {subject_entity} cross?
# """

# """
# What countries does the river Drava cross? ['Hungary', 'Italy', 'Austria', 'Slovenia', 'Croatia']
# What countries does the river Huai river cross? ['China']
# What countries does the river Paraná river cross? ['Bolivia', 'Paraguay', 'Argentina', 'Brazil']
# What countries does the river Oise cross? ['Belgium', 'France']
# What countries does the river {subject_entity} cross?
# """

#     elif relation == "ChemicalCompoundElement":
#         prompt = f"""
# What are all the atoms that make up the molecule Water? 1-Hydrogen 2-Oxygen
# What are all the atoms that make up the molecule Bismuth subsalicylate? 1-Bismuth
# What are all the atoms that make up the molecule Sodium Bicarbonate? 1-Hydrogen 2-Oxygen 3-Sodium 4-Carbon
# What are all the atoms that make up the molecule Aspirin? 1-Oxygen 2-Carbon 3-Hydrogen
# What are all the atoms that make up the molecule {subject_entity}?
# """

# """
# What are all the atoms that make up the molecule Water? ['Hydrogen', 'Oxygen']
# What are all the atoms that make up the molecule Bismuth subsalicylate	? ['Bismuth']
# What are all the atoms that make up the molecule Sodium Bicarbonate	? ['Hydrogen', 'Oxygen', 'Sodium', 'Carbon']
# What are all the atoms that make up the molecule Aspirin? ['Oxygen', 'Carbon', 'Hydrogen']
# What are all the atoms that make up the molecule {subject_entity}?
# """
#     elif relation == "PersonLanguage":
#         prompt = f"""
# Which languages does Aamir Khan speak? 1-Hindi 2-English 3-Urdu
# Which languages does Pharrell Williams speak? 1-English
# Which languages does Xabi Alonso speak? 1-German 2-Basque 3-Spanish 4-English
# Which languages does Shakira speak? 1-Catalan 2-English 3-Portuguese 4-Spanish 5-Italian 6-French
# which languages does {subject_entity} speak?
# """

# """
# Which languages does Aamir Khan speak? ['Hindi', 'English', 'Urdu']
# Which languages does Pharrell Williams speak? ['English']
# Which languages does Xabi Alonso speak? ['German', 'Basque', 'Spanish', 'English']
# Which languages does Shakira speak? ['Catalan', 'English', 'Portuguese', 'Spanish', 'Italian', 'French']
# which languages does {subject_entity} speak?
# """

#     elif relation == "PersonProfession":
#         prompt = f"""
# What is Danny DeVito's profession? 1-Comedian 2-Film Director 3-Voice Actor 4-Actor 5-Film Producer 6-Film Actor 7-Dub Actor 8-Activist 9-Television Actor
# What is David Guetta's profession? 1-DJ
# What is Gary Lineker's profession? 2-Commentator 3-Association 4-Football Player 5-Journalist 6-Broadcaster
# What is Gwyneth Paltrow's profession? 1-Film Actor 2-Musician
# What is {subject_entity}'s profession?
# """

# """
# What is Danny DeVito's profession? ['Comedian', 'Film Director', 'Voice Actor', 'Actor', 'Film Producer', 'Film Actor', 'Dub Actor', 'Activist', 'Television Actor']
# What is David Guetta's profession? ['DJ']
# What is Gary Lineker's profession? ['Commentator', 'Association Football Player', 'Journalist', 'Broadcaster']
# What is Gwyneth Paltrow's profession? ['Film Actor','Musician']
# What is {subject_entity}'s profession?
# """

#     elif relation == "PersonInstrument":
#         prompt = f"""
# Which instruments does Liam Gallagher play? 1-Maraca 2-Guitar
# Which instruments does Jay Park play? NONE
# Which instruments does Axl Rose play? 1-Guitar 2-Piano 3-Pander 4-Bass
# Which instruments does Neil Young play? 1-Guitar
# Which instruments does {subject_entity} play?
# """

# """
# Which instruments does Liam Gallagher play? ['Maraca', 'Guitar']
# Which instruments does Jay Park play? []
# Which instruments does Axl Rose play? ['Guitar', 'Piano', 'Pander', 'Bass']
# Which instruments does Neil Young play? ['Guitar']
# Which instruments does {subject_entity} play?
# """
#     elif relation == "PersonEmployer":
#         prompt = f"""
# Where is or was Susan Wojcicki employed? 1-Google
# Where is or was Steve Wozniak employed? 1-Apple Inc 2-Hewlett-Packard 3-University of Technology Sydney 4-Atari
# Where is or was Yukio Hatoyama employed? 1-Senshu University 2-Tokyo Institute of Technology
# Where is or was Yahtzee Croshaw employed? 1-PC Gamer 2-Hyper 3-Escapist
# Where is or was {subject_entity} employed?
# """

# """
# Where is or was Susan Wojcicki employed? ['Google']
# Where is or was Steve Wozniak employed? ['Apple Inc', 'Hewlett-Packard', 'University of Technology Sydney', 'Atari']
# Where is or was Yukio Hatoyama employed? ['Senshu University','Tokyo Institute of Technology']
# Where is or was Yahtzee Croshaw employed? ['PC Gamer', 'Hyper', 'Escapist']
# Where is or was {subject_entity} employed?
# """
#     elif relation == "PersonPlaceOfDeath":
#         prompt = f"""
# What is the place of death of Barack Obama? NONE
# What is the place of death of Ennio Morricone? 1-Rome
# What is the place of death of Elon Musk? NONE
# What is the place of death of Prince? 1-Chanhassen
# What is the place of death of {subject_entity}? 
# """

# """
# What is the place of death of Barack Obama? []
# What is the place of death of Ennio Morricone? ['Rome']
# What is the place of death of Elon Musk? []
# What is the place of death of Prince? ['Chanhassen']
# What is the place of death of {subject_entity}? 
# """

#     elif relation == "PersonCauseOfDeath":
#         prompt = f"""
# How did André Leon Talley die? 1-Infarction
# How did Angela Merkel die? NONE
# How did Bob Saget die? 1-Injury 2-Blunt Trauma
# How did Jamal Khashoggi die? 1-Murder
# How did {subject_entity} die?
# """

# """
# How did André Leon Talley die? ['Infarction']
# How did Angela Merkel die? []
# How did Bob Saget die? ['Injury', 'Blunt Trauma']
# How did Jamal Khashoggi die? ['Murder']
# How did {subject_entity} die?
# """

#     elif relation == "CompanyParentOrganization":
#         prompt = f"""
# What is the parent company of Microsoft? NONE
# What is the parent company of Sony? 1-Sony Group
# What is the parent company of Saab? 1-Saab Group 2-Saab-Scania 3-Spyker N.V. 4-National Electric Vehicle Sweden 5-General Motors
# What is the parent company of Max Motors? NONE
# What is the parent company of {subject_entity}?
# """

# """
# What is the parent company of Microsoft? []
# What is the parent company of Sony? ['Sony Group']
# What is the parent company of Saab? ['Saab Group', 'Saab-Scania', 'Spyker N.V.', 'National Electric Vehicle Sweden', 'General Motors']
# What is the parent company of Max Motors? []
# What is the parent company of {subject_entity}?
# """
#     return prompt


# def run(args):
#     # Load the model
#     model_type = args.model
#     logger.info(f"Loading the model \"{model_type}\"...")
    
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_type)
#     tokenizer = AutoTokenizer.from_pretrained(model_type)

#     pipe = pipeline(
#         model=model_type,
#         tokenizer=tokenizer,
#         top_k=args.top_k,
#         device=args.gpu
#     )

#     # Load the input file
#     logger.info(f"Loading the input file \"{args.input}\"...")
#     input_rows = read_lm_kbc_jsonl(args.input)
#     logger.info(f"Loaded {len(input_rows):,} rows.")

#     # Create prompts
#     # The code doesn't fit in one message, so I'm continuing in this one.

#     logger.info(f"Creating prompts...")
#     prompts = PromptSet([create_prompt(
#         subject_entity=row["SubjectEntity"],
#         relation=row["Relation"],
#     ) for row in input_rows])

#     # Run the model
#     logger.info(f"Running the model...")
#     outputs = []
#     for out in tqdm(pipe(prompts, batch_size=32, max_length=256), total=len(prompts)):
#         outputs.append(out)
#     results = []
#     for row, prompt, output in zip(input_rows, prompts, outputs):
#         #print(output)
#         result = {
#             "SubjectEntity": row["SubjectEntity"],
#             "Relation": row["Relation"],
#             "Prompt": prompt,
#             "ObjectEntities": [seq["generated_text"] for seq in output],
#         }
#         results.append(result)

#     # Save the results
#     logger.info(f"Saving the results to \"{args.output}\"...")
#     with open(args.output, "w") as f:
#         for result in results:
#             f.write(json.dumps(result) + "\n")
# """

# def generate_responses(model, tokenizer, prompts, max_length, num_return_sequences, penalty_alpha, top_k, max_new_tokens):
#     responses = []
#     for prompt in prompts:
#         inputs = tokenizer.encode(prompt, return_tensors='pt')
#         output = model.generate(inputs, 
#                                 max_length=max_length, 
#                                 num_return_sequences=num_return_sequences,
#                                 penalty_alpha=penalty_alpha, 
#                                 top_k=top_k, 
#                                 max_new_tokens=max_new_tokens)
#         decoded_output = [tokenizer.decode(out, skip_special_tokens=True) for out in output]
#         responses.append(decoded_output)
#     return responses


# def run(args):
#     # Load the model
#     model_type = args.model
#     logger.info(f"Loading the model \"{model_type}\"...")
    
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_type)
#     tokenizer = AutoTokenizer.from_pretrained(model_type)

#     # Load the input file
#     logger.info(f"Loading the input file \"{args.input}\"...")
#     input_rows = read_lm_kbc_jsonl(args.input)
#     logger.info(f"Loaded {len(input_rows):,} rows.")

#     # Create prompts
#     logger.info(f"Creating prompts...")
#     prompts = PromptSet([create_prompt(
#         subject_entity=row["SubjectEntity"],
#         relation=row["Relation"],
#     ) for row in input_rows])

#     # Run the model
#     logger.info(f"Running the model...")
#     outputs = generate_responses(model, tokenizer, prompts, max_length=256, num_return_sequences=args.top_k, penalty_alpha=0.3, top_k=50, max_new_tokens=50)
#     results = []
#     for row, prompt, output in zip(input_rows, prompts, outputs):
#         result = {
#             "SubjectEntity": row["SubjectEntity"],
#             "Relation": row["Relation"],
#             "Prompt": prompt,
#             "ObjectEntities": output,
#         }
#         results.append(result)

#     # Save the results
#     logger.info(f"Saving the results to \"{args.output}\"...")
#     with open(args.output, "w") as f:
#         for result in results:
#             f.write(json.dumps(result) + "\n")
            
# """

# def main():
#     parser = argparse.ArgumentParser(
#         description="Probe a Language Model and "
#                     "Run the Baseline Method on Prompt Outputs"
#     )

#     parser.add_argument(
#         "-m",
#         "--model",
#         type=str,
#         default="declare-lab/flan-alpaca-large",  # Change the default model
#         help="HuggingFace model name (default: declare-lab/flan-alpaca-large)",
#     )
#     parser.add_argument(
#         "-i",
#         "--input",
#         type=str,
#         required=True,
#         help="Input test file (required)",
#     )
#     parser.add_argument(
#         "-o",
#         "--output",
#         type=str,
#         required=True,
#         help="Output file (required)",
#     )
#     parser.add_argument(
#         "-k",
#         "--top_k",
#         type=int,
#         default=100,
#         help="Top k prompt outputs (default: 100)",
#     )
#     parser.add_argument(
#         "-t",
#         "--threshold",
#         type=float,
#         default=0.5,
#         help="Probability threshold (default: 0.5)",
#     )
#     parser.add_argument(
#         "-g",
#         "--gpu",
#         type=int,
#         default=-1,
#         help="GPU ID, (default: -1, i.e., using CPU)"
#     )

#     args = parser.parse_args()

#     run(args)


# if __name__ == '__main__':
#     main()
