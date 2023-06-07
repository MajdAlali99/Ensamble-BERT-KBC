import json
from pathlib import Path
from typing import List, Dict, Union

import pandas as pd
import re

all_elements = ['hydrogen', 'helium', 'lithium', 'beryllium', 'boron', 'carbon', 'nitrogen', 'oxygen', 'fluorine', 'neon',
                    'sodium', 'magnesium', 'aluminium', 'silicon', 'phosphorus', 'sulfur', 'chlorine', 'argon', 'potassium', 'calcium',
                    'scandium', 'titanium', 'vanadium', 'chromium', 'manganese', 'iron', 'cobalt', 'nickel', 'copper', 'zinc',
                    'gallium', 'germanium', 'arsenic', 'selenium', 'bromine', 'krypton', 'rubidium', 'strontium', 'yttrium', 'zirconium',
                    'niobium', 'molybdenum', 'technetium', 'ruthenium', 'rhodium', 'palladium', 'silver', 'cadmium', 'indium', 'tin',
                    'antimony', 'tellurium', 'iodine', 'xenon', 'caesium', 'barium', 'lanthanum', 'cerium', 'praseodymium', 'neodymium',
                    'promethium', 'samarium', 'europium', 'gadolinium', 'terbium', 'dysprosium', 'holmium', 'erbium', 'thulium', 'ytterbium',
                    'lutetium', 'hafnium', 'tantalum', 'tungsten', 'rhenium', 'osmium', 'iridium', 'platinum', 'gold', 'mercury',
                    'thallium', 'lead', 'bismuth', 'polonium', 'astatine', 'radon', 'francium', 'radium', 'actinium', 'thorium',
                    'protactinium', 'uranium', 'neptunium', 'plutonium', 'americium', 'curium', 'berkelium', 'californium', 'einsteinium', 'fermium',
                    'mendelevium', 'nobelium', 'lawrencium', 'rutherfordium', 'dubnium', 'seaborgium', 'bohrium', 'hassium', 'meitnerium', 'darmstadtium',
                    'roentgenium', 'copernicium', 'nihonium', 'flerovium', 'moscovium', 'livermorium', 'tennessine', 'oganesson']

def extract_entities(text_list: List[str], relation: str) -> List[str]:
    # Regular expressions to match entities in the text
    # entity_regex = re.compile(r'\b[A-Z][\w\s-]*\b')
    entity_regex = re.compile(r'\b[A-Z][\w-]*(?:\s+[A-Z][\w-]*)*\b')
    # compound_regex = re.compile(r'\b\w+(?:ium|ate|ide|gen|on)\b', re.IGNORECASE)
    language_regex = re.compile(r'\bis (\w+)(?: language)?\b')

    entities = []

    for text in text_list:
        #print(str(text))
        if relation == "ChemicalCompoundElement":
            # Find all the entities in the text using the compound pattern
            # print(text)
            for element in text:
                elements = []
                if element in all_elements and element not in elements:
                    elements.append(element)
            extracted_entities = elements

        elif relation == "CountryOfficialLanguage" or relation  == "PersonLanguage":
            # Find all the entities in the text using the language pattern
            extracted_entities = language_regex.findall(str(text))
        else:
            # Find all the entities in the text using the general pattern
            extracted_entities = entity_regex.findall(str(text))

        # Remove unnecessary words like 'The'
        extracted_entities = [e.strip() for e in extracted_entities if e.lower() not in {"the", "is"}]
        # if extracted_entities:
        #     print(extracted_entities)
        # entities.extend(extracted_entities)

    return list(entities)

def read_lm_kbc_jsonl(filepath: str, preprocess: bool = False) -> List[Dict]:
    data = []
    with open(filepath, "r") as f:
        for line in f:
            row = json.loads(line)
            if preprocess:
                row["ObjectEntities"] = extract_entities(row["ObjectEntities"], row['Relation'])
            data.append(row)
    return data

def is_none_gts(gts: List[str]) -> bool:
    return not gts


def read_lm_kbc_jsonl_to_df(file_path: Union[str, Path]) -> pd.DataFrame:
    """
    Reads a LM-KBC jsonl file and returns a dataframe.
    """
    rows = read_lm_kbc_jsonl(file_path)
    df = pd.DataFrame(rows)
    return df


def write_lm_kbc_jsonl(data: List[Dict], file_path: str):
    """
    Writes a list of dictionaries to a LM-KBC jsonl file.

    Args:
        data: list of dictionaries, each possibly has the following keys:
            - "SubjectEntity": str
            - "Relation": str
            - "ObjectEntities":
                None or List[List[str]] (can be omitted for the test input)
        file_path: path to the jsonl file
    """
    with open(file_path, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')