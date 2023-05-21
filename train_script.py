import json
from pathlib import Path
from typing import List, Dict, Union
import pandas as pd
import torch
from transformers import BertForMaskedLM, BertTokenizerFast, LineByLineTextDataset, DataCollatorForLanguageModeling, logging
from transformers import Trainer, TrainingArguments
import argparse

def read_lm_kbc_jsonl(file_path: Union[str, Path]) -> List[Dict]:
    rows = []
    with open(file_path, "r") as f:
        for line in f:
            row = json.loads(line)
            rows.append(row)
    return rows

def read_lm_kbc_jsonl_to_df(file_path: Union[str, Path]) -> pd.DataFrame:
    rows = read_lm_kbc_jsonl(file_path)
    df = pd.DataFrame(rows)
    return df

def main(args):
    logging.set_verbosity_error()  # avoid irritating transformers warnings

    model = BertForMaskedLM.from_pretrained(args.model_name)
    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)

    train_df = read_lm_kbc_jsonl_to_df(args.input_file_path)
    test_df = read_lm_kbc_jsonl_to_df(args.dev_file_path)

    train_df["text"] = train_df["SubjectEntity"] + ", " + \
                    train_df["Relation"] + ", " + \
                    train_df["ObjectEntities"].apply(lambda x: ', '.join([i[0] if len(i)>0 else "" for i in x]))

    test_df["text"] = test_df["SubjectEntity"] + ", " + \
                    test_df["Relation"] + ", " + \
                    test_df["ObjectEntities"].apply(lambda x: ', '.join([i[0] if len(i)>0 else "" for i in x]))

    train_df["text"].to_csv(args.train_txt_file_path, header=False, index=False)
    test_df["text"].to_csv(args.test_txt_file_path, header=False, index=False)

    train_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=args.train_txt_file_path, block_size=128)
    test_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=args.test_txt_file_path, block_size=128)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("USED DEVICE ====> ", device, torch.cuda.is_available())

    training_args = TrainingArguments(
        output_dir=args.output_path,
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_device_train_batch_size=8,
        save_steps=10_000,
        save_total_limit=2,
        prediction_loss_only=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    trainer.save_model(args.output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_name', type=str, required=True, help='The name of the model')
    parser.add_argument('-i', '--input_file_path', type=str, required=True, help='Input file path')
    parser.add_argument('-d', '--dev_file_path', type=str, required=True, help='Development file path')
    parser.add_argument('-tr', '--train_txt_file_path', type=str, required=True, help='Train text file path')
    parser.add_argument('-ts', '--test_txt_file_path', type=str, required=True, help='Test text file path')
    parser.add_argument('-o', '--output_path', type=str, required=True, help='Output path for the trained model')

    args = parser.parse_args()
    main(args)
