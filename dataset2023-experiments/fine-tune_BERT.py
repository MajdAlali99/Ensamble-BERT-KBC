import torch
from transformers import BertForMaskedLM, BertTokenizerFast, LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from evaluate import *

model = BertForMaskedLM.from_pretrained("bert-base-uncased")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

train_filepath = "./data/train.jsonl"
test_filepath = "./data/val.jsonl"
train_text_file = "train_text.txt"
test_text_file = "test_text.txt"
train_df = read_lm_kbc_jsonl_to_df(train_filepath)
test_df = read_lm_kbc_jsonl_to_df(test_filepath)
train_df["text"] = train_df["SubjectEntity"] + ", " + train_df["Relation"] + ", " + train_df["ObjectEntities"].apply(lambda x: ', '.join(x))
test_df["text"] = test_df["SubjectEntity"] + ", " + test_df["Relation"] + ", " + test_df["ObjectEntities"].apply(lambda x: ', '.join(x))

print("TEST EXAMPLE ====> ", test_df.text[0])

train_df["text"].to_csv(train_text_file, header=False, index=False)
test_df["text"].to_csv(test_text_file, header=False, index=False)

print("test_text_file EXAMPLE ====> ", test_text_file)

train_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=train_text_file, block_size=128)
test_dataset = LineByLineTextDataset(tokenizer=tokenizer, file_path=test_text_file, block_size=128)

print("LineByLineTextDataset EXAMPLE ====> ", train_dataset[0])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True, mlm_probability=0.15)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("USED DEVICE ====> ", device, torch.cuda.is_available())

training_args = TrainingArguments(
    output_dir="./mlm_output",
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
trainer.save_model("./mlm_output")
trainer.evaluate()