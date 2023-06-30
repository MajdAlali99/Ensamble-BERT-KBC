import json
import torch
from transformers import BertForMaskedLM, BertTokenizerFast
from file_io import read_lm_kbc_jsonl_to_df

def generate_predictions(model, tokenizer, df):
    model.eval()

    predictions = []

    for _, row in df.iterrows():
        subject = row["SubjectEntity"]
        relation = row["Relation"]

        text = f"{subject}, {relation}, [MASK]"

        input_ids = tokenizer.encode(text, return_tensors="pt")
        with torch.no_grad():
            output = model(input_ids)
        logits = output.logits
        masked_indices = torch.where(input_ids == tokenizer.mask_token_id)[1]
        predicted_indices = torch.topk(logits[:, masked_indices], k=5, dim=2).indices.squeeze()
        predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indices)

        object_entities = ', '.join(predicted_tokens)
        predictions.append({"SubjectEntity": subject, "Relation": relation, "ObjectEntities": object_entities})

    return predictions


def save_predictions(predictions, output_file):
    with open(output_file, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

if __name__ == "__main__":
    model = BertForMaskedLM.from_pretrained("./mlm_output")
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

    test_filepath = "./data/val.jsonl"
    test_df = read_lm_kbc_jsonl_to_df(test_filepath)
    test_df["text"] = test_df["SubjectEntity"] + ", " + test_df["Relation"] + ", " + test_df["ObjectEntities"].apply(lambda x: ', '.join(x) + ', ' + tokenizer.mask_token)

    predictions = generate_predictions(model, tokenizer, test_df)
    save_predictions(predictions, "./data/bert-finetuned-out.jsonl")
