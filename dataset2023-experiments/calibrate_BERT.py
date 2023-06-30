import torch
import numpy as np
from transformers import BertForMaskedLM, BertTokenizerFast, LineByLineTextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from evaluate import *
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, log_loss
# from scipy.special import log_softmax
import torch
from torch.nn.functional import log_softmax
from scipy.optimize import minimize_scalar



# fin-tuning step
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

# function for Temperature Scaling
class TemperatureScaled(torch.nn.Module):
    def __init__(self, model, temperature=1.0):
        super(TemperatureScaled, self).__init__()
        self.model = model
        self.temperature = temperature

    def forward(self, *args, **kwargs):
        logits = self.model(*args, **kwargs).logits
        return logits / self.temperature

# Step 3: Update the evaluation function to include temperature scaling
# and search for the optimal temperature
def tune_temperature(trainer, eval_dataset, max_iter=100):
    def loss_with_temperature(temperature):
        log_probs = []
        labels = []

        with torch.no_grad():
            for batch in DataLoader(eval_dataset, batch_size=trainer.args.per_device_eval_batch_size, collate_fn=trainer.data_collator):
                batch = {k: v.to(trainer.args.device) for k, v in batch.items()}
                outputs = trainer.model(**batch, return_dict=True)
                logits = outputs.logits.detach().cpu().numpy() / temperature

                mask = batch['labels'].cpu().numpy() != -100
                logits_masked = logits[mask]
                labels_masked = batch['labels'].detach().cpu().numpy()[mask]

                log_probs.extend(logits_masked)
                labels.extend(labels_masked)

        log_probs = np.array(log_probs)
        labels = np.array(labels)
        # log_probabilities = log_softmax(log_probs, axis=-1)
        log_probabilities = log_softmax(torch.tensor(log_probs), dim=-1).numpy()
        loss = -np.mean(log_probabilities[np.arange(len(labels)), labels])

        return loss, log_probs, labels

    res = minimize_scalar(lambda temp: loss_with_temperature(temp)[0], bounds=(1e-3, 10), method='bounded', options={'maxiter': max_iter})
    _, log_probs, labels = loss_with_temperature(res.x)
    return res.x, log_probs, labels

# Step 4: Update the Trainer to use the new evaluation function
class TemperatureScaledTrainer(Trainer):
    def evaluate(self, eval_dataset=None, ignore_keys=None):
        # ... (use the default evaluation function)
        output = super().evaluate(eval_dataset=eval_dataset, ignore_keys=ignore_keys)

        # Apply temperature scaling and find the optimal temperature
        self.optimal_temperature, log_probs, labels = tune_temperature(self, eval_dataset)
        # scaled_accuracy = accuracy_score(labels.ravel(), np.argmax(log_probs.detach().numpy(), axis=1).ravel())
        scaled_accuracy = accuracy_score(labels.ravel(), np.argmax(log_probs, axis=1).ravel())

        # Update the evaluation output with the scaled accuracy and optimal temperature
        output.update({"scaled_accuracy": scaled_accuracy, "optimal_temperature": self.optimal_temperature})
        return output


training_args = TrainingArguments(
    output_dir="./mlm_output",
    overwrite_output_dir=True,
    num_train_epochs=5,
    per_device_train_batch_size=8,
    save_steps=10_000,
    save_total_limit=2,
    prediction_loss_only=True,
)

trainer = TemperatureScaledTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)

trainer.train()
trainer.save_model("./mlm_output")

# Step 5: Evaluate the model with the optimal temperature
evaluation_output = trainer.evaluate(eval_dataset=test_dataset)
print(evaluation_output)

# Apply temperature scaling to the saved model
model = BertForMaskedLM.from_pretrained("./mlm_output")
scaled_model = TemperatureScaled(model, temperature=trainer.optimal_temperature)
scaled_model.to(device)
# scaled_model.save_pretrained("./mlm_output_temperature_scaled")

torch.save({
    'state_dict': scaled_model.model.state_dict(),
    # 'optimal_temperature': scaled_model.optimal_temperature
    'optimal_temperature': scaled_model.temperature

}, 'mlm_output_temperature_scaled.pth')


# # Load the model's state_dict and optimal_temperature
# saved_data = torch.load('mlm_output_temperature_scaled.pth')
# state_dict = saved_data['state_dict']
# optimal_temperature = saved_data['optimal_temperature']

# # Instantiate the model and load the state_dict
# loaded_model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
# loaded_temperature_scaled = TemperatureScaled(loaded_model, optimal_temperature)
# loaded_temperature_scaled.model.load_state_dict(state_dict)
