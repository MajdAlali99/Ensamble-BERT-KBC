from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import torch
import torch.nn.functional as F

# ------------------------------------------------------------------------------------------------
#        Test MLM training
# ------------------------------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained('bert-large-cased')
# model = AutoModelForMaskedLM.from_pretrained("../dataset2023-main-v0/mlm_output")
model = AutoModelForMaskedLM.from_pretrained("./models/bert-large-kgcMLM")
# model = AutoModelForMaskedLM.from_pretrained("bert-large-cased")

mask_token = tokenizer.mask_token
subject_entity = "Syria"

# prompt = f"{subject_entity}, CountryBordersWithCountry, {mask_token}."
prompt = f"{subject_entity} shares border with {mask_token}."

pipe = pipeline(
    task="fill-mask",
    model=model,
    tokenizer=tokenizer,
    top_k=10,
)

outputs = pipe(prompt)

for out in outputs:
    print(out)

# ------------------------------------------------------------------------------------------------
#        Test temp scaling
# ------------------------------------------------------------------------------------------------

# import torch
# from transformers import BertForMaskedLM, BertTokenizerFast
# from evaluate import *
# from torch.utils.data import DataLoader
# # from scipy.special import log_softmax
# import torch
# from calibrate_BERT import TemperatureScaled

# # Load the saved model and the temperature scaling factor
# saved_model = torch.load('mlm_output_temperature_scaled.pth')

# # Load model and set state_dict
# model = BertForMaskedLM.from_pretrained("./mlm_output")
# model.load_state_dict(saved_model['state_dict'])
# temperature = saved_model['optimal_temperature']

# tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Wrap your model into TemperatureScaled
# model = TemperatureScaled(model, temperature)
# model.to(device)

# # Tokenize your prompt
# prompt = "Syria, country-borders-country, [MASK]"
# input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# # Run the model on the prompt and get predicted token
# model.eval()
# with torch.no_grad():
#     outputs = model(input_ids)
#     logits = outputs
#     probabilities = torch.nn.functional.softmax(logits / temperature, dim=-1)
#     predicted_index = torch.argmax(probabilities[0, -1, :]).item()

# # Decode predicted token
# predicted_token = tokenizer.decode([predicted_index])
# print("Predicted token: ", predicted_token)





# # Load the saved model and the temperature scaling factor
# saved_model = torch.load('mlm_output_temperature_scaled.pth')

# # Load model and set state_dict
# model = BertForMaskedLM.from_pretrained("bert-base-uncased")
# model.load_state_dict(saved_model['state_dict'])
# temperature = saved_model['optimal_temperature']

# tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # Wrap your model into TemperatureScaled
# model = TemperatureScaled(model, temperature)
# model.to(device)

# # Suppose we have test data in a file test_text.txt
# test_data = LineByLineTextDataset(tokenizer=tokenizer, file_path="test_text.txt", block_size=128)
# test_dataloader = DataLoader(test_data, batch_size=8)

# # Run the model on the test data and make predictions
# model.eval()
# with torch.no_grad():
#     for i, batch in enumerate(test_dataloader):
#         inputs = batch['input_ids'].to(device)
#         attention_mask = batch['attention_mask'].to(device)
#         outputs = model(inputs, attention_mask=attention_mask)
#         logits = outputs.logits
#         probabilities = torch.nn.functional.softmax(logits / temperature, dim=-1)
#         predictions = torch.argmax(probabilities, dim=-1)

#         # Now you can do something with the predictions