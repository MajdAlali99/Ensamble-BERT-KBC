from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
import torch
import torch.nn.functional as F

# ------------------------------------------------------------------------------------------------
#        Test MLM training
# ------------------------------------------------------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForMaskedLM.from_pretrained("./mlm_output")

mask_token = tokenizer.mask_token
subject_entity = "Syria"

prompt = f"{subject_entity}, country-borders-country, {mask_token}."

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




# calibrated triples prompt
# {'score': 0.616692841053009, 'token': 8341, 'token_str': 'lebanon', 'sequence': 'syria, country - borders - country, lebanon.'}
# {'score': 0.13639438152313232, 'token': 5207, 'token_str': 'jordan', 'sequence': 'syria, country - borders - country, jordan.'}
# {'score': 0.13069632649421692, 'token': 4977, 'token_str': 'turkey', 'sequence': 'syria, country - borders - country, turkey.'}
# {'score': 0.03673074394464493, 'token': 5712, 'token_str': 'iraq', 'sequence': 'syria, country - borders - country, iraq.'}
# {'score': 0.015300646424293518, 'token': 3956, 'token_str': 'israel', 'sequence': 'syria, country - borders - country, israel.'}
# {'score': 0.013778207823634148, 'token': 7795, 'token_str': 'syria', 'sequence': 'syria, country - borders - country, syria.'}
# {'score': 0.005671328399330378, 'token': 13437, 'token_str': 'tunisia', 'sequence': 'syria, country - borders - country, tunisia.'}
# {'score': 0.005660138092935085, 'token': 13968, 'token_str': 'yemen', 'sequence': 'syria, country - borders - country, yemen.'}
# {'score': 0.004796930588781834, 'token': 16094, 'token_str': 'damascus', 'sequence': 'syria, country - borders - country, damascus.'}
# {'score': 0.00432570930570364, 'token': 5279, 'token_str': 'egypt', 'sequence': 'syria, country - borders - country, egypt.'}

# uncalibrated triples prompt
# {'score': 0.04288412630558014, 'token': 2313, 'token_str': 'population', 'sequence': 'syria, country - borders - country, population.'}
# {'score': 0.03911660239100456, 'token': 2406, 'token_str': 'country', 'sequence': 'syria, country - borders - country, country.'}
# {'score': 0.028627995401620865, 'token': 7795, 'token_str': 'syria', 'sequence': 'syria, country - borders - country, syria.'}
# {'score': 0.024241860955953598, 'token': 2381, 'token_str': 'history', 'sequence': 'syria, country - borders - country, history.'}
# {'score': 0.02215092070400715, 'token': 4949, 'token_str': 'map', 'sequence': 'syria, country - borders - country, map.'}
# {'score': 0.02155419811606407, 'token': 8341, 'token_str': 'lebanon', 'sequence': 'syria, country - borders - country, lebanon.'}
# {'score': 0.016238851472735405, 'token': 2555, 'token_str': 'region', 'sequence': 'syria, country - borders - country, region.'}
# {'score': 0.013454382307827473, 'token': 6645, 'token_str': 'borders', 'sequence': 'syria, country - borders - country, borders.'}
# {'score': 0.013307929039001465, 'token': 2171, 'token_str': 'name', 'sequence': 'syria, country - borders - country, name.'}
# {'score': 0.012529679574072361, 'token': 5712, 'token_str': 'iraq', 'sequence': 'syria, country - borders - country, iraq.'}

# calibrated triples prompt
# {'score': 0.2666928172111511, 'token': 5118, 'token_str': 'austria', 'sequence': 'germany, country - borders - country, austria.'}
# {'score': 0.265587717294693, 'token': 5288, 'token_str': 'switzerland', 'sequence': 'germany, country - borders - country, switzerland.'}
# {'score': 0.045650918036699295, 'token': 2605, 'token_str': 'france', 'sequence': 'germany, country - borders - country, france.'}
# {'score': 0.03513256832957268, 'token': 3735, 'token_str': 'poland', 'sequence': 'germany, country - borders - country, poland.'}
# {'score': 0.033166710287332535, 'token': 4549, 'token_str': 'netherlands', 'sequence': 'germany, country - borders - country, netherlands.'}
# {'score': 0.032198190689086914, 'token': 2762, 'token_str': 'germany', 'sequence': 'germany, country - borders - country, germany.'}
# {'score': 0.0303726214915514, 'token': 3304, 'token_str': 'italy', 'sequence': 'germany, country - borders - country, italy.'}
# {'score': 0.024882208555936813, 'token': 4701, 'token_str': 'sweden', 'sequence': 'germany, country - borders - country, sweden.'}
# {'score': 0.019319217652082443, 'token': 5706, 'token_str': 'belgium', 'sequence': 'germany, country - borders - country, belgium.'}
# {'score': 0.0165106151252985, 'token': 16426, 'token_str': 'brandenburg', 'sequence': 'germany, country - borders - country, brandenburg.'}

# uncalibrated triples prompt
# {'score': 0.03639591857790947, 'token': 2762, 'token_str': 'germany', 'sequence': 'germany, country - borders - country, germany.'}
# {'score': 0.03433101996779442, 'token': 2381, 'token_str': 'history', 'sequence': 'germany, country - borders - country, history.'}
# {'score': 0.024689391255378723, 'token': 1052, 'token_str': 'p', 'sequence': 'germany, country - borders - country, p.'}
# {'score': 0.02108820155262947, 'token': 2406, 'token_str': 'country', 'sequence': 'germany, country - borders - country, country.'}
# {'score': 0.01997896283864975, 'token': 2885, 'token_str': 'europe', 'sequence': 'germany, country - borders - country, europe.'}
# {'score': 0.015119803138077259, 'token': 2555, 'token_str': 'region', 'sequence': 'germany, country - borders - country, region.'}
# {'score': 0.012396415695548058, 'token': 4385, 'token_str': 'etc', 'sequence': 'germany, country - borders - country, etc.'}
# {'score': 0.011953307315707207, 'token': 3842, 'token_str': 'nation', 'sequence': 'germany, country - borders - country, nation.'}
# {'score': 0.01175936684012413, 'token': 3304, 'token_str': 'italy', 'sequence': 'germany, country - borders - country, italy.'}
# {'score': 0.00904925912618637, 'token': 3226, 'token_str': 'culture', 'sequence': 'germany, country - borders - country, culture.'}



# calibrated original prombt
# {'score': 0.2158522754907608, 'token': 8341, 'token_str': 'lebanon', 'sequence': 'syria shares border with lebanon.'}
# {'score': 0.18444256484508514, 'token': 4238, 'token_str': 'iran', 'sequence': 'syria shares border with iran.'}
# {'score': 0.12268328666687012, 'token': 3956, 'token_str': 'israel', 'sequence': 'syria shares border with israel.'}
# {'score': 0.10872320085763931, 'token': 5712, 'token_str': 'iraq', 'sequence': 'syria shares border with iraq.'}
# {'score': 0.09254060685634613, 'token': 4977, 'token_str': 'turkey', 'sequence': 'syria shares border with turkey.'}
# {'score': 0.07763785868883133, 'token': 5207, 'token_str': 'jordan', 'sequence': 'syria shares border with jordan.'}
# {'score': 0.0308292955160141, 'token': 5279, 'token_str': 'egypt', 'sequence': 'syria shares border with egypt.'}
# {'score': 0.028058456256985664, 'token': 13437, 'token_str': 'tunisia', 'sequence': 'syria shares border with tunisia.'}
# {'score': 0.024586619809269905, 'token': 10411, 'token_str': 'sudan', 'sequence': 'syria shares border with sudan.'}
# {'score': 0.016549233347177505, 'token': 12917, 'token_str': 'libya', 'sequence': 'syria shares border with libya.'}

# uncalibrated original prombt
# {'score': 0.23750248551368713, 'token': 8341, 'token_str': 'lebanon', 'sequence': 'syria shares border with lebanon.'}
# {'score': 0.1196231096982956, 'token': 5207, 'token_str': 'jordan', 'sequence': 'syria shares border with jordan.'}
# {'score': 0.10193824768066406, 'token': 4977, 'token_str': 'turkey', 'sequence': 'syria shares border with turkey.'}
# {'score': 0.08328896015882492, 'token': 4238, 'token_str': 'iran', 'sequence': 'syria shares border with iran.'}
# {'score': 0.07923831045627594, 'token': 5279, 'token_str': 'egypt', 'sequence': 'syria shares border with egypt.'}
# {'score': 0.0722748413681984, 'token': 5712, 'token_str': 'iraq', 'sequence': 'syria shares border with iraq.'}
# {'score': 0.06539595872163773, 'token': 3956, 'token_str': 'israel', 'sequence': 'syria shares border with israel.'}
# {'score': 0.03538859263062477, 'token': 10411, 'token_str': 'sudan', 'sequence': 'syria shares border with sudan.'}
# {'score': 0.01247909665107727, 'token': 5483, 'token_str': 'greece', 'sequence': 'syria shares border with greece.'}
# {'score': 0.011043316684663296, 'token': 13968, 'token_str': 'yemen', 'sequence': 'syria shares border with yemen.'}