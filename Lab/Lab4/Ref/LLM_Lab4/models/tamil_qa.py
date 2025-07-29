from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

model_name = "deepset/xlm-roberta-large-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def answer_question_tamil(context, question):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1
    answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True)

context = "ஒரு சிறிய கிராமத்தில் லீலா என்ற சிறுமி வாழ்ந்தாள். அவளுக்கு பறவைகளை மிகவும் பிடித்தது.தினமும் காலை எழுந்தவுடன் அவள் பூந்தோட்டத்தில் பறவைகளை பார்வையிட்டு பேசுவாள்.ஒரு நாள் ஒரு சிறிய கிளி வண்டியில் விழுந்தது.லீலா அதை வீட்டிற்கு கொண்டு வந்து பராமரித்தாள்.பிறகு அந்த கிளி மீண்டும் பறக்கத் தொடங்கியது.அதனால் லீலா மிகவும் மகிழ்ந்தாள்."
question = "லீலா கிளியை எப்படி கண்டாள் மற்றும் அவளால் என்ன நடந்தது?"
print(answer_question_tamil(context, question))