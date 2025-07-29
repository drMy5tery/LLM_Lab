from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

model_name = "ai4bharat/indic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)

def answer_question_hindi(context, question):
    inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
    outputs = model(**inputs)
    start_idx = torch.argmax(outputs.start_logits)
    end_idx = torch.argmax(outputs.end_logits) + 1
    answer_tokens = inputs["input_ids"][0][start_idx:end_idx]
    return tokenizer.decode(answer_tokens, skip_special_tokens=True)