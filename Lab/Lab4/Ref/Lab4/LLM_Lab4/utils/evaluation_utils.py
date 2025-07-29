import sacrebleu
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

def compute_bleu(reference, candidate):
    if isinstance(reference, list):
        reference = reference[0]  # handle single reference inside a list
    
    bleu = sacrebleu.sentence_bleu(candidate, [reference])
    return bleu.score / 100

def compute_rouge(reference, candidate):
    scores = scorer.score(reference, candidate)
    return {
        'rouge1_f': scores['rouge1'].fmeasure,
        'rouge2_f': scores['rouge2'].fmeasure,
        'rougeL_f': scores['rougeL'].fmeasure,
    }

def evaluate_generated_answers(language, data_list):
    results = []
    for item in data_list:
        question = item["question"]
        ref = item["reference"]
        gen = item["generated"]
        
        # Debug prints to check types
        print(f"Evaluating QA - Lang: {language}")
        print(f"Reference type: {type(ref)}, value: {ref}")
        print(f"Generated type: {type(gen)}, value: {gen}")

        bleu_score = compute_bleu(ref, gen)
        rouge_scores = compute_rouge(ref, gen)

        results.append({
            "language": language,
            "question": question,
            "reference": ref,
            "generated": gen,
            "bleu": bleu_score,
            "rouge": rouge_scores
        })
    return results
