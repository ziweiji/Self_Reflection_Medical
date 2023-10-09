from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
from nltk.tokenize import sent_tokenize
import numpy as np
import jsonlines
from tqdm import tqdm
import os
    
def classify_text(model, tokenizer, text):
    encoding = tokenizer.encode_plus(text, return_tensors="pt")
    input_ids = encoding["input_ids"].to(model.device)
    attention_masks = encoding["attention_mask"].to(model.device)
    try:
        outputs = model.generate(
            input_ids=input_ids, 
            attention_mask=attention_masks,
            max_length=8,
            early_stopping=True
        )
    except:
        print("classify_text", text, len(tokenizer.encode(text)))
    result = tokenizer.decode(outputs[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    if "contradiction" in result:
        return "contradiction", -1
    if "entailment" in result:
        return "entailment", 1
    return "neutral", 0
    
    
    
def truncate(tokenizer, text):
    ids = tokenizer.encode(text)
    return tokenizer.decode(ids[:2000])
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--out_file', type=str, default='results.jsonl')
    args = parser.parse_args()
    
    tokenizer = AutoTokenizer.from_pretrained("razent/SciFive-large-Pubmed_PMC-MedNLI")  
    model = AutoModelForSeq2SeqLM.from_pretrained("razent/SciFive-large-Pubmed_PMC-MedNLI")
    model.cuda()
    
    data_file = args.data_file
    out_file = args.out_file
    if os.path.exists(data_file):

        overall_results_compare_answer = []
        sent_results_compare_answer = []
        overall_results_compare_context = []
        sent_results_compare_context = []

        with jsonlines.open(data_file) as reader, jsonlines.open(out_file, mode='w') as writer:
            for line in tqdm(reader):
                generated_answer = line['generated_answer']
                generated_answer_sents = sent_tokenize(generated_answer)
                answer = line['answer'][0]
                answer = truncate(tokenizer, answer)
                text = f"mednli: sentence1: {answer} sentence2: {generated_answer}"
                res = classify_text(model, tokenizer, text)
                line.update({"overall_compare_answer": res[0]})
                overall_results_compare_answer.append(res[1])

                sent_each_sample = []
                for i, sent in enumerate(generated_answer_sents):
                    text = f"mednli: sentence1: {answer} sentence2: {sent}"
                    res = classify_text(model, tokenizer, text)
                    line.update({f"sent_{i}_compare_answer": [sent, res[0]]})
                    sent_each_sample.append(res[1])
                sent_results_compare_answer.append(np.nanmean(sent_each_sample))


#                         if line['context']:
#                             context = line['context']
#                             text = f"mednli: sentence1: {context} sentence2: {generated_answer}"
#                             res = classify_text(model, tokenizer, text)
#                             line.update({"overall_compare_context": res[0]})
#                             overall_results_compare_context.append(res[1])

#                             sent_each_sample = []
#                             for i, sent in enumerate(generated_answer_sents):
#                                 text = f"mednli: sentence1: {context} sentence2: {sent}"
#                                 res = classify_text(model, tokenizer, text)
#                                 line.update({f"sent_{i}_compare_context": [sent, res[0]]})
#                                 sent_each_sample.append(res[1])
#                             sent_results_compare_context.append(np.nanmean(sent_each_sample))

                writer.write(line)

        out_file = '.'.join(out_file.split('.')[:-1])+".txt"
        with open(out_file, 'w') as f:
            f.write(f'compare with answer\n')
            f.write(f"{np.nanmean(overall_results_compare_answer)}\t{np.nanmean(sent_results_compare_answer)}\n")

            if overall_results_compare_context:
                f.write('compare with context\n')
                f.write(f"{np.nanmean(overall_results_compare_context)}\t{np.nanmean(sent_results_compare_context)}")


    else:
        print(f'no exists {data_file}')