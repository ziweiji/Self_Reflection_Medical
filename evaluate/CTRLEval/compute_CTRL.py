from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import argparse
import numpy as np
import jsonlines
from tqdm import tqdm
import csv
from ctrleval import CTRLEval
import re
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=str)
    parser.add_argument('--out_file', type=str, default='results.csv')
    args = parser.parse_args()
    
    ctrleval_scorer = CTRLEval(device='cuda') 
    
    data_file = args.data_file
    out_file = args.out_file
    if os.path.exists(data_file):
        all_results, all_results2 = [], []
        with jsonlines.open(data_file) as reader, \
        open(out_file, 'w') as fout:
            writer = csv.writer(fout)
            for line in tqdm(reader):
                try:
                    question = line['question']
                except:
                    print(data_file)
#                     print(line)
                    exit()
                generated_answer = line['generated_answer']
                generated_answer = re.sub("\u200b", " ", generated_answer)
                if generated_answer.strip():
                    prefix = [question]
                    data = [question+'\n'+generated_answer]
                    try:
                        cons_result = ctrleval_scorer.score(aspect='cons', data=data, prefix=prefix, batch_size=1)
                        exp_cons_result = np.exp(cons_result[0])
                    except:
                        print(data_file)
                        print(data)
                        exit()
                else:
                    cons_result = [np.nan]
                    exp_cons_result = 0


                writer.writerow([cons_result[0], exp_cons_result])
                all_results.append(cons_result[0])
                all_results2.append(exp_cons_result)

        out_file = out_file[:-3]+"txt"
        with open(out_file, 'w') as f:
            f.write(f'average CTRLEval\n{np.nanmean(all_results)}\n')
            f.write(f'average CTRLEval exp \n{np.nanmean(all_results2)}')

    else:
        print(f"there is no {data_file}")