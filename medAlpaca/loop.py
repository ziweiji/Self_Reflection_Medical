#!/usr/bin/env python
# coding: utf-8
import argparse
import os
n = os.getcwd().split('/')[2]
import sys

import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import json
import re
import jsonlines
from tqdm import tqdm

from medalpaca.inferer import Inferer

import sys
sys.path.append(f'/home/{n}/hallucination_LLM/evaluate')

from sent_similarity import Sent_Similar
from CTRLEval.ctrleval import CTRLEval
import numpy as np
from GPTScore.gpt3_score import gpt3score

from loop_eval_utils import evaluate_response, evaluate_knowledge

sys.path.append(f'/home/{n}/hallucination_LLM')
from loop_utils import main_loop

def strip_special_chars(input_str):
    "Remove special characters from string start/end"
    if not input_str:
        return input_str
    
    start_index = 0
    end_index = len(input_str) - 1

    while start_index < len(input_str) and input_str[start_index] not in string.ascii_letters + string.digits:
        start_index += 1

    while end_index >= 0 and input_str[end_index] not in string.ascii_letters + string.digits:
        end_index -= 1

    if start_index <= end_index:
        return input_str[start_index:end_index + 1]
    else:
        return ""

def starts_with_capital_letter(input_str):
    """
    The answers should start like this: 
        'A: '
        'A. '
        'A '
    """
    pattern = r'^[A-Z](:|\.|) .+'
    return bool(re.match(pattern, input_str))


def generate_step(args, infer, prompt):
    
    input_tokens = infer.data_handler.tokenizer(prompt, return_tensors="pt")
    input_token_ids = input_tokens["input_ids"].to("cuda")
    
    generation_kwargs = {
        "num_beams" : args.num_beams, 
        "do_sample" : args.do_sample,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "temperature": args.temperature,
        "max_new_tokens" : args.max_new_tokens, 
        "early_stopping" : True
        }
    generation_config = GenerationConfig(**generation_kwargs)

    with torch.no_grad():
        generation_output = infer.model.generate(
            input_ids=input_token_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=generation_kwargs["max_new_tokens"],
        )
    generation_output_decoded = infer.data_handler.tokenizer.decode(generation_output.sequences[0])
    split = f'{infer.data_handler.prompt_template["output"]}{None or ""}'
    response = generation_output_decoded.split(split)[-1].strip()
    res = re.search("###", response)
    if res:
        response = response[:res.span()[0]].strip()
    return response

    
def knowledge_loop(args, model, tokenizer, question, knowledge_loop_list=[]):
    print("knowledge_loop")
    THRESHOLD_FACTUAL = args.threshold_fact
    MAX_KNOWLEDGE_LOOP = args.max_knowledge_loop
    candidates = []
    history = []

    instruction = "Provide background knowledge to answer the given question."
    prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{question}\n\n### Response:"
    if knowledge_loop_list:
        knowledge = knowledge_loop_list[0]
    else:
        knowledge = generate_step(args, infer, prompt)
        
    loop_i = 0
    if MAX_KNOWLEDGE_LOOP > 1:
        if args.gptscore_model == 'gpt3':
            factuality_score = evaluate_knowledge('gpt3', args.demo_num, question, knowledge)
        else:
            
            factuality_score = evaluate_knowledge(model, args.demo_num, question, knowledge, tokenizer)
        candidates.append([factuality_score, knowledge])
        history.append([loop_i, knowledge, factuality_score])
    
    # refine knowledge
    loop_i += 1
    while (loop_i < MAX_KNOWLEDGE_LOOP) and factuality_score<THRESHOLD_FACTUAL:
        instruction = f"The factuality score for the knowledge is {factuality_score} less than {THRESHOLD_FACTUAL}, which means the knowledge is not strongly supported by empirical evidence. Please refine the knowledge to improve its factuality."
        
        prompt = f"{prompt}\n{knowledge}\n\nInstruction:\n{instruction}\n\n### Input:\n\n### Response:"
        knowledge = generate_step(args, infer, prompt)
        # print('==========\n', knowledge)
        
        if args.gptscore_model == 'gpt3':
            factuality_score = evaluate_knowledge('gpt3', args.demo_num, question, knowledge)
        else:
            factuality_score = evaluate_knowledge(model, args.demo_num, question, knowledge, tokenizer)
            
        candidates.append([factuality_score, knowledge])
        history.append([loop_i, knowledge, factuality_score])
        loop_i += 1
        
            
    if (MAX_KNOWLEDGE_LOOP > 1) and factuality_score<THRESHOLD_FACTUAL:
        # still not satisified, highest_score
        candidates.sort()
        return candidates[-1][-1], history
    else:
        return knowledge, history

    
def response_loop(args, model, tokenizer, question, final_knowledge):
    print("response_loop")
    THRESHOLD_CONS = args.threshold_consistency
    MAX_RESPONSE_LOOP = args.max_response_loop
    candidates = []
    entailment_score_question_list = []
    history = []
    
    instruction = f'''Refer to the knowledge: "{final_knowledge}" and answer the question: "{question}" with one paragraph.'''
    prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.\n\n### Instruction:\n{instruction}\n\n### Input:\n{question}\n\n### Response:"
    response = generate_step(args, infer, prompt)
    
    loop_i = 0
    if MAX_RESPONSE_LOOP > 1:
        entailment_score_question, cons_score_knowledge = evaluate_response(entailment_scorer, ctrleval_scorer, question, response, final_knowledge)
        candidates.append([(entailment_score_question+cons_score_knowledge)/2, response])
        entailment_score_question_list.append(entailment_score_question)
        history.append([loop_i, response, entailment_score_question, cons_score_knowledge])
    
    loop_i += 1
    while loop_i < MAX_RESPONSE_LOOP and cons_score_knowledge<THRESHOLD_CONS:
        instruction = f"The consistency score for the knowledge is {cons_score_knowledge} less than {THRESHOLD_CONS}, which means the alignment and consistency between responses and knowledge is low. Please refine the response to improve its consistency."
        prompt = f"{prompt}\n{response}\n\nInstruction:\n{instruction}\n\n### Input:\n\n### Response:"
        response = generate_step(args, infer, prompt)
        # print('==========\n', response)
        
        entailment_score_question, cons_score_knowledge = evaluate_response(entailment_scorer, ctrleval_scorer, question, response, final_knowledge)
        candidates.append([(entailment_score_question+cons_score_knowledge)/2, response])
        entailment_score_question_list.append(entailment_score_question)
        history.append([loop_i, response, entailment_score_question, cons_score_knowledge])
        
        loop_i += 1
        
    if MAX_RESPONSE_LOOP > 1 and cons_score_knowledge<THRESHOLD_CONS:
        # still not satisified, highest_score
        merge = zip(candidates, entailment_score_question_list)
        merge = sorted(merge)
        candidates, entailment_score_question_list = zip(*merge)
        return candidates[-1][-1], history, entailment_score_question_list[-1] #max
    else:
        return response, history, entailment_score_question
        

        
    
parser = argparse.ArgumentParser()
parser.add_argument("--input-file", type=str)
parser.add_argument('--continue-generate', action='store_true')
parser.add_argument("--max-sample", type=int, default=5000)

parser.add_argument("--out-dir", type=str, default="medAlpaca_7B_loop")
parser.add_argument('--sources', nargs='+', required=True)
parser.add_argument("--max-loop", type=int, default=1)
parser.add_argument("--max-knowledge-loop", type=int, default=1)
parser.add_argument("--max-response-loop", type=int, default=1)
parser.add_argument("--gptscore-model", type=str)
parser.add_argument("--demo-num", type=int, default=0)

parser.add_argument("--threshold-entailment", type=float, default=0.8)
parser.add_argument("--threshold-fact", type=float, default=-1)
parser.add_argument("--threshold-consistency", type=float, default=-5)

parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument('--do-sample', action='store_true')
parser.add_argument("--top_p", type=float, default=1)
parser.add_argument("--top_k", type=int, default=1)
parser.add_argument("--max-new-tokens", type=int, default=128)#128
parser.add_argument("--num_beams", type=int, default=1)

args = parser.parse_args()



if args.max_response_loop > 1:
    ctrleval_scorer = CTRLEval(device='cuda') #consistency
# if args.max_knowledge_loop > 1:
entailment_scorer = Sent_Similar()
    
THRESHOLD_ENTAIL = args.threshold_entailment
MAX_LOOP = args.max_loop



infer = Inferer(
        model_name='medalpaca/medalpaca-lora-7b-8bit',
        prompt_template="medalpaca/prompt_templates/medalpaca.json", #没用
        base_model='decapoda-research/llama-7b-hf',
        peft=True,
        load_in_8bit=True,
    ) 

model = infer.model.base_model
tokenizer = infer.data_handler.tokenizer

out_dir = f"{args.out_dir}_MaxL{args.max_loop}_MaxKL{args.max_knowledge_loop}MaxRL{args.max_response_loop}_ThE{args.threshold_entailment}ThF{args.threshold_fact}ThC{args.threshold_consistency}_{args.gptscore_model}_Demo{args.demo_num}"
os.makedirs(out_dir, exist_ok=True)

for source in args.sources:
    print(source)
    input_file = args.input_file.format(source=source)
    if args.top_p == 1 and args.num_beams == 1:
        out_file = f'{out_dir}/{source}_T{args.temperature}.jsonl'
    elif args.top_p != 1:
        out_file = f'{out_dir}/{source}_T{args.temperature}_P{args.top_p}.jsonl'
    elif args.num_beams != 1:
        out_file = f'{out_dir}/{source}_T{args.temperature}_B{args.num_beams}.jsonl'
    
    if args.continue_generate and os.path.exists(out_file):
        print("continue generate")
        with jsonlines.open(out_file) as reader:
            old_lines = list(reader)
        with jsonlines.open(input_file) as reader:
            reader = list(reader)
            for i, line in tqdm(enumerate(reader), total=len(reader)):
                if i < len(old_lines):
                    continue
                if i > args.max_sample:
                    break
                
                final_knowledge, final_response, all_history_knowledge, all_history_response = main_loop(args, line, model, tokenizer, knowledge_loop, response_loop)
                
                line.update({'history_knowledge': all_history_knowledge})
                line.update({'history_response': all_history_response})
                line.update({'generated_knowledge': final_knowledge})
                line.update({'generated_answer': final_response})
                
                writer = jsonlines.open(out_file, mode='a')
                writer.write(line)
                writer.close()
                
    else:
        with jsonlines.open(input_file) as reader:
            reader = list(reader)
            for i, line in tqdm(enumerate(reader), total=len(reader)):
                if i > args.max_sample:
                    break
                final_knowledge, final_response, all_history_knowledge, all_history_response = main_loop(args, line, model, tokenizer, knowledge_loop, response_loop)
                
                line.update({'history_knowledge': all_history_knowledge})
                line.update({'history_response': all_history_response})
                line.update({'generated_knowledge': final_knowledge})
                line.update({'generated_answer': final_response})
                
                writer = jsonlines.open(out_file, mode='a')
                writer.write(line)
                writer.close()