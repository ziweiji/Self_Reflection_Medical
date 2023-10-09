#!/usr/bin/env python
# coding: utf-8

import argparse
import os
from tqdm import tqdm

from fastchat.serve.cli import generate_stream, load_model
from fastchat.conversation import conv_templates, SeparatorStyle
import jsonlines
import re
import sys

n = os.getcwd().split('/')[2]
sys.path.append(f'/home/{n}/hallucination_LLM/evaluate')

from sent_similarity import Sent_Similar

from CTRLEval.ctrleval import CTRLEval
import numpy as np



from loop_eval_utils import evaluate_response, evaluate_knowledge

sys.path.append(f'/home/{n}/hallucination_LLM')
from loop_utils import main_loop
import torch

def generate_step(args, model, tokenizer, prompt, conv):
    params = {
        "prompt": prompt,
        "temperature": args.temperature,
        "max_new_tokens": args.max_new_tokens,
        "stop": conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2,
    }
    
    pre = 0
    
    for outputs in generate_stream(tokenizer, model, params, args.device):
        outputs = outputs[len(prompt) + 1:].strip()
        outputs = outputs.split(" ")
        
    return " ".join(outputs)

# def generate_step(args, model, tokenizer, prompt, conv):
#     # print('prompt', prompt)
#     stop_str = conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2
    
#     input_tokens = tokenizer(prompt, return_tensors="pt")
#     input_token_ids = input_tokens["input_ids"].to("cuda")


#     with torch.no_grad():
#         generation_output = model.generate(
#             input_ids=input_token_ids,
#             max_new_tokens=args.max_new_tokens,
#             early_stopping=True,
#             num_beams=args.num_beams,
#             temperature=args.temperature,
#             do_sample=args.do_sample,
#             top_k=args.top_k,
#             top_p=args.top_p,
#         )
#     output = tokenizer.decode(generation_output[0][len(input_token_ids[0]):])
#     result = re.search(stop_str, output)
#     if result:
#         prompt = prompt[:result.span()[0]]
#     return output

def knowledge_loop(args, model, tokenizer, question, knowledge_loop_list=[]):
    print("knowledge_loop")
    THRESHOLD_FACTUAL = args.threshold_fact
    MAX_KNOWLEDGE_LOOP = args.max_knowledge_loop
    candidates = []
    history = []
    
    conv = conv_templates[args.conv_template].copy()
    instruction = f'''Provide background knowledge to answer the given question: "{question}".'''
#   bad prompt  f"You are a knowledge generator, and the knowledge you generate must support:\n{question}" 

    conv.append_message(conv.roles[0], instruction)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    
    if knowledge_loop_list:
        knowledge = knowledge_loop_list[0]
    else:
        knowledge = generate_step(args, model, tokenizer, prompt, conv)
    conv.messages[-1][-1] = knowledge
    # print('======generated knowledge======\n', knowledge)
    
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
        if args.no_aspect:
            instruction = f"Please refine the knowledge."
        elif args.no_number:
            instruction = f"The knowledge is not strongly supported by empirical evidence. Please refine the knowledge to improve its factuality."
        else:
            instruction = f"The factuality score for the knowledge is {factuality_score} less than {THRESHOLD_FACTUAL}, which means the knowledge is not strongly supported by empirical evidence. Please refine the knowledge to improve its factuality."
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        knowledge = generate_step(args, model, tokenizer, prompt, conv)
        conv.messages[-1][-1] = knowledge
        # print('======generated knowledge======\n', knowledge)
        
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
    
    conv = conv_templates[args.conv_template].copy()
    instruction = f'''Refer to the knowledge: "{final_knowledge}" and answer the question: "{question}" with one paragraph.'''

    conv.append_message(conv.roles[0], instruction)
    conv.append_message(conv.roles[1], None)
    
    prompt = conv.get_prompt()
    response = generate_step(args, model, tokenizer, prompt, conv)
    conv.messages[-1][-1] = response

    loop_i = 0
#     if MAX_RESPONSE_LOOP > 1:
    entailment_score_question, cons_score_knowledge = evaluate_response(entailment_scorer, ctrleval_scorer, question, response, final_knowledge)
    candidates.append([(entailment_score_question+cons_score_knowledge)/2, response])
    entailment_score_question_list.append(entailment_score_question)
    history.append([loop_i, response, entailment_score_question, cons_score_knowledge])
        
    loop_i += 1
    while loop_i < MAX_RESPONSE_LOOP and cons_score_knowledge<THRESHOLD_CONS:
        if args.no_aspect:
            instruction = f"Please refine the response."
        elif args.no_number:
            instruction = f"The alignment and consistency between response and knowledge are low. Please refine the response to improve its consistency."
        else:
            instruction = f"The consistency score for the knowledge is {cons_score_knowledge} less than {THRESHOLD_CONS}, which means the alignment and consistency between response and knowledge are low. Please refine the response to improve its consistency."
        
        conv.append_message(conv.roles[0], instruction)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        response = generate_step(args, model, tokenizer, prompt, conv)
        conv.messages[-1][-1] = response
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
parser.add_argument("--model-name", type=str, default="vicuna_7B")
parser.add_argument("--input-file", type=str)
parser.add_argument("--continue-generate", action="store_true")
parser.add_argument("--no-number", action="store_true")
parser.add_argument("--no-aspect", action="store_true")

parser.add_argument("--out-dir", type=str, default="vicuna_7B_loop")
parser.add_argument('--sources', nargs='+', required=True)
parser.add_argument("--max-loop", type=int, default=1)
parser.add_argument("--max-knowledge-loop", type=int, default=1)
parser.add_argument("--max-response-loop", type=int, default=1)
parser.add_argument("--gptscore-model", type=str)
parser.add_argument("--demo-num", type=int, default=0)

parser.add_argument("--threshold-entailment", type=float, default=0.8)
parser.add_argument("--threshold-fact", type=float, default=-1)
parser.add_argument("--threshold-consistency", type=float, default=-5)

parser.add_argument("--max-sample", type=int, default=3000)
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--num-gpus", type=str, default="2")
parser.add_argument("--conv-template", type=str, default="v1")
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--do-sample", action="store_true")
parser.add_argument("--top_k", type=int, default=1)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--max-new-tokens", type=int, default=512)

parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--debug", action="store_true")
# args.load_8bit = False
# args.debug = False

args = parser.parse_args()


# if args.max_response_loop > 1:
ctrleval_scorer = CTRLEval(device='cuda') #consistency
# if args.max_knowledge_loop > 1:
entailment_scorer = Sent_Similar()
model, tokenizer = load_model(args.model_name, args.device, args.num_gpus, args.load_8bit, args.debug)

out_dir = f"{args.out_dir}_MaxL{args.max_loop}_MaxKL{args.max_knowledge_loop}MaxRL{args.max_response_loop}_ThE{args.threshold_entailment}ThF{args.threshold_fact}ThC{args.threshold_consistency}_{args.gptscore_model}_Demo{args.demo_num}"
os.makedirs(out_dir, exist_ok=True)

# for source in ['mashqa', 'LiveQA_MedicalTask_TREC2017', 'MEDIQA2019', 'pubmedqa', 'MedQuAD']:
for source in args.sources:
    print(source)
    input_file = args.input_file.format(source=source)
    
    if args.no_aspect:
        out_file = f'{out_dir}/{source}_T{args.temperature}_no_aspect.jsonl'
    elif args.no_number:
        out_file = f'{out_dir}/{source}_T{args.temperature}_no_number.jsonl'
    else:
        out_file = f'{out_dir}/{source}_T{args.temperature}.jsonl'
    
    
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


                


