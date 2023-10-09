import argparse
import argparse
import os
import sys

import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import re
import json
import string
import time
from tqdm.autonotebook import tqdm
from medalpaca.inferer import Inferer

import jsonlines
from tqdm import tqdm
import os
n = os.getcwd().split('/')[2]

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


model = Inferer(
        model_name='medalpaca/medalpaca-lora-7b-8bit',
        prompt_template="medalpaca/prompt_templates/medalpaca.json",
        base_model='decapoda-research/llama-7b-hf',
        peft=True,
        load_in_8bit=True,
    ) 


def generate_step(prompt):
    generation_kwargs = {
    "num_beams" : 1, 
    "do_sample" : False,
    "max_new_tokens" : 128, 
    "early_stopping" : True
    }
    
    input_tokens = model.data_handler.tokenizer(prompt, return_tensors="pt")
    input_token_ids = input_tokens["input_ids"].to("cuda")

    generation_config = GenerationConfig(**generation_kwargs)

    with torch.no_grad():
        generation_output = model.model.generate(
            input_ids=input_token_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=generation_kwargs["max_new_tokens"],
        )
    generation_output_decoded = model.data_handler.tokenizer.decode(generation_output.sequences[0])
    split = f'{model.data_handler.prompt_template["output"]}{None or ""}'
    response = generation_output_decoded.split(split)[-1].strip()
    res = re.search("###", response)
    if res:
        response = response[:res.span()[0]].strip()
    return response


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--out_file", type=str)
args = parser.parse_args()

out_file = args.out_file
with jsonlines.open(args.input_file) as reader:
    for i, line in tqdm(enumerate(reader)):
        question = line['question']
        response = model(instruction="Answer this question.", 
                         input=question,
                         output="The Answer to the question is:",
                         **greedy_search)
        response = strip_special_chars(response)
        res = re.search("### References", response)
        if res:
            response = response[:res.span()[0]].strip()

        line.update({'generated_answer': response})
        writer = jsonlines.open(out_file, mode='a')
        writer.write(line)
        writer.close()
