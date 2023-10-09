import argparse
import os

import sys

import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer

from utils.callbacks import Iteratorize, Stream
from utils.prompter import Prompter

import jsonlines
from tqdm import tqdm
import time
n = os.getcwd().split('/')[2]


def generate_step(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    
    input_ids = inputs["input_ids"].to(device)
    generation_config = GenerationConfig(
        temperature=1.0,
        top_p=1.0,
        top_k=1,
        num_beams=1,
        early_stopping=True,
    )

    # Without streaming
    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=512,
            
        )
    s = generation_output.sequences[0][len(input_ids[0]):]
    output = tokenizer.decode(s)
    return output

parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--out_file", type=str)
args = parser.parse_args()

device = "cuda"
load_8bit = True
base_model = 'decapoda-research/llama-7b-hf'
lora_weights = 'tloen/alpaca-lora-7b'

tokenizer = LlamaTokenizer.from_pretrained(base_model)

model = LlamaForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=load_8bit,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(
    model,
    lora_weights,
    torch_dtype=torch.float16,
)

# unwind broken decapoda-research config
model.config.pad_token_id = tokenizer.pad_token_id = 0  # unk
model.config.bos_token_id = 1
model.config.eos_token_id = 2
# if load_8bit:
#     model.half()
model.eval()
model = torch.compile(model)

out_file = args.out_file
with jsonlines.open(args.input_file) as reader:
    reader = list(reader)
    for i, line in tqdm(enumerate(reader), total=len(reader)):
        question = line['question']

        prompter = Prompter('')
        instruction = "Answer the following question."
        prompt = prompter.generate_prompt(instruction, question)
        response = generate_step(prompt)

        line.update({'generated_answer': response})
        writer = jsonlines.open(out_file, mode='a')
        writer.write(line)
        writer.close()
        