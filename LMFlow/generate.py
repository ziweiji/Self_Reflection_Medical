# env lmflow
import argparse
import os
import jsonlines
from tqdm import tqdm
import torch
import transformers
from peft import PeftModel
from transformers import GenerationConfig, LlamaForCausalLM, LlamaTokenizer
import json
import os
import sys
sys.path.append("src/")
from transformers import HfArgumentParser

from lmflow.datasets.dataset import Dataset
from lmflow.pipeline.auto_pipeline import AutoPipeline
from lmflow.models.auto_model import AutoModel
from lmflow.args import ModelArguments, DatasetArguments, AutoArguments


def generate_step(prompt):
    inputs = model.encode(prompt, return_tensors="pt").to('cuda')
    outputs = model.inference(
        inputs,
        max_new_tokens=512,
        temperature=1.0,
        repetition_penalty=1.0,
        do_sample=False,
    )
    text_out = model.decode(outputs[0], skip_special_tokens=True)

    # only return the generation, trucating the input
    prompt_length = len(model.decode(inputs[0], skip_special_tokens=True,))
    output = text_out[prompt_length:]
    
    return output


pipeline_name = "evaluator"
PipelineArguments = AutoArguments.get_pipeline_args_class(pipeline_name)

model_args = ModelArguments()
data_args = DatasetArguments()
pipeline_args = PipelineArguments()
pipeline_args.deepspeed = "examples/ds_config.json"
model_args.model_name_or_path = 'decapoda-research/llama-7b-hf'
model_args.lora_model_path = 'llama7b-lora-medical'

with open (pipeline_args.deepspeed, "r") as f:
    ds_config = json.load(f)

model = AutoModel.get_model(
    model_args, 
    tune_strategy='none', 
    ds_config=ds_config, 
    use_accelerator=pipeline_args.use_accelerator_for_evaluator
)



parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--out_file", type=str)
args = parser.parse_args()

with jsonlines.open(args.input_file) as reader, jsonlines.open(args.out_file, mode='w') as writer:
    reader = list(reader)
    for i, line in tqdm(enumerate(reader), total=len(reader)):
        question = line['question']
        prompt = f"Input: {question}"
        response = generate_step(prompt)

        line.update({'generated_answer': response})
        writer.write(line)
                