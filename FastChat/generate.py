import argparse
import os
import jsonlines
from tqdm import tqdm

import os
n = os.getcwd().split('/')[2]
import re


def generate_step(prompt, conv):
    # print('prompt', prompt)
    stop_str = conv.sep if conv.sep_style == SeparatorStyle.SINGLE else conv.sep2
    
    input_tokens = tokenizer(prompt, return_tensors="pt")
    input_token_ids = input_tokens["input_ids"].to("cuda")


    with torch.no_grad():
        generation_output = model.generate(
            input_ids=input_token_ids,
            max_new_tokens=args.max_new_tokens,
            early_stopping=True,
            num_beams=args.num_beams,
            temperature=args.temperature,
            do_sample=args.do_sample,
            top_k=args.top_k,
            top_p=args.top_p,
        )
    output = tokenizer.decode(generation_output[0][len(input_token_ids[0]):])
    result = re.search(stop_str, output)
    if result:
        prompt = prompt[:result.span()[0]]
    return output


parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--out_file", type=str)
parser.add_argument("--model-name", type=str, default="vicuna_7B")
parser.add_argument("--device", type=str, default="cuda")
parser.add_argument("--num-gpus", type=str, default="2")
parser.add_argument("--load-8bit", action="store_true")
parser.add_argument("--conv-template", type=str, default="v1")
parser.add_argument("--temperature", type=float, default=1.0)
parser.add_argument("--max-new-tokens", type=int, default=512)
parser.add_argument("--top_k", type=int, default=1)
parser.add_argument("--num_beams", type=int, default=1)
parser.add_argument("--top_p", type=float, default=1.0)
parser.add_argument("--do_sample", action="store_true")
parser.add_argument("--debug", action="store_true")
args = parser.parse_args()

args.load_8bit = False
args.debug = False

from fastchat.serve.cli import generate_stream, load_model
from fastchat.conversation import conv_templates, SeparatorStyle
from transformers import GenerationConfig

# Model
model, tokenizer = load_model(args.model_name, args.device, args.num_gpus, args.load_8bit, args.debug)


out_file = args.out_file
with jsonlines.open(args.input_file) as reader:
    reader = list(reader)
    for i, line in tqdm(enumerate(reader), total=len(reader)):
        question = line['question']

        conv = conv_templates[args.conv_template].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        response = generate_step(prompt, conv)

        line.update({'generated_answer': response})

        writer = jsonlines.open(out_file, mode='a')
        writer.write(line)
        writer.close()
        
            
                