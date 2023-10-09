import os
import torch
import torch.nn as nn
import traceback

import transformers
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers import GPT2Tokenizer, OPTForCausalLM, GPT2LMHeadModel, GPTJForCausalLM
import sys
import json

def trunk_input(tokenizer, inputs, outputs, reduce_seq, max_length):
    input_ids = tokenizer.encode(inputs)[1:-1]
    output_ids = tokenizer.encode(outputs)[1:-1]
    reduce_seq_ids = tokenizer.encode(reduce_seq)[1:-1]
    total_len = len(input_ids) + len(output_ids)
    if total_len > max_length:
        del_len = len(input_ids) + len(output_ids) - max_length
        reduce_seq_ids = reduce_seq_ids[:len(reduce_seq_ids) - del_len]
        reduce_seq = tokenizer.decode(reduce_seq_ids[1:-1])
    return reduce_seq

    
def directly_get_score(model, tokenizer, srcs, tgts, prompt_text, device="cuda", max_length=2000):
    score_list = []
    for i,(src, tgt) in enumerate(zip(srcs, tgts)):
        new_src = trunk_input(tokenizer, src, tgt, src, max_length)
        src = new_src
        text = src + prompt_text + tgt
        input_ids = tokenizer.encode(text)
        tgt_ids = tokenizer.encode(tgt)[1:]
        output_ids = [-100] * len(input_ids)
        output_ids[len(input_ids) - len(tgt_ids):] = tgt_ids
        input_ids = torch.LongTensor(input_ids).unsqueeze(0).to(device)
        output_ids = torch.LongTensor(output_ids).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                labels=output_ids,
                output_hidden_states=True
            )
        loss, logits, hidden_states = outputs[0], outputs[1], outputs.hidden_states[0]
        loss = loss.item()
        score = -loss
        score_list.append(score)
    return score_list


