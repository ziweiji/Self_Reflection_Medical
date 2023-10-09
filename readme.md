
# Environment settings
```
conda create --name hallucination python=3.8.16
conda activate hallucination
pip install -r requirements.txt
(you need to change the cuda version if necessary)
```
# Dataset
Download the raw datasets to local folder ~/dataset:
[pubmedqa](https://github.com/pubmedqa/pubmedqa) 
[MedQuAD](https://github.com/abachaa/MedQuAD) 
[MEDIQA2019](https://github.com/abachaa/MEDIQA2019/tree/master/MEDIQA_Task3_QA) 
[mashqa](https://github.com/mingzhu0527/MASHQA) 
[LiveQA_MedicalTask_TREC2017](https://github.com/abachaa/LiveQA_MedicalTask_TREC2017) 

## Data processing
run convert.ipynb in each dataset folder


# Models
## Vicuna
Clone the official [repository](https://github.com/lm-sys/FastChat) and navigate to the FastChat folder.

### Directly Generate (Baseline)
```
CUDA_VISIBLE_DEVICES=0,1 python generate.py \
--model-name [path to vicuna 7B] \
--num-gpus 2\
--input_file input.jsonl \
--out_file output.jsonl
```

## Generate with Self-Reflection Loop (Ours)

```
CUDA_VISIBLE_DEVICES=0,1 python3 loop.py \
--model-name [path to vicuna 7B]\
--num-gpus 2 \
--input-file dataset/{source}/test_data.jsonl \
--sources 'pubmedqa' \
--out-dir output \
--max-loop 3 \
--max-knowledge-loop 3 \
--max-response-loop 3 \
--gptscore-model "vicuna" \
--demo-num 1 \
--threshold-entailment 0.8 \
--threshold-fact -1.0 \
--threshold-consistency -5
```
## Alpaca-Lora
Clone the official [repository](https://github.com/tloen/alpaca-lora) and navigate to the alpaca-lora folder.

### Directly Generate (Baseline)
```
CUDA_VISIBLE_DEVICES=0 python generate.py \
--input_file input.jsonl \
--out_file output.jsonl
```

## Generate with Self-Reflection Loop (Ours)

```
CUDA_VISIBLE_DEVICES=0 python3 loop.py \
--input-file dataset/{source}/test_data.jsonl \
--sources 'pubmedqa' \
--out-dir output \
--max-loop 3 \
--max-knowledge-loop 3 \
--max-response-loop 3 \
--gptscore-model "Alpaca_Lora" \
--demo-num 1 \
--threshold-entailment 0.8 \
--threshold-fact -1.0 \
--threshold-consistency -5
```
## ChatGPT
navigate to the ChatGPT folder.

### Directly Generate (Baseline)
```
CUDA_VISIBLE_DEVICES=0 python generate.py \
--input_file input.jsonl \
--out_file output.jsonl
```

## Generate with Self-Reflection Loop (Ours)
```
CUDA_VISIBLE_DEVICES=0 python3 loop.py \
--input-file dataset/{source}/test_data.jsonl \
--sources 'pubmedqa' \
--out-dir output \
--max-loop 3 \
--max-knowledge-loop 3 \
--max-response-loop 3 \
--demo-num 1 \
--threshold-entailment 0.8 \
--threshold-fact -1 \
--threshold-consistency -5 
```

## MedAlpaca
Clone the official [repository](https://github.com/kbressem/medAlpaca) and navigate to the medAlpaca folder.
### Directly Generate (Baseline)
```
CUDA_VISIBLE_DEVICES=0 python generate.py \
--input_file input.jsonl \
--out_file output.jsonl
```

## Generate with Self-Reflection Loop (Ours)
```
CUDA_VISIBLE_DEVICES=0 python3 loop.py \
--input-file dataset/{source}/test_data.jsonl \
--sources 'pubmedqa' \
--out-dir output \
--max-loop 3 \
--max-knowledge-loop 3 \
--max-response-loop 3 \
--demo-num 1 \
--threshold-entailment 0.8 \
--threshold-fact -1 \
--threshold-consistency -5
```


## Robin-Medical
Clone the official [repository](https://github.com/OptimalScale/LMFlow) and navigate to the LMFlow folder.

### Directly Generate (Baseline)
```
CUDA_VISIBLE_DEVICES=0 python generate.py \
--input_file input.jsonl \
--out_file output.jsonl
```

# Metrics
## GPTScore
Please refers to [GPTScore](https://github.com/jinlanfu/GPTScore)


## MedNLI
```
CUDA_VISIBLE_DEVICES=0 python compute_MedNLI.py \
--data_file generated_answers.jsonl \
--out_file MedNLI_results.jsonl 
```
## CTRLEval
```
git clone https://github.com/thu-coai/CTRLEval

cd CTRLEval
CUDA_VISIBLE_DEVICES=0 python compute_CTRL.py \
--data_file generated_answers.jsonl \
--out_file CTRLEval_results.csv 
```



