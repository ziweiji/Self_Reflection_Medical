import jsonlines
import time
import openai
from tqdm import tqdm
# Set your API key
openai.api_key = "[YOUR KEY]"
import argparse

# Set the model you want to use
model_engine = "gpt-3.5-turbo"

# Generate a response from the model
def get_response(prompt):
#     time.sleep(3)
    completion = openai.ChatCompletion.create(
        model=model_engine,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=512,
    )
    # Print the response
    response = completion.choices[0].message.content
    return response



parser = argparse.ArgumentParser()
parser.add_argument("--input_file", type=str)
parser.add_argument("--out_file", type=str)
args = parser.parse_args()


source = read_jsonl(args.input_file)
with jsonlines.open(args.out_file, 'w') as writer:
    total = len(source)
    new_lines = []
    while i < total:
        try:
            line = source[i]
            response = get_response(line['question'])
            line['generated_answer'] = response
            new_lines.append(line)
            i += 1
        except:
            print('generation wrong, regenerate')
            
            
    for line in new_lines:
        writer.write(line)
        