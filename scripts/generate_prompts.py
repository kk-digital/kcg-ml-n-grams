import sys
sys.path.append('.')

from transformers import CLIPTokenizer, CLIPTextModel
from utility.msgpack_dataloader import write_msgpack

import os
import argparse
import torch
import tqdm
import transformers

import pandas as pd
import numpy as np

transformers.logging.set_verbosity_error()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").eval().to('cuda')

def sampling_algo1(data_path):
    df = pd.read_csv(data_path)
    df['cum log prob'] = df['log probability'].cumsum()
    df['cum prob'] = df['probability'].cumsum()

    generated_prompt = []
    prompt_length = 0
    joint_prob = []
    joint_log_prob = []
    while True:
        # sample phrase
        random_number = np.random.rand(1)[0]
        row = df[df['cum prob'] >= random_number].iloc[0]
        phrase = row['phrase str']

        # check if prompt length exceeds 75 tokens
        # tokens = tokenizer.tokenize(phrase + ', ')
        prompt_length += row['token_length']
        if prompt_length >= 75:
            break

        generated_prompt.append(phrase)
        joint_prob.append(row['probability'])
        joint_log_prob.append(row['log probability'])

    generated_prompt = ', '.join(generated_prompt)
    joint_prob = np.product(joint_prob)
    joint_log_prob = np.sum(joint_log_prob)

    with torch.no_grad():
        token_encoding = tokenizer(generated_prompt, return_length=True, return_tensors='pt')
        embedding = model(input_ids=token_encoding['input_ids'].to('cuda')).last_hidden_state[0]
    data = {
        'positive_prompt': generated_prompt,
        'token_length': token_encoding['length'].item(),
        'joint_prob': joint_prob,
        'joint_log_prob': joint_log_prob,
        'positive_embedding': embedding.cpu().numpy().tolist()
    }

    return data

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_csv_path', type=str, required=True, help='Path to CSV file with phrases. Must have probability and log probability columns')
    ap.add_argument('--generate1', action='store_true', default=False, help='Generate 1 sample and print to terminal.')
    ap.add_argument('--n_prompt', default=100, type=int, help='Number of random prompts to generate.')
    ap.add_argument('--save_path', type=str, help='Folder path to save msgpack files')
    args = ap.parse_args()

    if args.generate1:
        data = sampling_algo1(args.data_csv_path)
        print(data)

    else:
        os.makedirs(args.save_path, exist_ok=True)
        print('[*] Generating random prompts')
        for i in tqdm.tqdm(range(args.n_prompt)):
            data = sampling_algo1(args.data_csv_path)
            write_msgpack(
                obj=data,
                file_path=os.path.join(args.save_path, f'{str(i).zfill(7)}.msgpack')
            )