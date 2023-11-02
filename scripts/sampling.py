from transformers import CLIPTokenizer, CLIPTextModel
from transformers.utils import logging

import torch
import transformers

import pandas as pd
import numpy as np

transformers.logging.set_verbosity_error()

df = pd.read_csv('./data/fc_logprob_results.csv')
df['cum log prob'] = df['log probability'].cumsum()
df['cum prob'] = df['probability'].cumsum()

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
model = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").eval().to('cuda')

def sampling_algo1():
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
        tokens = tokenizer.tokenize(phrase + ', ')
        prompt_length += len(tokens)
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
        'prompt': generated_prompt,
        'token_length': token_encoding['length'].item(),
        'joint_prob': joint_prob,
        'joint_log_prob': joint_log_prob,
        'embedding': embedding.cpu().numpy().tolist()
    }

    return data

if __name__ == '__main__':
    data = sampling_algo1()
    print(data['prompt'])