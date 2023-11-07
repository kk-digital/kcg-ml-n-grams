import sys
sys.path.append('.')

import pandas as pd
import numpy as np

from utility.msgpack_dataloader import read_msg_pack
from itertools import chain
from collections import Counter
from transformers import CLIPTokenizer, CLIPTextModel

import glob
import tqdm
import os
import torch
import argparse
import transformers

transformers.logging.set_verbosity_error()

tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')
model = CLIPTextModel.from_pretrained('openai/clip-vit-large-patch14').eval().to('cuda')

# max token length and character length constraints for n-grams
ngram_constraints = {
    1: {'max_token_length': 20, 'max_phrase_length': 150},
    2: {'max_token_length': 40, 'max_phrase_length': 150},
    3: {'max_token_length': 60, 'max_phrase_length': 200}
}

def create_ngrams(vocab, n):
    ngram_data = []
    max_index = len(vocab) - n + 1

    if n <= 0 or n > len(vocab):
        return []

    for i in range(max_index):
        ngram = ', '.join(vocab[i:i + n])
        ngram_data.append(ngram)

    return ngram_data

def get_token_length(row):
    with torch.no_grad():
        token_encoding = tokenizer(row['phrase str'], return_length=True, return_tensors='pt')

    return token_encoding['length'].item()

def filter_phrase(df, max_phrase_length=150, max_token_length=20):
    # Remove rows with unicode characters
    df = df[df['phrase str'].apply(lambda x: isinstance(x, str) and all(ord(char) < 128 for char in x))]

    # Remove rows with tab
    df = df[df['phrase str'].str.contains('\t|\n|\r|\b') == False]

    # Remove rows with more than 100 characters
    df = df[(df['phrase str'].str.len() <= max_phrase_length) & (df['phrase str'].str.len() > 0)]

    # Remove rows with token length more than 20
    df = df[df['token_length'] <= max_token_length]

    return df

def count_positive_negative_phrase(df, ngram):
    positive_prompt_tokens = df['positive_prompt'].str.split(', ').tolist()
    negative_prompt_tokens = df['negative_prompt'].str.split(', ').tolist()

    positive_prompt_ngrams = [create_ngrams(prompts, ngram) for prompts in positive_prompt_tokens]
    negative_prompt_ngrams = [create_ngrams(prompts, ngram) for prompts in negative_prompt_tokens]

    positive_tokens_chain = list(chain.from_iterable(positive_prompt_ngrams))
    negative_tokens_chain = list(chain.from_iterable(negative_prompt_ngrams))

    phrases = positive_tokens_chain + negative_tokens_chain
    phrases = list(set(phrases))

    positive_phrase_counts = pd.DataFrame(Counter(positive_tokens_chain), index=['positive count']).T
    negative_phrase_counts = pd.DataFrame(Counter(negative_tokens_chain), index=['negative count']).T

    df_phrase_counts = positive_phrase_counts.merge(negative_phrase_counts, how='outer', left_index=True, right_index=True)
    df_phrase_counts = df_phrase_counts.fillna(0).astype(np.int32)
    df_phrase_counts['phrase str'] = df_phrase_counts.index
    df_phrase_counts['token_length'] = df_phrase_counts.apply(get_token_length, axis=1)
    df_phrase_counts = df_phrase_counts.reset_index(drop=True)

    df_phrase_counts = filter_phrase(df_phrase_counts)
    df_phrase_counts = df_phrase_counts[['phrase str', 'token_length', 'positive count', 'negative count']]

    return df_phrase_counts

def main(data_path, ngram, save_path):
    file_paths = sorted(glob.glob(os.path.join(data_path, '*_embedding.msgpack')))
    df = []
    for file_path in tqdm.tqdm(file_paths):
        data = read_msg_pack(file_path)
        del data['positive_embedding']
        del data['negative_embedding']
        df.append(data)

    df = pd.DataFrame(df)
    df_phrase_counts = count_positive_negative_phrase(df, ngram)

    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, 'environment_data.csv'), index=False)
    df_phrase_counts.to_csv(os.path.join(save_path, f'environment_{ngram}-gram.csv'), index=False)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_path', type=str, help='Path to environment embedding msgpack')
    ap.add_argument('--ngram', type=int, help='ngram to extract')
    ap.add_argument('--save_path', type=str, help='Path to save output files')
    args = ap.parse_args()

    main(
        data_path=args.data_path,
        ngram=args.ngram,
        save_path=args.save_path
    )