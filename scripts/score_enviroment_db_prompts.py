import sys
sys.path.append('.')

import pandas as pd
import numpy as np

import argparse
import torch
import tqdm
import glob
import os
import json
import transformers
import time

from utility.msgpack_dataloader import read_msg_pack
from models.ab_ranking_elm_v1 import ABRankingELMModel
from models.ab_ranking_linear import ABRankingModel

def score_prompt(model, prompt_dict):
    embedding = np.array(prompt_dict['positive_embedding']['__ndarray__'])
    embedding = torch.from_numpy(embedding).permute(0, 2, 1).float().to('cuda')
    score = model.predict_positive_or_negative_only(embedding)

    return score

def load_models(
    elm_model_weights,
    linear_model_weights
):
    elm_model = ABRankingELMModel(768)
    elm_model.load(elm_model_weights)

    linear_model = ABRankingModel(768)
    linear_model.load(linear_model_weights)

    return elm_model, linear_model

def main(
    prompt_path,
    data_csv_path,
    elm_model_weights,
    linear_model_weights,
    results_save_path
):
    prompt_file_paths = sorted(glob.glob(os.path.join(prompt_path, '*_embedding.msgpack')))
    elm_model, linear_model = load_models(
        elm_model_weights,
        linear_model_weights
    )
    # ngram / phrase csv
    # df = pd.read_csv(data_csv_path)
    ngram_list = df['phrase str'].tolist()

    df_result = []
    for path in tqdm.tqdm(prompt_file_paths):
        prompt_dict = read_msg_pack(path)

        prompt_dict['elm_score'] = score_prompt(elm_model, prompt_dict).item()
        prompt_dict['linear_score'] = score_prompt(linear_model, prompt_dict).item()
        del prompt_dict['positive_embedding']
        del prompt_dict['negative_embedding']
        df_result.append(prompt_dict)

    df_result = pd.DataFrame(df_result)
    # filter prompts that contain invalid tokens
    df_result['tokens'] = df_result['positive_prompt'].str.split(', ')
    df_result = df_result[df_result['tokens'].apply(lambda tokens: all(token in ngram_list for token in tokens))]

    # scores of prompts that have a certain phrase
    phrase_scores = {phrase: {'elm_scores': [], 'linear_scores': []} for phrase in ngram_list}

    # for each phrase, store the prompts that contain the phrase
    phrase_prompts = {phrase: [] for phrase in ngram_list}

    # store the scores for each phrase
    for idx, row in df_result.iterrows():
        tokens = row['tokens']
        for token in tokens:
            phrase_scores[token]['elm_scores'].append(row['elm_score'])
            phrase_scores[token]['linear_scores'].append(row['linear_score'])
            phrase_prompts[token].append(row['positive_prompt'])

    # compute average
    df_phrase_scores = pd.DataFrame(phrase_scores).T
    df_phrase_scores['elm_average_score'] = df_phrase_scores['elm_scores'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)
    df_phrase_scores['linear_average_score'] = df_phrase_scores['linear_scores'].apply(lambda x: np.mean(x) if len(x) > 0 else 0)

    # remove phrases with no prompts
    df_phrase_scores = df_phrase_scores[
        ~((df_phrase_scores['elm_average_score'] == 0) & (df_phrase_scores['linear_average_score'] == 0)) 
    ]

    # compute standard deviation
    df_phrase_scores['elm_std'] = df_phrase_scores['elm_scores'].apply(lambda x: np.std(x))
    df_phrase_scores['linear_std'] = df_phrase_scores['linear_scores'].apply(lambda x: np.std(x))

    # compute "sigma"
    df_phrase_scores['elm_sigma'] = df_phrase_scores['elm_average_score'].apply(
        lambda x: (x - df_phrase_scores['elm_average_score'].mean()) / df_phrase_scores['elm_average_score'].std()
    )
    df_phrase_scores['linear_sigma'] = df_phrase_scores['linear_average_score'].apply(
        lambda x: (x - df_phrase_scores['linear_average_score'].mean()) / df_phrase_scores['linear_average_score'].std()
    )

    df_phrase_scores['phrase'] = df_phrase_scores.index
    df_phrase_scores = df_phrase_scores.reset_index(drop=True)


    def percentile_rank(column):
        return column.rank(pct=True)
    
    df_phrase_scores['elm_percentile'] = percentile_rank(df_phrase_scores['elm_average_score'])
    df_phrase_scores['linear_percentile'] = percentile_rank(df_phrase_scores['linear_average_score'])

    # add token lengths to dataframe
    df_phrase_scores = df_phrase_scores.reset_index(drop=True)
    df_phrase_scores = df_phrase_scores.merge(df[['phrase str', 'token_length']], left_on='phrase', right_on='phrase str', how='left')

    # binned percentiles
    df_phrase_scores['elm_percentile_bin'] = pd.qcut(df_phrase_scores['elm_percentile'], q=4, labels=False)
    df_phrase_scores['linear_percentile_bin'] = pd.qcut(df_phrase_scores['linear_percentile'], q=4, labels=False)

    # add number of prompts for each phrase
    for phrase, prompts in phrase_prompts.items():
        if len(prompts) == 0: continue
        retrieved_row = df_phrase_scores.loc[df_phrase_scores['phrase'] == phrase]
        index = retrieved_row.index[0]
        df_phrase_scores.at[index, 'n_prompts'] = len(prompts)

    df_phrase_scores = df_phrase_scores[[
        'phrase', 'token_length', 'n_prompts', 
        'elm_average_score', 'elm_std', 'elm_percentile', 'elm_percentile_bin', 'elm_sigma',
        'linear_average_score', 'linear_std', 'linear_percentile', 'linear_percentile_bin', 'linear_sigma'
    ]]
    df_phrase_scores = df_phrase_scores.astype(
        {'token_length': int, 'n_prompts': int, 'elm_percentile_bin': int, 'linear_percentile_bin': int}
    )

    os.makedirs(results_save_path, exist_ok=True)
    with open(os.path.join(results_save_path, 'phrase_prompts.json'), 'w') as f:
        json.dump(phrase_prompts, f, indent=2)

    df_phrase_scores.to_csv(os.path.join(results_save_path, 'phrase_scores.csv'), index=False, float_format='%.15f')
    df_result.to_csv(os.path.join(results_save_path, 'results.csv'), float_format='%.15f')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt_path', type=str, required=True, help='Path to msgpack of generated prompts')
    ap.add_argument('--data_csv_path', type=str, required=True, help='Path to CSV file with phrases')
    ap.add_argument('--linear_model_weights', type=str, required=True, help='Linear ranking model weights path')
    ap.add_argument('--elm_model_weights', type=str, required=True, help='ELM ranking model weights path')
    ap.add_argument('--results_save_path', type=str, required=True, help='Results save path')

    args = ap.parse_args()

    start = time.time()
    main(
        prompt_path=args.prompt_path,
        data_csv_path=args.data_csv_path,
        linear_model_weights=args.linear_model_weights,
        elm_model_weights=args.elm_model_weights,
        results_save_path=args.results_save_path
    )
    end = time.time()

    print(f'Time taken: {end - start:.2f} seconds')