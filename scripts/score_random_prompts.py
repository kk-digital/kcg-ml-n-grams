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

from utility.msgpack_dataloader import read_msg_pack
from models.ab_ranking_elm_v1 import ABRankingELMModel
from models.ab_ranking_linear import ABRankingModel

def load_models(
    elm_model_weights,
    linear_model_weights
):
    elm_model = ABRankingELMModel(768)
    elm_model.load(elm_model_weights)

    linear_model = ABRankingModel(768)
    linear_model.load(linear_model_weights)

    return elm_model, linear_model

def load_data(data_csv_path):
    """Load csv data
    CSV must include 'phrase str' column

    Args:
        data_csv_path (_type_): Path to csv with phrase data
    """
    return pd.read_csv(data_csv_path)

def score_prompt(model, prompt_dict):
    embedding = np.array(prompt_dict['positive_embedding'])
    embedding = torch.from_numpy(embedding).unsqueeze(0).permute(0, 2, 1).float().to('cuda')
    score = model.predict_positive_or_negative_only(embedding)

    return score

def main(
    prompt_path,
    data_csv_path,
    elm_model_weights,
    linear_model_weights,
    results_save_path
):
    df = load_data(data_csv_path)
    prompt_file_paths = sorted(glob.glob(os.path.join(prompt_path, '*')))
    elm_model, linear_model = load_models(elm_model_weights, linear_model_weights)

    print('[*] Scoring prompts')
    df_result = []
    for path in tqdm.tqdm(prompt_file_paths):
        prompt_dict = read_msg_pack(path)

        prompt_dict['elm_score'] = score_prompt(elm_model, prompt_dict).item()
        prompt_dict['linear_score'] = score_prompt(linear_model, prompt_dict).item()
        df_result.append(prompt_dict)

    # drop the embedding column so it does not get saved into csv
    df_result = pd.DataFrame(df_result)
    df_result = df_result.drop('positive_embedding', axis=1)
    df_result['tokens'] = df_result['positive_prompt'].str.split(', ')

    # scores of prompts that have a certain phrase
    phrase_scores = {phrase: {'elm_score': [], 'linear_score': []} for phrase in df['phrase str'].tolist()}

    # for each phrase, store the prompts that contain the phrase
    phrase_prompts = {phrase: [] for phrase in df['phrase str'].tolist()}

    # store the scores for each phrase
    for idx, row in df_result.iterrows():
        tokens = row['tokens']

        for token in tokens:
            phrase_scores[token]['elm_score'].append(row['elm_score'])
            phrase_scores[token]['linear_score'].append(row['linear_score'])
            phrase_prompts[token].append(row['positive_prompt'])

    # get average scores
    df_phrase_scores = pd.DataFrame(phrase_scores).T
    df_phrase_scores = df_phrase_scores.map(lambda x: sum(x) / len(x) if len(x) > 0 else 0)

    with open(os.path.join(results_save_path, 'phrase_prompts.json'), 'w') as f:
        json.dump(phrase_prompts, f, indent=2)

    df_phrase_scores.to_csv(os.path.join(results_save_path, 'phrase_scores.csv'))
    df_result.to_csv(os.path.join(results_save_path, 'results.csv'))

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--prompt_path', type=str, required=True, help='Path to msgpack of generated prompts')
    ap.add_argument('--data_csv_path', type=str, required=True, help='Path to CSV file with phrases')
    ap.add_argument('--linear_model_weights', type=str, required=True, help='Linear ranking model weights path')
    ap.add_argument('--elm_model_weights', type=str, required=True, help='ELM ranking model weights path')
    ap.add_argument('--results_save_path', type=str, required=True, help='Results save path')

    args = ap.parse_args()

    main(
        prompt_path=args.prompt_path,
        data_csv_path=args.data_csv_path,
        linear_model_weights=args.linear_model_weights,
        elm_model_weights=args.elm_model_weights,
        results_save_path=args.results_save_path
    )