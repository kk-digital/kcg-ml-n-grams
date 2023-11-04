import pandas as pd
import numpy as np

import argparse

from transformers import CLIPTokenizer

tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-large-patch14')

def filter_civitai(df):
    # Remove rows with unicode characters
    df = df[df['phrase str'].apply(lambda x: isinstance(x, str) and all(ord(char) < 128 for char in x))]

    # Remove rows with tab
    df = df[df['phrase str'].str.contains('\t') == False]

    # Remove rows with more than 100 characters
    df = df[df['phrase str'].str.len() <= 100]

    return df

def add_features(df):
    """Adds probability to dataframe and log probability based on positive counts

    Args:
        df (pd.DataFrame): civitai csv Dataframe
    """
    df['probability'] = df['positive count'] / df['positive count'].sum()
    df['log probability'] = np.log(df['probability'] + 1e-15)

    def get_phrase_length(row):
        token_encoding = tokenizer(row['phrase str'], return_length=True, return_tensors='pt')
        return token_encoding['length'].item()
    
    df['token_length'] = df.apply(get_phrase_length, axis=1)

    return df

def main(df):
    df = filter_civitai(df)
    df = add_features(df)

    return df

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data_csv_path', type=str, help='Path to civitai csv file')
    ap.add_argument('--output_path', type=str, help='Path to save processed csv')
    args = ap.parse_args()

    df = pd.read_csv(args.data_csv_path)
    df = main(df)
    df.to_csv(args.output_path, index=False)

