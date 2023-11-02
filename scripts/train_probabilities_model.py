import csv
import argparse
import math
import os
import msgpack
import sys
import clip
import torch

base_directory = os.getcwd()
sys.path.insert(0, base_directory)
from models.probabilities_linear_model import ProbabilitiesModel

def count_data_points(csv_reader, value):
    return sum(int(row[value]) for row in csv_reader)

def load_dataset(input_path):
    # Initialize an empty list to store the parsed data.
    data = []
    total_sum=0
    pos_sum=0
    neg_sum=0

    # Open the CSV file and read its content.
    with open(input_path, mode='r', encoding='utf-8',newline='') as file:
        reader = csv.DictReader(file)
        count=0
        # Iterate through each row in the CSV file.
        for row in reader:
                index = int(row['index'])
                total_count = int(row['total count'])
                positive_count = int(row['positive count'])
                negative_count = int(row['negative count'])
                token_size = int(row['token size'])
                phrase_str = row['phrase str']
                
                total_sum+=total_count
                pos_sum+=positive_count
                neg_sum+=negative_count
                # Create a dictionary for each row and append it to the data list.
                data.append({
                    'index': index,
                    'total_prob': total_count,
                    'pos_prob': positive_count,
                    'neg_prob': negative_count,
                    'token_size': token_size,
                    'phrase_str': phrase_str
                })

                count+=1
        

    
    for entry in data:
        entry['total_prob']=math.log(entry['total_prob'] / total_sum) if entry['total_prob'] > 0 else 0.0 
        entry['pos_prob']=math.log(entry['pos_prob'] / pos_sum) if entry['pos_prob'] > 0 else 0.0 
        entry['neg_prob']=math.log(entry['neg_prob'] / neg_sum) if entry['total_prob'] > 0 else 0.0 
    
    return data

def get_clip_embeddings(input_path):
    data = load_dataset(input_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)

    # Create a list to store the data entries
    data_entries = []
    count=1
    for phrase in data:
        print(f"{count} files/{len(data)} files----------")
        try:
            text = clip.tokenize([phrase['phrase_str']]).to(device)
            with torch.no_grad():
                phrase['embedding'] = model.encode_text(text)

            # Create a dictionary for the data entry, including all fields
            data_entry = {
                'index': phrase['index'],
                'total_prob': phrase['total_prob'],
                'pos_prob': phrase['pos_prob'],
                'neg_prob': phrase['neg_prob'],
                'token_size': phrase['token_size'],
                'phrase_str': phrase['phrase_str'],
                'embedding': phrase['embedding'].tolist(),  # Convert embedding to a list
            }

            # Add the data entry to the list
            data_entries.append(data_entry)
        except:
            print('error')

        count+=1

    # Specify the path for the output msgpack file
    if not os.path.exists('embeddings'):
        os.makedirs('embeddings')
        
    msgpack_file_path = os.path.join('embeddings', 'civitai_embeddings.msgpack')

    # Open the msgpack file for writing in binary mode
    with open(msgpack_file_path, 'wb') as f:
        # Serialize and save the list of data entries to the msgpack file
        msgpack.dump(data_entries, f)

    return data_entries

def parse_args():
    parser = argparse.ArgumentParser()

    #parameters
    parser.add_argument("--input", type=str, default="datasets",
                        help="The path to dataset directory")

    return parser.parse_args()

def main():
    # Specify the path to the msgpack file
    msgpack_file_path = 'embeddings/civitai_embeddings.msgpack'

    # Open the msgpack file for reading in binary mode
    with open(msgpack_file_path, 'rb') as f:
        # Load the data entries from the msgpack file
        dataset = msgpack.load(f)

    #Extract the embeddings from the data entries
    embeddings = [entry['embedding'] for entry in dataset]
    output = [entry['total_prob'] for entry in dataset]

    model=ProbabilitiesModel(input_size=512)
    model.train(embeddings,output, epochs=100, training_batch_size=100)


if __name__ == '__main__':
    main()