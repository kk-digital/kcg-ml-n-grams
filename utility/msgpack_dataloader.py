import msgpack
import os
import tqdm

def write_msgpack(obj, file_path):
    with open(file_path, 'wb') as f:
        msgpack.dump(obj, f)

def read_msg_pack(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    decoded_data = msgpack.unpackb(data)

    return decoded_data

def load_data(data_path):
    embedding_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('.msgpack'):
                embedding_files.append(os.path.join(root, file))

    dataset = []
    print('[*] Reading msgpack files')
    for path in tqdm.tqdm(embedding_files[:100]):
        data = read_msg_pack(path)
        dataset.append(data)

    return dataset