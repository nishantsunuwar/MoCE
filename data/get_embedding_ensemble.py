import argparse
import json
import numpy as np

from InstructorEmbedding import INSTRUCTOR
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

def open_json(fpath):
    file = []
    with open(fpath, 'r') as json_file:
        json_list = list(json_file)

    for idx, json_str in enumerate(json_list):
        file.append(json.loads(json_str))

    return file

class Encoding:
    def __init__(self, model_name):
        self.model_name = model_name
        self.model = self.set_model(model_name)

    def get_embedding(self, data):
        if self.model_name == "instructor":
            sentences = [[dic['instruction'], dic['conversation'][0]['input']] for dic in tqdm(data)]
            inst_array = self.model.encode(sentences, batch_size=256, show_progress_bar=True)

            return inst_array

        if self.model_name == "e5":
            sentences = [str(dic['instruction'] + dic['conversation'][0]['input']) for dic in tqdm(data)]
            inst_array = self.model.encode(sentences, normalize_embeddings=True)

            return inst_array

    def set_model(self, model_name):
        # 1. instructor embedding
        if model_name == 'instructor':
            return INSTRUCTOR('hkunlp/instructor-xl')

        # 2. intfloat/e5-large-v2
        elif model_name == 'e5':
            return SentenceTransformer('intfloat/e5-large-v2')

if __name__=="__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default=True, help='instructor,e5')
    arg = argparser.parse_args()

    data = open_json("it_data.jsonl")
    encoder = Encoding(arg.model)

    encoded = encoder.get_embedding(data)
    np.save(f'embedding_{arg.model}.npy', encoded)
