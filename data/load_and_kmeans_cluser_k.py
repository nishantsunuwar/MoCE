import argparse
import json
import pandas as pd
import numpy as np
import faiss
from tqdm import tqdm

def open_json(fpath):
    file = []
    with open(fpath, 'r') as json_file:
        json_list = list(json_file)

    for idx, json_str in enumerate(json_list):
        file.append(json.loads(json_str))

    return file

def write_jsonl(data, add, fname):
    with open('{}.jsonl'.format(fname), 'w', encoding='utf-8') as idx:
        for i, cl in tqdm(zip(data, add)):
            i['cluster'] = cl
            idx.write(json.dumps(i) + "\n")

    return

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--model', type=str, default=True, help='instructor,e5 + num_cluster')
    argparser.add_argument('--k', type=int, default=True, help='n_clusters')
    arg = argparser.parse_args()

    data = open_json("it_data.jsonl")
    emb = np.load(f'embedding_{arg.model}.npy')
    print(emb.shape)

    kmeans = faiss.Kmeans(emb.shape[1], arg.k, niter=25, nredo=3, verbose=True, spherical=True)
    kmeans.train(emb.astype(np.float32))
    _, I = kmeans.index.search(emb.astype(np.float32), 1)
    labels = I.reshape(-1)
    new = [[row1, row2] for row1, row2 in zip(data, labels)]
    returned_df = pd.DataFrame(new, columns=['data', 'cluster'])

    print('Save kmeans model..')
    load_path = f"kmeans_model_cluster_{arg.k}_{arg.model}.pkl"
    np.save(f"centroids_{arg.model}_{arg.k}.npy", kmeans.centroids)
    np.save(f"labels_{arg.model}_{arg.k}.npy", labels)

    print('Done! Saved kmeans model')

    print('Done, Write...')
    write_jsonl(data, returned_df['cluster'], f'elbow_data/it_data_add_cluster_{arg.k}_{arg.model}')
    print('DONE!')