from datasets import load_dataset
import json
import random
from tqdm import tqdm

def load_dataset_lst():
    slimorca = load_dataset("Open-Orca/SlimOrca")
    metamathqa = load_dataset("meta-math/MetaMathQA")
    magicoder1 = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K")
    magicoder2 = load_dataset("ise-uiuc/Magicoder-Evol-Instruct-110K")
    return slimorca, metamathqa, magicoder1, magicoder2

def make_lst(slimorca, metamathqa, magicoder1, magicoder2):
    data_lst = []
    for i in tqdm(slimorca['train']):
        # slimorca
        data = {'dataset': "", "instruction": "", "conversation": [{"input": "", "output": ""}]}
        data['dataset'] = 'SlimOrca'
        for idx1 in i['conversations']:
            if idx1['from'] == 'system':
                data['instruction'] = idx1['value']
            if idx1['from'] == 'human':
                data['conversation'][0]['input'] = idx1['value']
            if idx1['from'] == 'gpt':
                data['conversation'][0]['output'] = idx1['value']
        data_lst.append(data)

    for j in tqdm(metamathqa['train']):
        # metamathqa
        data = {'dataset': "", "instruction": "", "conversation": [{"input": "", "output": ""}]}
        data['dataset'] = 'MetaMathQA'
        data['conversation'][0]['input'] = j['query']
        data['conversation'][0]['output'] = j['response']
        data_lst.append(data)

    for k in tqdm(magicoder1['train']):
        # magicoder_75k
        data = {'dataset': "", "instruction": "", "conversation": [{"input": "", "output": ""}]}
        data['dataset'] = 'Magicoder'
        data['conversation'][0]['input'] = k['problem']
        data['conversation'][0]['output'] = k['solution']
        data_lst.append(data)

    for l in tqdm(magicoder2['train']):
        # magicoder_110k
        data = {'dataset': "", "instruction": "", "conversation": [{"input": "", "output": ""}]}
        data['dataset'] = 'Magicoder'
        data['conversation'][0]['input'] = l['instruction']
        data['conversation'][0]['output'] = l['response']
        data_lst.append(data)

    return data_lst

# save data
def write_jsonl(data, fname):
    with open('{}.jsonl'.format(fname), 'w', encoding='utf-8') as idx:
        for i in data_lst:
            idx.write(json.dumps(i) + "\n")

if __name__=="__main__":
    s, me, ma, ma2 = load_dataset_lst()
    data_lst = make_lst(s, me, ma, ma2)
    random.shuffle(data_lst)
    write_jsonl(data_lst, "it_data")
    print(len(data_lst))
