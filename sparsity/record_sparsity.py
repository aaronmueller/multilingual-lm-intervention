import pickle
import os
import numpy as np
from argparse import ArgumentParser

def get_percentage(file):

    model = file.split('_')[-3:]
    df = pickle.load(open(file,'rb'))
    num_neurons = df['n_neurons'][-1]
    # get percentage of neurons to get max IE
    max_neuron = np.argmax(df['odds_ratio'])
    max_num = df['n_neurons'][max_neuron]
    max_ratio = max_num / num_neurons
    # get percentage of neurons to get total IE
    total_neuron = df['odds_ratio'][-1]
    total_num = np.array(df['n_neurons'])[df['odds_ratio'] >= total_neuron][0]
    total_ratio = total_num / num_neurons
    return (model[0], model[1], model[2][:-7], max_ratio, total_ratio)

if __name__=='__main__':
    parser = ArgumentParser()
    parser.add_argument("--data_dir", type=str, default = "sparsity")
    parser.add_argument("--out_dir", type=str, default="sparsity")
    parser.add_argument("--model", type=str)
    args = parser.parse_args()

    files = list(filter(lambda x: 'topk' in x and args.model in x, os.listdir(args.data_dir)))
    record = [('model', 'intervention_type','language', 'max_ie_ratio', 'total_ie_ratio')]
    for f in files:
        f = os.path.join(args.data_dir, f)
        record.append(get_percentage(f))
    pickle.dump(record, open(args.out_dir +'/'+ args.model + ".pickle", "wb"))
