import pandas as pd
import numpy as np
import argparse
import scipy.stats
import itertools
import os, re
import pickle
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
sns.set_style("whitegrid")

parser = argparse.ArgumentParser()
parser.add_argument("data", type=str)
parser.add_argument("--out_dir", type=str, default="figures")
args = parser.parse_args()

data = pickle.load(open(args.data, "rb"))
if not os.path.isdir(args.out_dir):
    os.makedirs(args.out_dir)

# palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
#            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
fig, ax = plt.subplots()
xvals = []
for xval in data["n_neurons"]:
    xvals.append(np.log(xval))
ax.plot(xvals, data["odds_ratio"], marker='.', color='#d62728')

# dotted line representing total effect
plt.hlines(data["odds_ratio"][-1], 0, xvals[-1],
    color='black', alpha=0.5, linestyle='dotted')
plt.xlabel("Log(Number of Neurons)")
plt.ylabel("Indirect Effect")
ax.set_xlim([xvals[0]-0.1, xvals[-1]+0.1])

fig_name = os.path.basename(args.data).split(".pickle")[0] + ".pdf"
plt.savefig(f"{args.out_dir}/{fig_name}", bbox_inches='tight', format='pdf')
