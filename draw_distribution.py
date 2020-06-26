import os
import numpy as np
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm
import random


#training = 'cat_minhao'
#root = 'checkpoints/{:s}/WideResNet34-10_seed_233/eps'.format(training)

root = 'from_minhao/eps'



to_load = sorted(glob.glob(os.path.join(root, '*.npy')), key=lambda x: int(x.split('/')[-1].split('.')[0].split('_')[-1]))
save_path = root + '_plot.png'

random.seed(0)
sample_num = 2
sample_idx = random.sample(range(50000), sample_num)
epochs = len(to_load)

eps_vs_epoch = np.zeros((sample_num, epochs))

for ii, path in enumerate(tqdm(to_load)):
    eps_vs_epoch[:, ii] = np.load(path)[sample_idx]


fig, ax = plt.subplots()
for ii in range(sample_num):
    ax.plot(range(epochs), eps_vs_epoch[ii, :])

plt.savefig(save_path)
plt.close()
