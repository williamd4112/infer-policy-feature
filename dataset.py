import numpy as np
import argparse, logging

import cPickle as pickle
from tqdm import *

DEFAULT_CODE_DICT = {
        'sand': 0,
        'grass': 1,
        'lapis_block': 2,
        'Agent_1': 3,
        'Agent_2': 4,
        'Pig': 5
}

def _str2one_hot_code(s, code_dict, delimiter='/'):
    code = np.zeros([len(code_dict)])
    for s_ in s.split(delimiter):
        code[code_dict[s_]] = 1
    return code

def _few_hot_code2val(code):
    val = 0
    code = code[::-1]
    for i, v in enumerate(code):
        val += v * (2**i)
    return val
    
class ObservationTransformer(object):
    def __init__(self):
        self.obs_code_dict = DEFAULT_CODE_DICT
        self.n_obs_code = len(self.obs_code_dict)    

    def transform(self, obs):
        obs_trans = np.zeros(obs.shape, dtype=np.float32)
        for b in range(obs.shape[0]):
            for i in range(obs.shape[1]):
                for j in range(obs.shape[2]):
                    code = _str2one_hot_code(obs[b, i, j], DEFAULT_CODE_DICT)
                    obs_trans[b, i, j] = _few_hot_code2val(code)
        return obs_trans

if __name__ == '__main__':
    import glob
    logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', help='dir', type=str, required=True)
    args = parser.parse_args()

    trans = ObservationTransformer()
    
    X = []
    X_len = []
    T = []
    i = 0
    for ep_dir in tqdm(glob.glob('%s/*' % args.dir)):
        obs_file = np.load('%s/obs.npy' % ep_dir)
        obs = trans.transform(obs_file)
        with open('%s/agent_type' % ep_dir) as f:
            agent_type = f.readline()[:-1]
            agent_type = np.array([agent_type])
        obs_len = np.array([obs.shape[0]], dtype=np.int32)
        npad = ((0, 26 - obs.shape[0]), (0, 0), (0, 0))
        obs = np.pad(obs, pad_width=npad, mode='constant', constant_values=0)
        X.append(obs[np.newaxis, :])
        X_len.append(obs_len[np.newaxis, :])
        T.append(agent_type[np.newaxis, :])

        i += 1
        if i > 100:
            break
    X = np.concatenate(X)
    X_len = np.concatenate(X_len)
    T = np.concatenate(T)
    
    np.save('X_small.npy', X)
    np.save('X_len_small.npy', X_len)
    np.save('T_small.npy', T)
 
