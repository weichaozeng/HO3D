import pickle

meta_file = '/home/zvc/Data/HO3D_v2/evaluation/AP10/meta/0000.pkl'
with open(meta_file, 'rb') as f:
    meta_data = pickle.load(f)
print(meta_data.keys())