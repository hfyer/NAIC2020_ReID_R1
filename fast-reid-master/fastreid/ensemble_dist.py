import numpy as np
import os
import json

query_path='../logs/NAIC_All/B/1_101x_rcs/query_paths.npy'
gallery_path = '../logs/NAIC_All/B/1_101x_rcs/gallery_paths.npy'
query = np.load(query_path)
gallery = np.load(gallery_path)
print(len(query), len(gallery))



dist1_path = '../logs/NAIC_All/B/1_101x_rcs/dist.npy'
dist2_path = '../logs/NAIC_All/B/2_200x_rcs/dist.npy'
dist3_path = '../logs/NAIC_All/B/0_269x/dist.npy'
dist4_path = '../logs/NAIC_All/B/0_269x_augmix/dist.npy'
dist5_path = '../logs/NAIC_All/B/0_269x_rcs_augmix/dist.npy'

# dist6_path = '../logs/NAIC_All/B/1_101x/dist.npy'
# dist7_path = '../logs/NAIC_All/B/2_200x/dist.npy'



distmat = np.load(dist1_path)
distmat += np.load(dist2_path)
distmat += np.load(dist3_path)
distmat += np.load(dist4_path)
distmat += np.load(dist5_path)

# distmat += np.load(dist6_path)
# distmat += np.load(dist7_path)

print(distmat.shape)
indexes = np.argsort(distmat, axis=1)

res = {}
m,n = indexes.shape
for i in range(m):
    tmp=[]
    for j in indexes[i][:200]:
        tmp.append(gallery[j][-12:])
        res[query[i][-12:]]=tmp


save_path = 'R1_B.json'
print("Writing to {}".format(save_path))
json.dump(res, open(save_path, 'w'))