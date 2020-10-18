# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import copy
import logging
from collections import OrderedDict
from sklearn import metrics
import faiss
import json
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from .evaluator import DatasetEvaluator
from .query_expansion import aqe
from .rank import evaluate_rank
# from .rerank import re_ranking
from .roc import evaluate_roc
from fastreid.utils import comm

logger = logging.getLogger(__name__)


class ReidEvaluator(DatasetEvaluator):
    def __init__(self, cfg, num_query, output_dir=None):
        self.cfg = cfg
        print(cfg)
        self._num_query = num_query
        self._output_dir = cfg.OUTPUT_DIR

        self.features = []
        self.pids = []
        self.camids = []
        self.img_paths=[]
    def reset(self):
        self.features = []
        self.pids = []
        self.camids = []
        self.img_paths = []
    def process(self, inputs, outputs):
        self.pids.extend(inputs["targets"])
        self.camids.extend(inputs["camids"])
        self.img_paths.extend(inputs["img_paths"])
        self.features.append(outputs.cpu())

    @staticmethod
    def cal_dist(metric: str, query_feat: torch.tensor, gallery_feat: torch.tensor):
        assert metric in ["cosine", "euclidean"], "must choose from [cosine, euclidean], but got {}".format(metric)
        if metric == "cosine":
            dist = 1 - torch.mm(query_feat, gallery_feat.t())
        else:
            m, n = query_feat.size(0), gallery_feat.size(0)
            xx = torch.pow(query_feat, 2).sum(1, keepdim=True).expand(m, n)
            yy = torch.pow(gallery_feat, 2).sum(1, keepdim=True).expand(n, m).t()
            dist = xx + yy
            dist.addmm_(query_feat, gallery_feat.t(), beta=1, alpha=-2)
            dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
        return dist.cpu().numpy()

    def evaluate(self):
        if comm.get_world_size() > 1:
            comm.synchronize()
            features = comm.gather(self.features)
            features = sum(features, [])

            pids = comm.gather(self.pids)
            pids = sum(pids, [])

            camids = comm.gather(self.camids)
            camids = sum(camids, [])

            img_paths = comm.gather(self.img_paths)
            img_paths = sum(img_paths, [])

            if not comm.is_main_process():
                return {}
        else:
            features = self.features
            pids = self.pids
            camids = self.camids
            img_paths = self.img_paths

        features = torch.cat(features, dim=0)
        # query feature, person ids and camera ids
        query_features = features[:self._num_query]
        query_pids = np.asarray(pids[:self._num_query])
        query_camids = np.asarray(camids[:self._num_query])

        # gallery features, person ids and camera ids
        gallery_features = features[self._num_query:]
        gallery_pids = np.asarray(pids[self._num_query:])
        gallery_camids = np.asarray(camids[self._num_query:])

        self._results = OrderedDict()

        if self.cfg.TEST.AQE.ENABLED:
            logger.info("Test with AQE setting")
            qe_time = self.cfg.TEST.AQE.QE_TIME
            qe_k = self.cfg.TEST.AQE.QE_K
            alpha = self.cfg.TEST.AQE.ALPHA
            query_features, gallery_features = aqe(query_features, gallery_features, qe_time, qe_k, alpha)

        if self.cfg.TEST.METRIC == "cosine":
            query_features = F.normalize(query_features, dim=1)
            gallery_features = F.normalize(gallery_features, dim=1)

        dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, gallery_features)
        
        if self.cfg.TEST.RERANK.ENABLED:
            logger.info("Test with rerank setting")
            k1 = self.cfg.TEST.RERANK.K1
            k2 = self.cfg.TEST.RERANK.K2
            lambda_value = self.cfg.TEST.RERANK.LAMBDA
            # q_q_dist = self.cal_dist(self.cfg.TEST.METRIC, query_features, query_features)
            # g_g_dist = self.cal_dist(self.cfg.TEST.METRIC, gallery_features, gallery_features)
            # re_dist = re_ranking(dist, q_q_dist, g_g_dist, k1, k2, lambda_value)
            # Luo rerank
            re_dist = re_ranking(query_features, gallery_features, k1, k2, lambda_value)
            print('re_dist',re_dist.shape)
            query_features = query_features.numpy()
            gallery_features = gallery_features.numpy()
            # gen json
            eval_json(img_paths,self._num_query,re_dist,query_features,gallery_features,self._output_dir,use_distmat=True)
            return 
            #
            cmc, all_AP, all_INP = evaluate_rank(re_dist, query_features, gallery_features,
                                                 query_pids, gallery_pids, query_camids,
                                                 gallery_camids, use_distmat=True)
        else:
            query_features = query_features.numpy()
            gallery_features = gallery_features.numpy()
            # gen json
            eval_json(img_paths,self._num_query,dist,query_features,gallery_features,self._output_dir,use_distmat=False)
            return
            # 
            cmc, all_AP, all_INP = evaluate_rank(dist, query_features, gallery_features,
                                                 query_pids, gallery_pids, query_camids, gallery_camids,
                                                 use_distmat=False)
        mAP = np.mean(all_AP)
        mINP = np.mean(all_INP)
        for r in [1, 5, 10]:
            self._results['Rank-{}'.format(r)] = cmc[r - 1]
        self._results['mAP'] = mAP
        self._results['mINP'] = mINP

        if self.cfg.TEST.ROC_ENABLED:
            scores, labels = evaluate_roc(dist, query_features, gallery_features,
                                          query_pids, gallery_pids, query_camids, gallery_camids)
            fprs, tprs, thres = metrics.roc_curve(labels, scores)

            for fpr in [1e-4, 1e-3, 1e-2]:
                ind = np.argmin(np.abs(fprs - fpr))
                self._results["TPR@FPR={:.0e}".format(fpr)] = tprs[ind]

        return copy.deepcopy(self._results)


def eval_json(img_paths,num_query,dist,query_features,gallery_features,output_dir,use_distmat):
    query_paths = img_paths[:num_query]
    gallery_paths = img_paths[num_query:]

    num_q, num_g = dist.shape
    #ensemble dist
    np.save(output_dir+'/dist.npy', dist)
    print("save dist")
    np.save(output_dir+'/query_paths.npy', query_paths)
    print("save query_paths")
    np.save(output_dir+'/gallery_paths.npy', gallery_paths)
    print("save gallery_paths")
    dim = query_features.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(gallery_features)
    if(use_distmat):
        indices = np.argsort(dist, axis=1)
    else:
        _, indices = index.search(query_features, k=num_g)
    m,n = indices.shape
    res={}
    for i in range(m):
        tmp=[]
        for j in indices[i][:200]:
            tmp.append(gallery_paths[j][-12:])
        res[query_paths[i][-12:]]=tmp
    if use_distmat:
        print(output_dir)
        with open(output_dir+'/res_rr.json','w') as f:
            json.dump(res, f, indent=4, separators=(',', ': '))
        print("writed")
    else:
        print(output_dir)
        with open(output_dir+'/res.json','w') as f:
            json.dump(res, f, indent=4, separators=(',', ': '))
        print("writed")



#Luo rerank
import gc 
import time
def euclidean_distance(qf, gf):

    m = qf.shape[0]
    n = gf.shape[0]

    # dist_mat = torch.pow(qf,2).sum(dim=1, keepdim=True).expand(m,n) +\
    #     torch.pow(gf,2).sum(dim=1, keepdim=True).expand(n,m).t()
    # dist_mat.addmm_(1,-2,qf,gf.t())

    # for L2-norm feature
    dist_mat = 2 - 2 * torch.matmul(qf, gf.t())
    return dist_mat


def batch_euclidean_distance(qf, gf, N=6000):
    m = qf.shape[0]
    n = gf.shape[0]

    dist_mat = []
    for j in range(n // N + 1):
        temp_gf = gf[j * N:j * N + N]
        temp_qd = []
        for i in range(m // N + 1):
            temp_qf = qf[i * N:i * N + N]
            temp_d = euclidean_distance(temp_qf, temp_gf)
            temp_qd.append(temp_d)
        temp_qd = torch.cat(temp_qd, dim=0)
        temp_qd = temp_qd / (torch.max(temp_qd, dim=0)[0])
        dist_mat.append(temp_qd.t().cpu())
    del temp_qd
    del temp_gf
    del temp_qf
    del temp_d
    torch.cuda.empty_cache()  # empty GPU memory
    dist_mat = torch.cat(dist_mat, dim=0)
    return dist_mat


# 将topK排序放到GPU里运算，并且只返回k1+1个结果
# Compute TopK in GPU and return (k1+1) results
def batch_torch_topk(qf, gf, k1, N=6000):
    m = qf.shape[0]
    n = gf.shape[0]

    dist_mat = []
    initial_rank = []
    for j in range(n // N + 1):
        temp_gf = gf[j * N:j * N + N]
        temp_qd = []
        for i in range(m // N + 1):
            temp_qf = qf[i * N:i * N + N]
            temp_d = euclidean_distance(temp_qf, temp_gf)
            temp_qd.append(temp_d)
        temp_qd = torch.cat(temp_qd, dim=0)
        temp_qd = temp_qd / (torch.max(temp_qd, dim=0)[0])
        temp_qd = temp_qd.t()
        initial_rank.append(torch.topk(temp_qd, k=k1, dim=1, largest=False, sorted=True)[1])

    del temp_qd
    del temp_gf
    del temp_qf
    del temp_d
    torch.cuda.empty_cache()  # empty GPU memory
    initial_rank = torch.cat(initial_rank, dim=0).cpu().numpy()
    return initial_rank


def batch_v(feat, R, all_num):
    V = np.zeros((all_num, all_num), dtype=np.float32)
    m = feat.shape[0]
    for i in tqdm(range(m)):
        temp_gf = feat[i].unsqueeze(0)
        # temp_qd = []
        temp_qd = euclidean_distance(temp_gf, feat)
        temp_qd = temp_qd / (torch.max(temp_qd))
        temp_qd = temp_qd.squeeze()
        temp_qd = temp_qd[R[i]]
        weight = torch.exp(-temp_qd)
        weight = (weight / torch.sum(weight)).cpu().numpy()
        V[i, R[i]] = weight.astype(np.float32)
    return V


def k_reciprocal_neigh(initial_rank, i, k1):
    forward_k_neigh_index = initial_rank[i, :k1 + 1]
    backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
    fi = np.where(backward_k_neigh_index == i)[0]
    return forward_k_neigh_index[fi]


def re_ranking(probFea, galFea, k1, k2, lambda_value):
    # The following naming, e.g. gallery_num, is different from outer scope.
    # Don't care about it.

    t1 = time.time()
    query_num = probFea.size(0)
    all_num = query_num + galFea.size(0)
    feat = torch.cat([probFea, galFea]).cuda()
    initial_rank = batch_torch_topk(feat, feat, k1 + 1, N=6000)
    # del feat
    del probFea
    del galFea
    torch.cuda.empty_cache()  # empty GPU memory
    gc.collect()  # empty memory
    print('Using totally {:.2f}s to compute initial_rank'.format(time.time() - t1))
    print('starting re_ranking')

    R = []
    for i in tqdm(range(all_num)):
        # k-reciprocal neighbors
        k_reciprocal_index = k_reciprocal_neigh(initial_rank, i, k1)
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_k_reciprocal_index = k_reciprocal_neigh(initial_rank, candidate, int(np.around(k1 / 2)))
            if len(np.intersect1d(candidate_k_reciprocal_index, k_reciprocal_index)) > 2. / 3 * len(
                    candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(k_reciprocal_expansion_index, candidate_k_reciprocal_index)
        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        R.append(k_reciprocal_expansion_index)

    gc.collect()  # empty memory
    print('Using totally {:.2f}S to compute R'.format(time.time() - t1))
    V = batch_v(feat, R, all_num)
    del R
    gc.collect()  # empty memory
    print('Using totally {:.2f}S to compute V-1'.format(time.time() - t1))
    initial_rank = initial_rank[:, :k2]

    ### 下面这个版本速度更快
    ### Faster version
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank

    ### 下面这个版本更省内存(约40%)，但是更慢
    ### Low-memory version
    '''gc.collect()  # empty memory
    N = 2000
    for j in range(all_num // N + 1):
        if k2 != 1:
            V_qe = np.zeros_like(V[:, j * N:j * N + N], dtype=np.float32)
            for i in range(all_num):
                V_qe[i, :] = np.mean(V[initial_rank[i], j * N:j * N + N], axis=0)
            V[:, j * N:j * N + N] = V_qe
            del V_qe
    del initial_rank'''

    gc.collect()  # empty memory
    print('Using totally {:.2f}S to compute V-2'.format(time.time() - t1))
    invIndex = []

    for i in range(all_num):
        invIndex.append(np.where(V[:, i] != 0)[0])
    print('Using totally {:.2f}S to compute invIndex'.format(time.time() - t1))

    jaccard_dist = np.zeros((query_num, all_num), dtype=np.float32)
    for i in tqdm(range(query_num)):
        temp_min = np.zeros(shape=[1, all_num], dtype=np.float32)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(V[i, indNonZero[j]],
                                                                               V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2. - temp_min)
    del V
    gc.collect()  # empty memory
    original_dist = batch_euclidean_distance(feat, feat[:query_num, :]).numpy()
    final_dist = jaccard_dist * (1 - lambda_value) + original_dist * lambda_value
    # print(jaccard_dist)
    del original_dist

    del jaccard_dist

    final_dist = final_dist[:query_num, query_num:]
    print(final_dist)
    print('Using totally {:.2f}S to compute final_distance'.format(time.time() - t1))
    return final_dist