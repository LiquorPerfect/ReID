# -*- encoding: utf-8 -*-
'''
-----------------------------------------------
* @File    :   evaluate.py
* @Time    :   2021/03/02 22:02:23
* @Author  :   zhang_jinlong
* @Version :   1.0
* @Contact :   zhangjinlongwin@163.com
* @Address ：  https://github.com/LiquorPerfect
-----------------------------------------------
'''

import argparse
import os
import pickle

import numpy as np
import torch


def evaluate(qf, ql, qc, gf, fl, gc):
    query = qf
    # 求矩阵的模
    score = np.dot(gf, query)
    # argsort()是将X中的元素从小到大排序后，提取对应的索引index
    index = np.argsort(score)
    # np.argwhere 返回非0的数组元组的索引，其中a是要索引数组的条件
    query_index = np.argwhere(gl == ql)
    camera_index = np.argwhere(gc == qc)

    # 把在a中的但是不在b中的元素按a中的顺序排序，并且不合并重复的元素
    good_index = np.setdiff1d(query_index, camera_index, assume_unique=True)
    junk_index1 = np.argwhere(gl == "-1")
    junk_index2 = np.intersect1d(query_index, camera_index)
    junk_index = np.append(junk_index2, junk_index1)

    ap_tmp, CMC_tmp = compute_mAP(index, good_index, junk_index)
    return ap_tmp, CMC_tmp


def compute_mAP(index, good_index, junk_index):
    ap = 0
    cmc = torch.IntTensor(len(index)).zero_()
    # if empty
    if good_index.size == 0:
        cmc[0] = -1
        return ap, cmc
    # remove junk_index
    mask = np.in1d(index, junk_index, invert=True)
    index = index[mask]

    # find good_index index
    ngood = len(good_index)
    mask = np.in1d(index, good_index)
    rows_good = np.argwhere(mask == True)
    rows_good = rows_good.flatten()

    cmc[rows_good[0]:] = 1
    for i in range(ngood):
        d_recall = 1.0 / ngood
        precision = (i + 1) * 1.0 / (rows_good[i] + 1)
        if rows_good[i] != 0:
            old_precision = i * 1.0 / rows_good[i]
        else:
            old_precision = 1.0
        ap = ap + d_recall * (old_precision + precision) / 2
    return ap, cmc


def parser_args():
    parser = argparse.ArgumentParser(
        description="The model performance evaluation script")
    parser.add_argument("--result_file",
                        default="./test_result.p",
                        type=str,
                        help="the test save result file")
    return parser.parse_args()


def main():
    args = parser_args()
    with open(args.result_file, 'rb') as f:
        result = pickle.load(f)

    query_feature = result["query_f"]
    query_cam = np.array(result["query_cam"])  #camera
    query_label = np.array(result["query_label"])
    gallery_feature = result["gallery_f"]
    gallery_cam = np.array(result["gallery_cam"])
    gallery_label = np.array(result["gallery_label"])

    CMC = torch.IntTensor(len(gallery_label)).zero()
    ap = 0.0

    for i in range(len(query_label)):
        ap_tmp, CMC_tmp = evaluate(query_feature[i], query_label[i],
                                   query_cam[i], gallery_feature,
                                   gallery_label, gallery_cam)
        if CMC_tmp == -1:
            continue
        CMC = CMC + CMC_tmp
        ap += ap_tmp
        print(i, CMC_tmp[i], ap_tmp)

    CMC = CMC.float()
    CMC = CMC / len(query_label)
    print("Rank@1:%+f Rank@5:%+f Rank@10:%+f mAP:%+f" %
          (CMC[0], CMC[4], CMC[9], ap / len(query_label)))


if __name__ == "main":
    main()
