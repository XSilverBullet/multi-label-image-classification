# -*- coding: utf-8 -*-
"""
Created on Mon May 14 14:20:55 2018
@author: Wei Sun
"""

import numpy as np


def get_AP(test_label, prediction_score):
    AP = []
    for i in range(test_label.shape[1]):

        PRECISION = []
        RECALL = []

        ground_truth = int(np.sum(test_label[:, i]))
        #print(ground_truth)

        # p_label = np.sort(prediction_score[:, i])[-1::-1]
        index = np.argsort(prediction_score[:, i])[-1::-1]
        #print(index)

        for top_number in range(1, test_label.shape[0]+1, 1):
            total_true = 0
            for count in range(len(index)):
                if count < top_number:
                    total_true += test_label[index[count],i]
            tp = total_true
            #print(tp)
            # fp = top_number - tp
            # fn = ground_truth - tp
            # tn = test_label.shape[0] - tp - fp - fn

            p = float(tp) / top_number
            r = float(tp) / ground_truth

            PRECISION.append(p)
            RECALL.append(r)

        sum = 0
        PRECISION = np.array(PRECISION)
        RECALL = np.array(RECALL)
        #print(PRECISION)
        #print(RECALL)
        for j in range(1, ground_truth + 1, 1):
            a = PRECISION[np.where((RECALL >= (float(j) / ground_truth)))]
            maximum = np.max(a)
            #print(maximum)
            sum += maximum
        AP.append(sum / ground_truth)

    return AP


# p_label = [0.23, 0.76, 0.01, 0.91, 0.13, 0.45, 0.12, 0.03, 0.38, 0.11,
#            0.03, 0.09, 0.65, 0.07, 0.12, 0.24, 0.1, 0.23, 0.46, 0.08]
# t_label = [0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0,
#            0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
#
# p_label = np.array(p_label).reshape((20,1))
# t_label = np.array(t_label).reshape((20,1))
# print(get_AP(t_label, p_label))
