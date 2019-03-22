# -*- coding:utf-8 -*-
import numpy as np
# t_mat
#    a    b    c
# a  0.3, 0.5, 0.2
# b  0.8, 0.1, 0.1
# c  0.2, 0.5, 0.3

# e_mat
#     o1   o2   o3   o4
# a   0.2, 0.1, 0.5, 0.2
# b   0.4, 0.2, 0.1, 0.3
# c   0.6, 0.1, 0.1, 0.2

t_mat = np.array([[0.3, 0.5, 0.2],
                  [0.8, 0.1, 0.1],
                  [0.2, 0.5, 0.3]])
e_mat = np.array([[0.2, 0.1, 0.5, 0.2],
                  [0.4, 0.2, 0.1, 0.3],
                  [0.6, 0.1, 0.1, 0.2]])

start_pos = [0.5, 0.2, 0.1]

def viterbi(ob_seq):
    """

    """
    most_psb_path = []

    return most_psb_path

if __name__ == '__main__':
    pass