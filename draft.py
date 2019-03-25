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

t_mat = {
    "a":{
        "a": 0.3,
        "b": 0.5,
        "c": 0.2
    },
    "b":{
        "a": 0.8,
        "b": 0.1,
        "c": 0.1
    },
    "c":{
        "a": 0.2,
        "b": 0.5,
        "c": 0.3
    }
}
e_mat = {
    "a":{
        "o1": 0.2,
        "o2": 0.1,
        "o3": 0.5,
        "o4": 0.2
    },
    "b":{
        "o1": 0.4,
        "o2": 0.2,
        "o3": 0.1,
        "o4": 0.3
    },
    "c":{
        "o1": 0.6,
        "o2": 0.1,
        "o3": 0.1,
        "o4": 0.2
    }
}

start_pos = {"a": 0.3, "b": 0.4, "c": 0.3}

def get_argmax(a):
    return max(a, key=a.get)

def ls2s2o(ls, s, o):
    tp = t_mat.get(ls).get(s)
    ep = e_mat.get(s).get(o)
    return tp * ep

def viterbi(ob_seq):
    """
    
    """
    most_psb_path = []
    most_psb_path_prob = []
    s_start = get_argmax(start_pos)
    most_psb_path.append(s_start)

    def get_partial_max(ob, last_stat):
        """
        return partial max point, return max_prob    
        """
        tmp = {}
        # print("lll", last_stat)
        for s in t_mat:
            stat_prob = ls2s2o(last_stat, s, ob)
            tmp[s]= stat_prob
            
        mst_s = get_argmax(tmp)
        most_psb_path.append(mst_s)
        return 

    for ob in ob_seq:
        # print("111", most_psb_path)
        get_partial_max(ob, most_psb_path[-1])

    # print(most_psb_path)
    return most_psb_path

if __name__ == '__main__':
    test_ob_seq = ["o1", "o1", "o1"]
    mst_stat = viterbi(test_ob_seq)
    print(mst_stat)
