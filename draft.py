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
    "health":{
        "health": 0.7,
        "fever": 0.3,
    },
    "fever":{
        "health": 0.4,
        "fever": 0.6,
    } 
}
e_mat = {
    "health":{
        "normal": 0.5,
        "cold": 0.4,
        "dizzy": 0.1,
    },
    "fever":{
        "normal": 0.1,
        "cold": 0.3,
        "dizzy": 0.6,
    }
}

start_pos = {"health": 0.6, "fever": 0.4}

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
    # most_psb_path.append(s_start)

    def get_partial_max(ob):
        """
        return partial max point, return max_prob    
        """
        try:
            last_stat = most_psb_path[-1]
        except IndexError:
            last_stat = s_start
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
        get_partial_max(ob)

    # print(most_psb_path)
    return most_psb_path

if __name__ == '__main__':
    test_ob_seq = ["normal", "cold", "dizzy"]
    mst_stat = viterbi(test_ob_seq)
    print(mst_stat)
