# -*- coding: utf-8 -*-
import random 
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F 

import util

class CrfLayer(nn.Module):
    def __init__(self, tag_to_ix):
        super(CrfLayer, self).__init__()
        self._tag_to_ix = tag_to_ix
        self._tagset_size = len(tag_to_ix)
        self._transition_metrix = nn.Parameter(
            F.softmax(torch.randn(self._tagset_size, self._tagset_size), dim=1))

        # 调整__transition_metrix中相应的start和stop的概率，让他不可能是最大的
        self._transition_metrix.data[tag_to_ix["START_TAG"], :] = -10000  # 到“start_tag”的概率最小
        self._transition_metrix.data[:, tag_to_ix["STOP_TAG"]] = -10000  # 从“stort_tag”出发的概率最小
    
    def _find_best_next_tag_idx(self, tag_ix, tag_softmax_score):
        trans = self._transition_metrix[:, tag_ix]  #从tag—ix出发的概率
        rst = torch.mul(trans, tag_softmax_score)
        return torch.max(rst, 0)

    def _viterbi_decode(self, input_tensor):
        score = 0
        next_tag_ix = self._tag_to_ix["START_TAG"]
        idx_seq = [next_tag_ix]
        for i in input_tensor:
            s, next_tag_ix = self._find_best_next_tag_idx(next_tag_ix, i)
            s = s.item()
            # score += np.log(s)
            idx_seq.append(next_tag_ix.item())
        # self._find_best_next_tag_idx()
        return score, idx_seq

    def forward(self, input_tensor): 
        """inpter_tensor的size需要是（seq_len, tagset_size）
        """
        return  self._viterbi_decode(input_tensor)
    


def CrfLoss()



if __name__ == "__main__":
    tag_to_ix = {"B": 0, "I": 1, "O": 2, "E": 3, "S": 4, "START_TAG": 5, "STOP_TAG": 6}

    test_crf_layer = CrfLayer(tag_to_ix)
    # print(test_crf_layer._transition_metrix.data)

    seq_len = random.choice([*range(1, 15)])
    batch_size = 12
    tagset_size = 7
    # test_input_tensor = torch.randn(seq_len, batch_size, tagset_size)
    test_input_tensor = torch.randn(seq_len, tagset_size)
    test_input_tensor = F.softmax(test_input_tensor, dim=1)

    a,b = test_crf_layer._viterbi_decode(test_input_tensor)
    print(a, b)


