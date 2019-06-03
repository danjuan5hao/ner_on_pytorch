# -*- coding: utf-8 -*-
import random 
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F 


def argmax(vec):
    # return the argmax as a python int
    _, idx = torch.max(vec, 1)
    return idx.item()

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
    max_score = vec[0, argmax(vec)]
    max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
    return max_score + \
        torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class CrfLayer(nn.Module):
    def __init__(self, tag_to_ix, feats):
        super(CrfLayer, self).__init__()
        self._tag_to_ix = tag_to_ix
        self._tagset_size = len(tag_to_ix)
        self._transition_metrix = nn.Parameter(
            torch.randn(self._tagset_size, self._tagset_size))
        self.feats = feats

        # 调整__transition_metrix中相应的start和stop的概率，让他不可能是最大的
        self._transition_metrix.data[tag_to_ix["START_TAG"], :] = -10000  # 到“start_tag”的概率最小
        self._transition_metrix.data[:, tag_to_ix["STOP_TAG"]] = -10000  # 从“stort_tag”出发的概率最小
    
    # def _find_best_next_tag_idx(self, tag_ix, tag_softmax_score):
    #     trans = self._transition_metrix[:, tag_ix]  #从tag—ix出发的概率
    #     rst = torch.mul(trans, tag_softmax_score)
    #     return torch.max(rst, 0)
    def _score_sentence(self, feats, tags):
        # Gives the score of a provided tag sequence
        score = torch.zeros(1)
        tags = torch.cat([torch.tensor([self._tag_to_ix["START_TAG"]], dtype=torch.long), tags])
        for i, feat in enumerate(feats):
            score = score + \
                self._transition_metrix[tags[i + 1], tags[i]] + feat[tags[i + 1]]
        score = score + self._transition_metrix[self._tag_to_ix["STOP_TAG"], tags[-1]]
        return score

    def neg_log_likelihood(self, feats, tags):
        forward_score = self._forward_alg(feats)
        gold_score = self._score_sentence(feats, tags)
        return forward_score - gold_score

    def _viterbi_decode(self, feats):
            backpointers = []

            # Initialize the viterbi variables in log space
            init_vvars = torch.full((1, self._tagset_size), -10000.)
            init_vvars[0][self._tag_to_ix["START_TAG"]] = 0

            # forward_var at step i holds the viterbi variables for step i-1
            forward_var = init_vvars
            for feat in feats:
                bptrs_t = []  # holds the backpointers for this step
                viterbivars_t = []  # holds the viterbi variables for this step

                for next_tag in range(self._tagset_size):
                    # next_tag_var[i] holds the viterbi variable for tag i at the
                    # previous step, plus the score of transitioning
                    # from tag i to next_tag.
                    # We don't include the emission scores here because the max
                    # does not depend on them (we add them in below)
                    next_tag_var = forward_var + self._transition_metrix[next_tag]
                    best_tag_id = argmax(next_tag_var)
                    bptrs_t.append(best_tag_id)
                    viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
                # Now add in the emission scores, and assign forward_var to the set
                # of viterbi variables we just computed
                forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
                backpointers.append(bptrs_t)

            # Transition to STOP_TAG
            terminal_var = forward_var + self._transition_metrix[self._tag_to_ix["STOP_TAG"]]
            best_tag_id = argmax(terminal_var)
            path_score = terminal_var[0][best_tag_id]

            # Follow the back pointers to decode the best path.
            best_path = [best_tag_id]
            for bptrs_t in reversed(backpointers):
                best_tag_id = bptrs_t[best_tag_id]
                best_path.append(best_tag_id)
            # Pop off the start tag (we dont want to return that to the caller)
            start = best_path.pop()
            assert start == self._tag_to_ix["START_TAG"]  # Sanity check
            best_path.reverse()
            return path_score, best_path

    def forward(self, feats): 
        """inpter_tensor的size需要是（seq_len, tagset_size）
        """
        return  self._viterbi_decode(feats)
    


if __name__ == "__main__":
    tag_to_ix = {"B": 0, "I": 1, "O": 2, "E": 3, "S": 4, "START_TAG": 5, "STOP_TAG": 6}
   

    
    # print(test_crf_layer._transition_metrix.data)

    seq_len = random.choice([*range(1, 15)])
    batch_size = 12
    tagset_size = 7
    # test_input_tensor = torch.randn(seq_len, batch_size, tagset_size)
    # test_w_seq_tensor = torch.randn(seq_len, tagset_size)
    test_feat_seq = torch.randn(seq_len, tagset_size)

    test_crf_layer = CrfLayer(tag_to_ix, test_feat_seq)
    a,b = test_crf_layer._viterbi_decode(test_feat_seq)
    print(a, b)


