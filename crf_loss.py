# -*- coding: utf-8 -*-
import numpy as np
import torch  

def _log_sum_exp(vec):
    return np.log(np.sum(np.exp(vec)))

# def argmax(vec):
#     # return the argmax as a python int
#     _, idx = torch.max(vec, 1)
#     return idx.item()
    
# def log_sum_exp(vec):
#     max_score = vec[0, argmax(vec)]
#     max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
#     return max_score + \
#         torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))

class CrfLayer:
    def __init__(self, e_mat, t_mat):
        self._e_mat = e_mat 
        self._t_mat = t_mat 
        self.tagset_size = len(t_mat) 

    def _init_forward_var(self):
        forward_var = np.array([0]*self.tagset_size)
        return forward_var
    
    def forward_alg(self, w_seq):
        forward_var = self._init_forward_var()
        for w in w_seq:
            feat = self._e_mat[w]
            alphas_t = []
            
            for next_tag in range(self.tagset_size):
                emit_score = np.array([feat[next_tag]]*self.tagset_size)
                trans_score = self._t_mat[next_tag]
                next_tag_var = _log_sum_exp(emit_score + trans_score + forward_var)
                alphas_t.append(next_tag_var)

            forward_var = np.array(alphas_t) + forward_var
        return forward_var

    def seq_score(self, w_seq, tag_seq):
        t_score = 0
        e_score = self._e_mat[w_seq[0]][tag_seq[0]]
        score = t_score + e_score

        last_tag = tag_seq[0]
        for w, t in zip(w_seq[1:], tag_seq[1:]):
            t_score = self._t_mat[t, last_tag]    
            e_score = self._e_mat[w][t]
            score += t_score + e_score
            last_tag = t

        return score
    
# def _forward_alg(feats):
#         # # Do the forward algorithm to compute the partition function
#         # init_alphas = torch.full((1, self.tagset_size), -10000.)
#         # # START_TAG has all of the score.
#         # init_alphas[0][self.tag_to_ix[START_TAG]] = 0.

#         # # Wrap in a variable so that we will get automatic backprop
#         # forward_var = init_alphas

#         # Iterate through the sentence
#         for feat in feats:
#             alphas_t = []  # The forward tensors at this timestep
#             for next_tag in range(self.tagset_size()):
#                 # broadcast the emission score: it is the same regardless of
#                 # the previous tag
#                 emit_score = feat[next_tag].view(
#                     1, -1).expand(1, self.tagset_size)
#                 # the ith entry of trans_score is the score of transitioning to
#                 # next_tag from i
#                 trans_score = self.transitions[next_tag].view(1, -1)
#                 # The ith entry of next_tag_var is the value for the
#                 # edge (i -> next_tag) before we do log-sum-exp
#                 next_tag_var = forward_var + trans_score + emit_score
#                 # The forward variable for this tag is log-sum-exp of all the
#                 # scores.
#                 alphas_t.append(log_sum_exp(next_tag_var).view(1))
#             forward_var = torch.cat(alphas_t).view(1, -1)
#         terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
#         alpha = log_sum_exp(terminal_var)
#         return alpha

if __name__ == "__main__":
    tst_r = 1
    rst = np.array([tst_r]*5)
    
    rst =  _log_sum_exp(rst)
    print(rst)