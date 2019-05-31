# -*- coding: utf-8 -*-
import numpy as np
def _find_argmax(score_seq):
    return np.argmax(score_seq)

def _find_max(score_seq):
    return np.max(score_seq)

def _score1_add_score2(part1, part2):
    part1 = np.array(part1)
    part2 = np.array(part2)
    return part1 + part2

class Viterbi:
    def __init__(self, t_mat, e_mat, init_state):
        self._t_mat = t_mat
        self._e_mat = e_mat
        self._init_state = init_state
        self._tagset_size = len(t_mat)

    def _partial_best(self, forword_score, best_prev_tag_score_part2):
        best_prev_tags = []
        best_prev_tags_score_part1 = []
        for next_tag in range(self._tagset_size):  # 对于每个tag
            next_tag_score = _score1_add_score2(forword_score, self._t_mat[next_tag])  # 上一步的各个tag的累计分数到该tag的分数
            best_prev_tag = _find_argmax(next_tag_score)
            print("best_prev_tag", best_prev_tag)
            best_prev_tag_score_part1 = _find_max(next_tag_score)
            # print("next_tag_score", next_tag_score)
            # exit()
            best_prev_tags.append(best_prev_tag)
            best_prev_tags_score_part1.append(best_prev_tag_score_part1)
        return best_prev_tags, _score1_add_score2(best_prev_tag_score_part1, best_prev_tag_score_part2)
    
    def _backtracking(self, tags_seq, score):
        last_tag = _find_argmax(score)
        reversed_path = [last_tag] 

        for prev_tags in reversed(tags_seq):
            last_tag = prev_tags[last_tag]
            reversed_path.append(last_tag)
        return np.array([*reversed(reversed_path)])

    def _decode(self, state_emit_seq):  # state_emit_seq.size [seq_len[tagset_size], ] 长度为 seq_len, 每个元素是一个[tagset_size]
        best_prev_tags_seq = []
        score = self._init_state
        for state_emit in state_emit_seq:
            best_prev_tags, score = self._partial_best(score, state_emit)
            best_prev_tags_seq.append(best_prev_tags)
            # print(best_prev_tags)
            # print(score)
        print(len(best_prev_tags_seq))
        path = self._backtracking(best_prev_tags_seq, score)
        return path[1:]
    
    def decode(self, state_seq):
        return self._decode([self._e_mat[i] for i in state_seq])

if __name__ == "__main__":
    test_t_mat = [[0.1,0.2,0.3,0.3,0.1],
                  [0.4,0.2,0.1,0.1,0.2],
                  [0.2,0.2,0.2,0.1,0.3],
                  [0.3,0.1,0.2,0.2,0.2],
                  [0.5,0.2,0.1,0.1,0.1]]  # 五个标签
    test_e_mat = [[0.15,0.25,0.3,0.25,0.05],
                  [0.35,0.15,0.05,0.2,0.25],
                  [0.2,0.15,0.25,0.2,0.2],
                  [0.1,0.2,0.3,0.15, 0.25],
                  [0.3,0.1,0.05,0.2,0.35],
                  [0.1, 0.1, 0.2, 0.2, 0.4],
                  [0.35,0.15,0.2,0.2,0.1],
                  [0.03,0.17,0.35,0.2,0.25]]  # 七个字
    test_init_state = [0.2, 0.3, 0.1, 0.2, 0.1]

    test_tag_emit_seq = []

    test_viterbi = Viterbi(test_t_mat, test_e_mat, test_init_state)

    # test_partial_best = test_viterbi._partial_best(test_init_state, [0.1,0.2,0.3,0.3,0.1])
    rst = test_viterbi.decode([3,1,0,0,6,4,2])
    print(rst)

    # label2label_transition_mat = np.array(
    #                        [[, -10000.0],  # B 每一行代表其他tag转移到该tag的概率
    #                         [, -10000.0],  # I
    #                         [, -10000.0],  # O
    #                         [, -10000.0],  # E
    #                         [, -10000.0],  # S
    #                         [-10000.0, -10000.0, -10000.0, -10000.0, -10000.0, -10000.0, -10000.0],  # START_TAG
    #                         [, -10000.0]])  # STOP_TAG
    
    # word2label_emit_mat = np.array(
    #                       [[],  # w0 的label是[labelset_size]的概率 
    #                        [],  # w1
    #                        [],  # w2
    #                        [],  # w3
    #                        [],  # w4
    #                        [],  # w5
    #                        [],  # w6
    #                        []])  # w7
    
    # init_state = [-10000.0, -10000.0, -10000.0, -10000.0, -10000.0, 0.0, -10000.0]
    
