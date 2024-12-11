import sys
import math
import numpy as np
import logging
import random
from gomoku.GomokuUtils import *

log = logging.getLogger(__name__)
EPS = 1e-8
# sys.setrecursionlimit(3000)

def softmax(x):
    return 1 / (1 + math.exp(-x))

class MCTS():
    def __init__(self, game, nnet, args):
        self.game = game
        self.nnet = nnet
        self.args = args
        self.Qsa = {}  # i.e. Q(s, a)
        self.Nsa = {}  # visit-times of (s, a)
        self.Ns = {}  # visit-times of s
        self.Ps = {}  # initial policy (returned by neural net)

        self.Es = {}  # stores game.get_result ended for board s
        self.Vs = {}  # stores game.board_valid_moves for board s
    
    def getActionProb(self, canonicalBoard, temperature=1):
        for i in range(self.args.numMCTSSims):
            self.search(canonicalBoard)

        s = self.game.board_tostring(canonicalBoard)
        counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.game.get_action_space_size())]

        if temperature == 0:
            bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
            bestA = np.random.choice(bestAs)
            probs = [0] * len(counts)
            probs[bestA] = 1
            return probs

        counts = [x ** (1. / temperature) for x in counts]
        counts_sum = float(sum(counts))
        probs = [x / counts_sum for x in counts]
        return probs
    
    def search(self, canonicalBoard):
        s = self.game.board_tostring(canonicalBoard)

        if s not in self.Es:
            self.Es[s] = self.game.get_result(canonicalBoard)
        if self.Es[s] != 0:
            # terminal node
            return -self.Es[s]

        if s not in self.Ps:
            # leaf node, get the policy
            if self.nnet:
                self.Ps[s], v = self.nnet.predict(canonicalBoard)
            else:
                # Just use uniform policy
                # valids = self.game.board_valid_moves(board=canonicalBoard, player=1)
                # self.Ps[s] = valids / np.sum(valids)
                # v = 0  # Can be set to 0 if no network prediction is available

                # greedy policy
                valids = self.game.board_valid_moves(board=canonicalBoard, player=1)
                self.Ps[s] = np.zeros_like(valids, dtype=np.float64)
                for a in range(self.game.get_action_space_size()):
                    if valids[a]:
                        next_s, next_player = self.game.get_next_state(board=canonicalBoard, player=1, action=a)
                        # TODO
                        self.Ps[s][a] = get_gomoku_board_score(next_s, a)
                
                max_score = np.max(self.Ps[s])
                exp_scores = np.exp(self.Ps[s] - max_score)
                sum_exp_scores = np.sum(exp_scores)

                if sum_exp_scores > 0:
                    self.Ps[s] = exp_scores / sum_exp_scores
                else:
                    log.error("All valid moves were masked, doing a workaround.")
                    self.Ps[s] = valids / np.sum(valids)
                v = 0
                        
            valids = self.game.board_valid_moves(board=canonicalBoard, player=1)
            self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
            sum_Ps_s = np.sum(self.Ps[s])
            if sum_Ps_s > 0:
                self.Ps[s] /= sum_Ps_s
            else:
                # if all valid moves were masked make all valid moves equally probable
                log.error("All valid moves were masked, doing a workaround.")
                self.Ps[s] = self.Ps[s] + valids
                self.Ps[s] /= np.sum(self.Ps[s])

            self.Vs[s] = valids
            self.Ns[s] = 0
            return -v

        valids = self.Vs[s]
        cur_best = -float('inf')
        best_acts = []

        for a in range(self.game.get_action_space_size()):
            if valids[a]:
                if (s, a) in self.Qsa:
                    u = self.Qsa[(s, a)] + self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
                else:
                    u = self.args.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)

                if u > cur_best:
                    cur_best = u
                    best_acts = [a]
                elif u == cur_best:
                    best_acts.append(a)

        # Tie-break should be good for **pure-MCTS**
        a = random.choice(best_acts)
        next_s, next_player = self.game.get_next_state(board=canonicalBoard, player=1, action=a)
        next_s = self.game.get_canonical_form(next_s, next_player)
        v = self.search(next_s)

        if (s, a) in self.Qsa:
            self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
            self.Nsa[(s, a)] += 1
        else:
            self.Qsa[(s, a)] = v
            self.Nsa[(s, a)] = 1
        self.Ns[s] += 1
        return -v
