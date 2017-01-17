import pickle
import theano
from theano import tensor as T
import numpy as np
import data_util
import tree_lstm
import tree_rnn
import data_reader
FINE_GRAINED = False
DEPENDENCY = False
SEED = 88

LEARNING_RATE = 0.1

EMB_DIM = 50
HIDDEN_DIM = 200
OUTPUT_DIM = 3



class DependencyModel(tree_lstm.ChildSumTreeLSTM):
    def set_parmas(self,input_file):
        pkl_file = open(input_file, 'rb')
        self.embeddings.set_value(pickle.load(pkl_file))
        self.W_i.set_value(pickle.load(pkl_file))
        self.U_i.set_value(pickle.load(pkl_file))
        self.b_i.set_value(pickle.load(pkl_file))
        self.W_f.set_value(pickle.load(pkl_file))
        self.U_f.set_value(pickle.load(pkl_file))
        self.b_f.set_value(pickle.load(pkl_file))
        self.W_o.set_value(pickle.load(pkl_file))
        self.U_o.set_value(pickle.load(pkl_file))
        self.b_o.set_value(pickle.load(pkl_file))
        self.W_u.set_value(pickle.load(pkl_file))
        self.U_u.set_value(pickle.load(pkl_file))
        self.b_u.set_value(pickle.load(pkl_file))
        self.W_out.set_value(pickle.load(pkl_file))
        self.b_out.set_value(pickle.load(pkl_file))
        pkl_file.close()


    def create_output_fn(self):
        self.W_out = theano.shared(self.init_matrix([self.hidden_dim]))
        self.b_out = theano.shared(self.init_vector([1]))
        self.params.extend([self.W_out, self.b_out])
        def fn(final_state):
            score = T.dot(self.W_out, final_state) + self.b_out,
            return T.sum(score)
        return fn

    def train_step2(self,inst):
        lens = len(inst.kbest)
        losses = 0
        max = 0
        for j in range(1,lens):
            if inst.f1score[max] > inst.f1score[j]:
                gold = inst.kbest[max]
                pred = inst.kbest[j]
            else:
                gold = inst.kbest[j]
                pred = inst.kbest[max]
            loss = np.mean(self.train_margin(gold,pred))
            if loss > 0:
                losses += loss
        return losses
    def train_step(self, kbest_tree, gold_root):
        scores = []
        for tree in kbest_tree:
            if tree.size == gold_root.size:
                scores.append(self.predict(tree))
            else:
                scores.append(-1000)
        max_id = scores.index(max(scores))
        pred_root = kbest_tree[max_id]
        if pred_root.size != gold_root.size:
            return 0
        gold_score = self.predict(gold_root)
        pred_score = scores[max_id]
        loss = gold_score - pred_score
        if loss < 0:
            self.train_margin(gold_root, pred_root)
        return loss

    def loss_fn(self, gold_y, pred_y):
        # loss = T.sum(pred_y-gold_y)
        # regular = 0
        # L2 = T.sum(self.W_o ** 2)+T.sum(self.W_i ** 2)
        # L3 = (T.sum(self.W_o ** 2)+T.sum(self.W_i ** 2))
        # for param in [self.W_o,self.W_i,self.W_f,self.W_u,self.W_out]:
        #     regular += T.sum(param ** 2)
        return T.sum(pred_y-gold_y)

def get_model(num_emb, max_degree):
    return DependencyModel(
        num_emb, EMB_DIM, HIDDEN_DIM, OUTPUT_DIM,
        degree=max_degree, learning_rate=LEARNING_RATE,
        trainable_embeddings=True,
        labels_on_nonroot_nodes=False,
        irregular_tree=True)

