import data_util
import parser_test
import os
import train_iterator
import numpy as np
import data_reader
import tree_rnn
DIR = 'd:\\MacShare\\data\\'
DIR2 = '/home/wangq/parser/data2/'
TRAIN = 'train'
DEV = 'dev'
TEST = 'test'
OUTPUT_MODEL = 'model.pkl'
OUTPUT_DICT = 'dict.pkl'
TRAIN_BATCH_SIZE = 3
FINE_GRAINED = False
DEPENDENCY = False
SEED = 88

NUM_EPOCHS = 10

def check_input(x, tree):
    print len(x)
    print len(tree)
    print tree[:, -1]
    assert np.array_equal(tree[:, -1], np.arange(len(x) - len(tree), len(x)))


def check_tree(root_node):
    x, tree = tree_rnn.gen_nn_inputs(root_node, max_degree=12, only_leaves_have_vals=False)
    # x list the val of leaves and internal nodes
    child_exists = tree[0] > -1
    offset = 5 * 1 - child_exists * 1

    check_input(x, tree)


if __name__ == '__main__':
    print 'load vocab'
    max_degree, vocab = data_util.load_dict(os.path.join(DIR, OUTPUT_DICT))
    print 'vocab size:' + str(vocab.size())
    print 'max_degree' + str(max_degree)
    print 'create train batch'
    train_iter = train_iterator.train_iterator(os.path.join(DIR,TRAIN+'.kbest'),
                         os.path.join(DIR,TRAIN+'.gold'), vocab,TRAIN_BATCH_SIZE)
    print 'get 3237'
    inst = train_iter.read_give_tree(1)
    i = 0
    for root in inst.kbest:
        print i
        check_tree(root)
        print inst.gold_lines[0]
        print inst.lines[0]
        i += 1