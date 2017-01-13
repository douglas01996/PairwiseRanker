import data_reader
import tree_rnn
DIR = 'd:\\MacShare\\data\\'
TRAIN = 'train'
DEV = 'dev'
TEST = 'test'

class train_iterator(object):
    def __init__(self, kbest_filename , gold_filename, vocab,batch):
        with open(kbest_filename, 'r') as reader:
            self.data = reader.readlines()
            self.data.append('PTB_KBEST')
        with open(gold_filename, 'r') as reader:
            self.gdata = reader.readlines()
        self.kbest_id = 0
        self.vocab = vocab
        self.index = 0
        self.gindex = 0
        self.batch = batch
        self.length = len(self.data)
        self.glength = len(self.gdata)

    def read_give_tree(self,tree_index):
        scores = []
        tree = []
        ktrees = []
        kbestlines = []
        # read train
        index = 0
        best_num = 0
        while index < self.length:
            line = self.data[index]
            if best_num == tree_index:
                if line.strip() != 'PTB_KBEST':
                    if line.strip() == '':
                        ktrees.append(read_tree(tree, self.vocab))
                        kbestlines.append(tree[:])
                        tree = []
                    elif not '_' in line:
                        scores.append(float(line))
                    else:
                        tree.append(line)
                else:
                    break
            if line.strip() == 'PTB_KBEST':
                best_num += 1
            index += 1
        # read gold
        list = []
        gold = []
        index = 0
        num = 1
        while index < self.glength:
            line = self.gdata[index]
            if num == tree_index:
                if line.strip() == '':
                    root = read_tree(list, self.vocab)
                    gold.append(root)
                    break
                else:
                    list.append(line)
            if line.strip() == '':
                num += 1
            index += 1

        retval = data_reader.instance(ktrees, scores, gold,gold_lines=list,lines=kbestlines, id=tree_index)
        return retval

    def read_all(self):
        scores = []
        kscores = []
        tree = []
        ktrees = []
        kbest = []
        index = 0
        while index < self.length:
            line = self.data[index]
            if line.strip() != 'PTB_KBEST':
                if line.strip() == '':
                    ktrees.append(read_tree(tree, self.vocab))
                    tree = []
                elif not '_' in line:
                    scores.append(float(line))
                else:
                    tree.append(line)
            else:
                if len(ktrees) > 2:
                    kbest.append(ktrees[:])
                    kscores.append(scores[:])
                    ktrees = []
                    scores = []
            index += 1
        # read gold
        list = []
        gold = []
        gindex =  0
        while gindex < self.glength:
            line = self.gdata[gindex]
            if line.strip() == '':
                root = read_tree(list, self.vocab)
                gold.append(root)
                list = []
            else:
                list.append(line)
            gindex += 1
        train_batch = []
        for a, b, c in zip(kbest, kscores, gold):
            self.kbest_id += 1
            if len(c.children) == 0:
                continue
            train_batch.append(data_reader.instance(a, b, c, id=self.kbest_id))
        return train_batch
    def read_batch(self,read_batch = True):
        if self.index == self.length:
            return None
        scores = []
        kscores = []
        tree = []
        ktrees = []
        kbest = []
        #read train
        while self.index < self.length:
            line = self.data[self.index]
            if line.strip() != 'PTB_KBEST':
                if line.strip() == '':
                    ktrees.append(read_tree(tree,self.vocab))
                    tree = []
                elif not '_' in line:
                    scores.append(float(line))
                else:
                    tree.append(line)
            else:
                if len(ktrees) > 2:
                    kbest.append(ktrees[:])
                    kscores.append(scores[:])
                    if read_batch and len(kbest) == self.batch:
                        self.index += 1
                        break
                    ktrees = []
                    scores = []
            self.index += 1
        #read gold
        list = []
        gold = []
        while self.gindex < self.glength:
            line = self.gdata[self.gindex]
            if line.strip() == '':
                root = read_tree(list,self.vocab)
                gold.append(root)
                if read_batch and len(gold) == self.batch:
                    self.gindex += 1
                    break
                list = []
            else:
                list.append(line)
            self.gindex += 1
        train_batch = []
        for a,b,c in zip(kbest,kscores,gold):
            self.kbest_id += 1
            if len(c.children) == 0:
                continue
            train_batch.append(data_reader.instance(a,b,c,id=self.kbest_id))
        return train_batch
    def reset(self):
        self.index = 0
        self.gindex = 0
        self.kbest_id = 0

def read_tree(list,vocab):
    att_list = []
    nodes = []
    root = None
    for i in range(len(list)):
        att_list.append(list[i].split())
        word = att_list[i][1]
        if vocab is None:
            val = word
        else:
            val = vocab.index(word)
        nodes.append(tree_rnn.Node(val))
    for i in range(len(list)):
        parent = int(att_list[i][6]) - 1
        if parent >= 0:
            nodes[parent].add_child(nodes[i])
        elif parent == -1:
            root = nodes[i]
    return root
