class Vocab(object):
    def __init__(self,filename):
        self.words = []
        self.word2idx = {}
        self.unk_index = -1
        self.unk_token = 'unk'
        with open(filename) as reader:
            data = reader.readlines()
            for line in data:
                if line.strip() != 'PTB_KBEST' and '_' in line:
                    word = line.split()[1]
                    if not word in self.words:
                        self.words.append(word)
                        self.word2idx[word]=len(self.words)-1
        reader.close()


    def index(self,word):
        return self.word2idx.get(word, self.unk_index)
    def size(self):
        return len(self.words)


