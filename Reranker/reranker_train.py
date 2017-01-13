import data_util
import dependency_model
import parser_test
import os
import numpy as np
import data_reader
DIR = 'd:\\MacShare\\data\\'
TRAIN = 'train'
DEV = 'dev'
TEST = 'test'
OUTPUT_MODEL = 'model.pkl'
OUTPUT_BEST = 'model_best.pkl'
OUTPUT_DICT = 'dict.pkl'
TRAIN_BATCH_SIZE = 3
FINE_GRAINED = False
DEPENDENCY = False
SEED = 88

NUM_EPOCHS = 100


def train_dataset(model, data, echo):
    losses = []
    avg_loss = 0.0
    total_data = len(data)
    loss = 0
    for i, inst in enumerate(data):
        loss = model.train_step(inst.kbest,inst.gold)  # labels will be determined by model
        losses.append(loss)
        print 'instance: %s  loss: %s' %(i,loss)
        #avg_loss = avg_loss * (len(losses) - 1) / len(losses) + loss / len(losses)
        #print 'echo %d batch %d avg loss %.4f example id %d batch size %d\r' % (echo ,batch,avg_loss, inst.id, total_data)
    loss = np.mean(losses)
    print 'loss %.4f' % loss
    return loss

def train_model():
    data_tool = data_reader.data_manager(TRAIN_BATCH_SIZE,os.path.join(DIR,TRAIN+'.kbest'),
                         os.path.join(DIR,TRAIN+'.gold'),
                         os.path.join(DIR, DEV + '.kbest'),
                         os.path.join(DIR, DEV + '.gold'),vocab_path= os.path.join(DIR, OUTPUT_DICT))
    train_iter = data_tool.train_iter
    dev_data = data_tool.dev_data
    data = train_iter.read_all()
    print 'build model'
    model = dependency_model.get_model(data_tool.vocab.size(), data_tool.max_degree)
    print 'model established'
    max_uas = 0
    parser_test.evaluate_dataset(model, dev_data, False)
    for i in range(NUM_EPOCHS):
        print 'Echo %d train , data size: %d' % (i, len(data))
        train_dataset(model, data ,i)
        uas = parser_test.evaluate_dataset(model, dev_data , False)[0]
        if uas > max_uas:
            max_uas = uas
            data_util.save_model(model, os.path.join(DIR,OUTPUT_BEST))
        data_util.save_model(model, os.path.join(DIR, OUTPUT_MODEL))
    print 'best score %.4f' % max_uas

if __name__ == '__main__':
    train_model()
