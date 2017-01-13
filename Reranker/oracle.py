import dev_reader
import os
import parser_test
DIR = 'd:\\MacShare\\32best\\'
TRAIN = 'train'
DEV = 'dev32'
TEST = 'test32'

if __name__ == '__main__':
    test_data = dev_reader.read_dev(os.path.join(DIR, DEV + '.kbest'),
                                    os.path.join(DIR, DEV + '.gold'), None)
    parser_test.evaluate_oracle_worst(test_data)
    test_data = dev_reader.read_dev(os.path.join(DIR,TEST+'.kbest'),
                         os.path.join(DIR,TEST+'.gold'),None)
    parser_test.evaluate_oracle_worst(test_data)