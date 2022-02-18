import os

cur_path = os.path.abspath(os.path.dirname(__file__))
PROJECT_NAME = os.path.split(cur_path)[1]

# train：tr validation：cv test：tt
# TYPE = 'tr'
# train
# TODO
LR = 1e-4
EPOCH = 200
TRAIN_BATCH_SIZE = 2
TRAIN_DATA_PATH = '/data02/maying/data/FM_large_data_2.2/tr/'

# validation
# TODO
VALIDATION_BATCH_SIZE = 1
VALIDATION_DATA_PATH = '/data02/maying/data/FM_large_data_2.2/cv/all/'
VALIDATION_DATA_NUM = 3000

# test
# TODO
TEST_DATA_PATH = '/data02/maying/data/FM_large_data_2.2/cv/all/'
TEST_DATA_NUM = 2000

# model
# TODO
MODEL_STORE = os.path.join('/data02/maying/result/module_store', PROJECT_NAME + '/')
if not os.path.exists(MODEL_STORE):
    os.mkdir(MODEL_STORE)
    print('Create model store file  successful!\n'
          'Path: \"{}\"'.format(MODEL_STORE))
else:
    print('The model store path: {}'.format(MODEL_STORE))

# log
# TODO
LOG_STORE = os.path.join('/data02/maying/result/log_store', PROJECT_NAME + '/')
if not os.path.exists(LOG_STORE):
    os.mkdir(LOG_STORE)
    print('Create log store file  successful!\n'
          'Path: \"{}\"'.format(LOG_STORE))
else:
    print('The log store path: {}'.format(LOG_STORE))

# result
# TODO
RESULT_STORE = os.path.join('/data02/maying/result/result', PROJECT_NAME + '/')
TEST_STORE = os.path.join('/data02/maying/data/FM_data/')
if not os.path.exists(RESULT_STORE):
    os.mkdir(RESULT_STORE)
    print('Create validation result store file  successful!\n'
          'Path: \"{}\"'.format(RESULT_STORE))
else:
    print('The validation result store path: {}'.format(RESULT_STORE))

# other variable
# TODO
IS_LOG = False
FILTER_LENGTH = 512
HOP_LENGTH = 128
EPSILON = 1e-8

CUDA_ID = ['cuda:0']