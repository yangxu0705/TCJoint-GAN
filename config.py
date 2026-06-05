import tensorflow as tf
import numpy as np

LATENT_DEPTH = 100

BATCH_SIZE = 8
NUM_EPOCHS = 300

SEED = np.random.normal(0, 0.33, size=[16, LATENT_DEPTH]).astype(np.float32)

MODEL_SAVE_DIR = "./learned-models/3dGAN-tf2"
MODEL_NAME = "3dGAN"

OBJ = 'joint'
HYPARAMS = {
    "joint": {
        "project_shape": [4, 4, 4, 256],
        "gen_filters_list": [256, 128, 64, 32, 1],
        "disc_filters_list": [32, 64, 128, 256]
    }
}
DATA_PATH = './data/train/joint' #  ./是当前目录  ../是父级目录  /是根目录

#tensorflow.python.framework.errors_impl.ResourceExhaustedError: OOM when allocating tensor with shape[64,147,147,64]

#出现以上类似的错误，主要是因为模型中的batch_size值设置过大，导致内存溢出，batch_size是每次送入模型中的值，由于GPU的关系，一般设为16,32,64,128。


