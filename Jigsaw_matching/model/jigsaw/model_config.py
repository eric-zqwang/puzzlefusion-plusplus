from easydict import EasyDict as edict

__C = edict()

model_cfg = __C

__C.JIGSAW = edict()
__C.JIGSAW.ROT_TYPE = 'rmat'
__C.JIGSAW.PC_FEAT_DIM = 128
__C.JIGSAW.AFF_FEAT_DIM = 512
__C.JIGSAW.AFFINITY = 'aff_dual'
__C.JIGSAW.ENCODER = 'pointnet2_pt.msg.dynamic'
__C.JIGSAW.TEST_S_MASK = True
__C.JIGSAW.PC_CLS_METHOD = 'binary'  # ['binary', 'multi']
__C.JIGSAW.PC_NUM_CLS = 2

__C.JIGSAW.SINKHORN_MAXITER = 20
__C.JIGSAW.SINKHORN_TAU = 0.05

__C.JIGSAW.TF_NUM_HEADS = 8
__C.JIGSAW.TF_NUM_SAMPLE = 16

__C.JIGSAW.LOSS = edict()
__C.JIGSAW.LOSS.w_cls_loss = 1.0
__C.JIGSAW.LOSS.w_mat_loss = 0.0
__C.JIGSAW.LOSS.mat_epoch = 9
__C.JIGSAW.LOSS.w_rig_loss = 0.0
__C.JIGSAW.LOSS.rig_epoch = 199


def get_model_cfg():
    return model_cfg.JIGSAW
