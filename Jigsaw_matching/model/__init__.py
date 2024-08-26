from .modules import *
# This import must be at the top. DO NOT change the order.


def build_model(cfg):
    module_list = cfg.MODULE.lower().split('.')
    if module_list[0] == 'jigsaw':
        from .jigsaw import JointSegmentationAlignmentModel
        return JointSegmentationAlignmentModel(cfg)
    else:
        raise NotImplementedError(f'Model {cfg.MODULE.lower()} not implemented')
