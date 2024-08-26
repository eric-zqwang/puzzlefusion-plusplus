def build_encoder(arch, feat_dim, global_feat=True, **kwargs):
    """
    helper function for creating an encoder
    :param arch: str with format 'xxx.xxx.x'
    :param feat_dim: the feature dim output by the encoder
    :param global_feat: whether output a global feature for one piece or output per-point features
    :param kwargs: misc args for transformer, kpconv ...
    :return:
    """
    archs = arch.lower().split('.')
    if 'in_feat_dim' in kwargs:
        in_feat_dim = kwargs['in_feat_dim']
    else:
        in_feat_dim = 3
    if 'pointnet2_pt' in archs:
        from .pointnet2_pointwise import PointNet2PTMSG, PointNet2PTMSGDynamic
        assert not global_feat
        if 'msg' in archs:
            if 'dynamic' in archs:
                if isinstance(feat_dim, list):
                    model = PointNet2PTMSGDynamic(feat_dim[0], feat_dim[1])
                else:
                    model = PointNet2PTMSGDynamic(in_feat_dim, feat_dim)
            else:
                if isinstance(feat_dim, list):
                    model = PointNet2PTMSG(feat_dim[0], feat_dim[1])
                else:
                    model = PointNet2PTMSG(in_feat_dim, feat_dim)
        else:
            raise NotImplementedError(f'{arch} not supported')
    elif 'dgcnn' in archs:
        from .dgcnn import DGCNN, DGCNNDynamic
        if 'dynamic' in archs:
            model = DGCNNDynamic(feat_dim, global_feat, in_feat_dim)
        else:
            model = DGCNN(feat_dim, global_feat)
    else:
        raise NotImplementedError(f'{arch} is not supported')
    return model
