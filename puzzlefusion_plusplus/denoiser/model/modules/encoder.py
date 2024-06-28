import torch.nn as nn
import numpy as np
from puzzlefusion_plusplus.vqvae.model.modules.pn2 import PN2
from puzzlefusion_plusplus.vqvae.model.modules.quantizer import VectorQuantizer


class VQVAE(nn.Module):
    def __init__(self, cfg):
        super(VQVAE, self).__init__()
        self.pn2 = PN2(cfg)
        self.cfg = cfg
        self.encoder = self.pn2.encode
        self.vector_quantization = VectorQuantizer(
            cfg.ae.n_embeddings,
            cfg.ae.embedding_dim,
            cfg.ae.beta
        )
    

    def encode(self, part_pcs):
        """
        x.shape = (batch, C, L)
        """

        x = part_pcs.permute(0, 2, 1)
        z_e, xyz = self.encoder(x)
        
        B, L, C = z_e.shape
        _, z_q, _, _, _ = self.vector_quantization(
            z_e.reshape(B, 4 * L, -1)
        )
        z_q = z_q.reshape(B, L, -1)
        output_dict = {
            "z_q": z_q,
            "xyz": xyz
        }

        return output_dict
    