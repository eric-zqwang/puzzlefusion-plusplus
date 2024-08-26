import torch
import torch.nn.functional as F

from .pairwise_alignment import pairwise_alignment
from .utils import lexico_iter


def _valid_mean(loss_per_part, valids):
    """Average loss values according to the valid parts.

    Args:
        loss_per_part: [B, P]
        valids: [B, P], 1 for input parts, 0 for padded parts

    Returns:
        [B], loss per data in the batch, averaged over valid parts
    """
    if valids is not None:
        valids = valids.float().detach()
        loss_per_data = (loss_per_part * valids).sum(1) / valids.sum(1)
    else:
        loss_per_data = loss_per_part.sum(1) / loss_per_part.shape[1]
    return loss_per_data


def permutation_loss(pred_mat, gt_mat, src_ns, tgt_ns):
    """
    Permutation loss
    $$L_mat = -\frac{1}{N} {\sum_{1\leq i j \leq N} x_{ij}^{gt} \log \hat{x}_{ij}^{gt} + (1-x_{ij}^{gt}) \log (1-\hat{x}_{ij}^{gt})}$$
    @param pred_mat: [B, N_src, N_tgt]
    @param gt_mat: [B, N_src, N_tgt]
    @param src_ns: [B], the number of points of the source in each batch
    @param tgt_ns: [B], the number of points of the target in each batch
    @return: L_mat
    """
    batch_num = pred_mat.shape[0]

    pred_dsmat = pred_mat.to(dtype=torch.float32)

    try:
        assert torch.all((pred_dsmat >= 0) * (pred_dsmat <= 1))
        assert torch.all((gt_mat >= 0) * (gt_mat <= 1))
    except AssertionError as err:
        print(pred_dsmat)
        raise err

    loss = torch.tensor(0.0).to(pred_dsmat.device)
    n_sum = torch.zeros_like(loss)
    for b in range(batch_num):
        batch_slice = [b, slice(src_ns[b]), slice(tgt_ns[b])]
        loss += F.binary_cross_entropy(
            pred_dsmat[batch_slice], gt_mat[batch_slice], reduction="sum"
        )
        n_sum += src_ns[b].to(n_sum.dtype).to(pred_dsmat.device)

    return loss / n_sum


def rigid_loss(
        n_critical_pcs,
        match_mat,
        gt_pcs,
        critical_pcs_idx,
        part_pcs,
        n_valid,
        n_pcs,
):
    loss = torch.tensor(0.0).to(match_mat.device)
    B, N, _ = gt_pcs.shape
    n_critical_pcs_cumsum = torch.cumsum(n_critical_pcs, dim=-1)
    n_pcs_cumsum = torch.cumsum(n_pcs, dim=-1)
    n_sum = torch.zeros_like(loss)
    match_mat_d = match_mat.detach().cpu().numpy()
    for b in range(B):
        sum_full_matched = torch.sum(match_mat[b])
        for idx1, idx2 in lexico_iter(torch.arange(n_valid[b])):
            cri_st1 = 0 if idx1 == 0 else n_critical_pcs_cumsum[b, idx1 - 1]
            cri_ed1 = n_critical_pcs_cumsum[b, idx1]
            cri_st2 = 0 if idx2 == 0 else n_critical_pcs_cumsum[b, idx2 - 1]
            cri_ed2 = n_critical_pcs_cumsum[b, idx2]
            pc_st1 = 0 if idx1 == 0 else n_pcs_cumsum[b, idx1 - 1]
            pc_ed1 = n_pcs_cumsum[b, idx1]
            pc_st2 = 0 if idx2 == 0 else n_pcs_cumsum[b, idx2 - 1]
            pc_ed2 = n_pcs_cumsum[b, idx2]
            n1 = n_critical_pcs[b, idx1]
            n2 = n_critical_pcs[b, idx2]
            if n1 == 0 or n2 == 0:
                continue
            mat = match_mat[b, cri_st1:cri_ed1, cri_st2:cri_ed2]  # [N1, N2]
            mat_s = torch.sum(mat)
            mat2 = match_mat[b, cri_st2:cri_ed2, cri_st1:cri_ed1]
            mat_s2 = torch.sum(mat2)
            mat = mat + mat2.transpose(1, 0)
            mat_d = match_mat_d[
                    b, cri_st1:cri_ed1, cri_st2:cri_ed2
                    ] + match_mat_d[b, cri_st2:cri_ed2, cri_st1:cri_ed1].transpose(1, 0)
            mat_s = mat_s + mat_s2
            if n_valid[b] > 2 and mat_s == 0 and sum_full_matched > 0:
                continue
            pc1 = part_pcs[b, pc_st1:pc_ed1]  # N, 3
            pc2 = part_pcs[b, pc_st2:pc_ed2]  # N, 3
            if critical_pcs_idx is not None:
                critical_pcs_src = pc1[
                    critical_pcs_idx[b, pc_st1: pc_st1 + n1]
                ]
                critical_pcs_tgt = pc2[
                    critical_pcs_idx[b, pc_st2: pc_st2 + n2]
                ]
                rot, trans = pairwise_alignment(
                    critical_pcs_src.cpu().numpy(),
                    critical_pcs_tgt.cpu().numpy(),
                    mat_d,
                )
                rot = torch.tensor(
                    rot, dtype=torch.float32, device=mat.device
                ).reshape(3, 3)
                trans = torch.tensor(
                    trans, dtype=torch.float32, device=mat.device
                ).reshape(3)
                new_critical_pcs_src = torch.matmul(
                    rot, critical_pcs_src.transpose(1, 0)
                ).transpose(1, 0)
                new_critical_pcs_src = new_critical_pcs_src + trans
                new_critical_pcs_src = new_critical_pcs_src * torch.sum(
                    mat, dim=-1
                ).reshape(-1, 1)
                new_critical_pcs_tgt = torch.matmul(mat, critical_pcs_tgt)
                pair_loss = (
                        new_critical_pcs_src ** 2
                        + new_critical_pcs_tgt ** 2
                        - 2 * new_critical_pcs_src * new_critical_pcs_tgt
                )
                pair_loss = torch.sum(pair_loss)
                n_sum += critical_pcs_src.shape[0]
                loss += pair_loss * mat_s
                # if math.isnan(pair_loss.item()):
                #     print('rot = ', rot)
                #     print('trans = ', trans)
                #     print(critical_pcs_src.shape, critical_pcs_tgt.shape)
                #     print(critical_pcs_src, critical_pcs_src.cpu().numpy())

    return loss / n_sum

