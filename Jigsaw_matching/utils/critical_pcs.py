import torch


def get_critical_pcs_from_label(critical_label, n_pcs):
    B, N_sum = critical_label.shape
    P = n_pcs.shape[-1]
    n_pcs_cumsum = torch.cumsum(n_pcs, dim=1).to(torch.int64)
    n_critical_pcs = torch.zeros_like(n_pcs)  # B, P
    critical_pcs_idx = torch.zeros_like(critical_label).to(torch.int64)
    for b in range(B):
        for p in range(P):
            st = 0 if p == 0 else n_pcs_cumsum[b, p - 1]
            ed = n_pcs_cumsum[b, p]
            c_label = critical_label[b, st:ed]
            c_idx = c_label.nonzero().reshape(-1)
            n_critical_pcs[b, p] = c_idx.shape[0]
            critical_pcs_idx[b, st: st + c_idx.shape[0]] = c_idx
    return critical_pcs_idx, n_critical_pcs
