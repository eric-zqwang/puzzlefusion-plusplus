import torch
from pytorch3d import transforms
from pytorch3d import ops
from torch_cluster import fps


def append_trans_and_rots(nodes, trans, rots):
    for i, attr in nodes(data=True):
        pivot = attr["pivot"]
        if rots is not None:
            attr["trans_and_rots"].append(torch.cat([trans[pivot], rots[pivot]]))
        else:
            attr["trans_and_rots"].append(trans[pivot])


def get_final_pose_pts_dynamic(pts, n_pcs, pred_trans, pred_rots, num_parts, nodes):
    """
    Currently the batch size is 1

    pts: (B, 5000, 3)
    n_pcs: (B, P)
    pred_trans (B, P, 3)
    pred_rots (B, P, 4)
    """
    
    index = 0
    final_pts = []
    for i in range(num_parts[0].item()):
        c_pts = pts[0][index:index+n_pcs[0][i]]

        c_pivot = nodes[i]["pivot"]

        c_rots = pred_rots[0][c_pivot]
        c_trans = pred_trans[0][c_pivot]

        c_pts = transforms.quaternion_apply(c_rots, c_pts)
        c_pts = c_pts + c_trans
        final_pts.append(c_pts)
        index += n_pcs[0][i]
    
    return torch.cat(final_pts, dim=0)

def get_pc_start_end(idx, n_pcs):
    n_pcs_cumsum = n_pcs.cumsum(dim=1)
    pc_st1 = 0 if idx == 0 else n_pcs_cumsum[0, idx - 1]
    pc_ed1 = n_pcs_cumsum[0, idx]
    return pc_st1, pc_ed1


def get_distance_for_matching_pts(
        idx1, idx2, pts, n_pcs, 
        n_critical_pcs, critical_pcs_idx, 
        corr, data_id, cd_loss,
        method="outlier"):
    """
    pts: 5000, 3
    """
    
    pc_st1, pc_ed1 = get_pc_start_end(idx1, n_pcs)
    pc_st2, pc_ed2 = get_pc_start_end(idx2, n_pcs)

    n1 = n_critical_pcs[0, idx1]
    n2 = n_critical_pcs[0, idx2]
    pc1 = pts[pc_st1:pc_ed1]  # N, 3
    pc2 = pts[pc_st2:pc_ed2]  # N, 3
    critical_pcs_idx_1 = critical_pcs_idx[0, pc_st1: pc_st1 + n1]
    critical_pcs_idx_2 = critical_pcs_idx[0, pc_st2: pc_st2 + n2]
    critical_pcs_src = pc1[critical_pcs_idx_1]
    critical_pcs_tgt = pc2[critical_pcs_idx_2]
    
    corr_pts_0 = critical_pcs_src[corr[0][:, 0]]
    corr_pts_1 = critical_pcs_tgt[corr[0][:, 1]]

    # if data_id != None:
    #     vis_pair(pc1, pc2, corr_pts_0, corr_pts_1, data_id, idx1, idx2)
    
    return cd_loss(corr_pts_0.unsqueeze(0), corr_pts_1.unsqueeze(0), bidirectional=True, point_reduction=None, batch_reduction=None)


def node_merge_valids_check(edges, ref_part, nodes):
    ref_part_idx = torch.where(ref_part)[1]
    idx1 = edges[0]
    idx2 = edges[1]

    pivot_idx1 = nodes[idx1.item()]["pivot"]
    pivot_idx2 = nodes[idx2.item()]["pivot"]

    if torch.isin(ref_part_idx, edges).any():
        return False
    
    if ref_part[0][pivot_idx1] or ref_part[0][pivot_idx2]: # Any of part is reference part
        return False
    
    return True


def ref_part_classified_check(edges, ref_part, neighbor_part):
    ref_part_idx = torch.where(ref_part)[1]
    neighbor_parts_idx = torch.where(neighbor_part)[1]

    idx1 = edges[1]
    idx2 = edges[0]

    if torch.isin(ref_part_idx, edges).any() and torch.isin(neighbor_parts_idx, edges).any():
        if ref_part[0][idx1] and ref_part[0][idx2]: 
            return False
        else:
            return True
    
    return False


def merge_node(components, nodes, pcs):
    components = list(components)
    merge_pcs = []
    for idx in components:
        if nodes[idx]["valids"] == False:
            continue
        merge_pcs.append(pcs[idx])

    merge_pcs = torch.cat(merge_pcs, dim=0)

    return merge_pcs


def sample_by_scale(components, part_scale, pcs, merge_pcs, nodes, num_points=1000):
    # sample 1000 pts from the merged pc based on the part scale
    weights = []
    for c in components:
        if nodes[c]["valids"] == False:
            continue
        weights.append(part_scale[c])
    weights = torch.tensor(weights, dtype=torch.float, device=pcs.device)

    weights = weights / weights.sum()
    
    # Repeat weights for each point in each part
    weights = weights.repeat_interleave(pcs[0].shape[0])

    # Sample 1000 points from the merged pc based on the weights
    indices = torch.multinomial(weights, num_samples=num_points, replacement=False)
    sampled_pcs = merge_pcs[indices]

    return sampled_pcs


def remove_intersect_points_and_fps_ds(merge_pcs, cd_loss, num_points=1000, threshold=0.001):
    """
    merge_pcs: (P*N, 3)
    """
    
    merge_pcs = merge_pcs.reshape(-1, num_points, 3)

    P, num_point, _ = merge_pcs.shape

    index = 0

    normals = ops.estimate_pointcloud_normals(merge_pcs, neighborhood_size=20)
    
    normals = normals.reshape(-1, num_point, 3)
    
    final_pts_per_part = [None] * P  # P is the total number of parts

    for i in range(P):
        # Initialize a mask for keeping points in part i, assuming initially we keep all
        keep_mask_i = torch.ones(merge_pcs[i].shape[0], dtype=torch.bool)
        
        for j in range(P):
            if i == j:
                continue
            
            per_point_cd = cd_loss(
                merge_pcs[i].unsqueeze(0), 
                merge_pcs[j].unsqueeze(0), 
                batch_reduction=None,
                point_reduction=None, 
                bidirectional=True,
            ).squeeze(0)
            
            # Determine points within the threshold distance
            within_threshold = per_point_cd < threshold
            
            # Calculate the dot product for normals of points within the threshold
            dot_product = torch.sum(normals[i][within_threshold] * normals[j][within_threshold], dim=1)
            
            # Identify points with a different normal (negative dot product)
            different_normals = dot_product < 0
            
            # Update the keep_mask for part i based on this comparison with part j
            # Only update for points within the threshold distance
            threshold_indices = torch.where(within_threshold)[0]
            keep_mask_i[threshold_indices[different_normals]] = False
        
        # Apply the mask to keep the desired points in merge_pcs[i]
        filtered_merge_pcs_i = merge_pcs[i][keep_mask_i]
        
        # Store the filtered points for part i
        final_pts_per_part[i] = filtered_merge_pcs_i

    # Optionally, if you want to concatenate all filtered parts into a single tensor
    final_pts_concatenated = torch.cat(final_pts_per_part, dim=0)
    
    # density sample 1000 points from the final_pts_concatenated
    # sample by density
    
    ratio = num_points / final_pts_concatenated.shape[0]
    index = fps(final_pts_concatenated, ratio=ratio)
    final_pts_concatenated = final_pts_concatenated[index][:num_points]
    
    return final_pts_concatenated


def assign_init_pose(nodes, trans, rots, centroid, component):
    for idx in component:
        node = nodes[idx]
        pivot = node["pivot"]

        c_trans = trans[pivot]
        c_rots = rots[pivot]

        rot_m = transforms.quaternion_to_matrix(c_rots)

        affine_matrix = torch.eye(4, device=trans.device)
        affine_matrix[:3, :3] = rot_m
        affine_matrix[:3, 3] = c_trans - centroid

        if node["init_pose"] is None:
            init_pose = affine_matrix
        else:
            init_pose = affine_matrix @ node["init_pose"]

        node["init_pose"] = init_pose

def extract_final_pred_trans_rots(pred_trans, pred_rots, nodes):
    final_trans = torch.zeros_like(pred_trans, device=pred_trans.device)
    final_rots = torch.zeros_like(pred_rots, device=pred_rots.device)

    for i, attr in nodes(data=True):
        init_pose = attr["init_pose"]
        pivot = attr["pivot"]

        c_trans = pred_trans[pivot]
        c_rots = pred_rots[pivot]

        rot_m = transforms.quaternion_to_matrix(c_rots)

        affine_matrix = torch.eye(4, device=pred_trans.device)
        affine_matrix[:3, :3] = rot_m
        affine_matrix[:3, 3] = c_trans
        if init_pose is None:
            final_trans_rots = affine_matrix
        else:
            final_trans_rots = affine_matrix @ init_pose
        
        trans = final_trans_rots[:3, 3]
        rots = transforms.matrix_to_quaternion(final_trans_rots[:3, :3])
        final_trans[i] = trans
        final_rots[i] = rots

    return final_trans, final_rots


def get_param(param, nodes):
    trans = param[:, :3]
    rots = param[:, 3:]
    rots_m = transforms.quaternion_to_matrix(rots)
    
    final_trans = torch.zeros_like(trans, device=param.device)
    final_rots = torch.zeros_like(rots_m, device=param.device)
    affine_matrix = torch.eye(4, device=param.device)
    
    for i, attr in nodes(data=True):
        pivot = attr["pivot"]
        init_pose = attr["init_pose"]
        c_trans = trans[pivot]
        c_rots_m = rots_m[pivot]

        
        affine_matrix[:3, :3] = c_rots_m
        affine_matrix[:3, 3] = c_trans

        if init_pose is not None:
            final_trans_rots = affine_matrix @ init_pose 
        else:
            final_trans_rots = affine_matrix

        final_trans[i] = final_trans_rots[:3, 3]
        final_rots[i] = final_trans_rots[:3, :3]

    
    final_rots = transforms.matrix_to_quaternion(final_rots)
    final_param = torch.cat([final_trans, final_rots], dim=1)

    return final_param
