import numpy as np


def pairwise_alignment(pointsS, pointsT, weight, method="horn87"):
    if method == "horn87":
        return horn_87(pointsS, pointsT, weight)
    else:
        raise NotImplementedError(f"{method} not implemented")


def horn_87(pointsS, pointsT, weight):
    pointsS = pointsS.transpose(1, 0)
    pointsT = pointsT.transpose(1, 0)
    centerS = pointsS.mean(axis=1)
    centerT = pointsT.mean(axis=1)
    pointsS = pointsS - centerS.reshape(-1, 1)
    pointsT = pointsT - centerT.reshape(-1, 1)

    M = pointsS @ weight @ pointsT.T
    N = np.array(
        [
            [
                M[0, 0] + M[1, 1] + M[2, 2],
                M[1, 2] - M[2, 1],
                M[2, 0] - M[0, 2],
                M[0, 1] - M[1, 0],
            ],
            [
                M[1, 2] - M[2, 1],
                M[0, 0] - M[1, 1] - M[2, 2],
                M[0, 1] + M[1, 0],
                M[0, 2] + M[2, 0],
            ],
            [
                M[2, 0] - M[0, 2],
                M[0, 1] + M[1, 0],
                M[1, 1] - M[0, 0] - M[2, 2],
                M[1, 2] + M[2, 1],
            ],
            [
                M[0, 1] - M[1, 0],
                M[2, 0] + M[0, 2],
                M[1, 2] + M[2, 1],
                M[2, 2] - M[0, 0] - M[1, 1],
            ],
        ]
    )
    v, u = np.linalg.eigh(N)
    id = v.argmax()

    q = u[:, id]
    R = np.array(
        [
            [
                q[0] ** 2 + q[1] ** 2 - q[2] ** 2 - q[3] ** 2,
                2 * (q[1] * q[2] - q[0] * q[3]),
                2 * (q[1] * q[3] + q[0] * q[2]),
            ],
            [
                2 * (q[2] * q[1] + q[0] * q[3]),
                q[0] ** 2 - q[1] ** 2 + q[2] ** 2 - q[3] ** 2,
                2 * (q[2] * q[3] - q[0] * q[1]),
            ],
            [
                2 * (q[3] * q[1] - q[0] * q[2]),
                2 * (q[3] * q[2] + q[0] * q[1]),
                q[0] ** 2 - q[1] ** 2 - q[2] ** 2 + q[3] ** 2,
            ],
        ]
    )

    pointsS = pointsS + centerS.reshape(-1, 1)
    pointsT = pointsT + centerT.reshape(-1, 1)

    t = (weight @ pointsT.T).T - (
            np.sum(weight, axis=-1).reshape((-1, 1)) * (R @ pointsS).T
    ).T
    t = np.sum(t, axis=-1) / np.sum(weight)
    return R.astype(np.float32), t.astype(np.float32)


if __name__ == "__main__":
    a = np.random.random((10, 3))
    from scipy.spatial.transform import Rotation as R

    rot = R.random().as_matrix()
    # rot = np.array([[0, -1, 0], [1, 0, 0], [0,0,1]])
    t = np.random.random(3)
    b = (rot @ a.T).T + t
    weight = np.eye(10) * 0.1
    print(rot, t)
    print(horn_87(a, b, weight))
