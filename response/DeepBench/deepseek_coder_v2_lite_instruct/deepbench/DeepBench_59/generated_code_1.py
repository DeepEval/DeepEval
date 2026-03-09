import torch

def get_perspective_transform(points_src, points_dst):
    B = points_src.shape[0]
    # Normalize the points to ensure numerical stability
    src_mean = points_src.mean(dim=1, keepdim=True)
    dst_mean = points_dst.mean(dim=1, keepdim=True)
    src_norm = points_src - src_mean
    dst_norm = points_dst - dst_mean
    src_norm = src_norm / torch.norm(src_norm, dim=2, keepdim=True)
    dst_norm = dst_norm / torch.norm(dst_norm, dim=2, keepdim=True)

    # Construct the A matrix
    A = []
    for i in range(B):
        x, y = src_norm[i].t()
        x_dash, y_dash = dst_norm[i].t()
        A.append(torch.stack([
            torch.stack([x, y, 1, 0, 0, 0, -x * x_dash, -y * x_dash, -x_dash]),
            torch.stack([0, 0, 0, x, y, 1, -x * y_dash, -y * y_dash, -y_dash])
        ]))
    A = torch.stack(A).view(B, -1, 9)

    # Solve the linear system to get the homography matrix
    U, S, V = torch.svd(A)
    H = V[:, -1].view(B, 3, 3)

    # Denormalize the homography matrix
    H = H / H[:, 2:3, 2:3]
    H = torch.matmul(dst_mean.permute(0, 2, 1), torch.matmul(H, src_mean.permute(0, 2, 1)))
    return H

if __name__ == "__main__":
    # Example usage
    points_src = torch.tensor([
        [[100, 200], [300, 200], [100, 400], [300, 400]],
        [[50, 50], [150, 50], [50, 150], [150, 150]]
    ])
    points_dst = torch.tensor([
        [[150, 150], [350, 150], [150, 350], [350, 350]],
        [[100, 100], [200, 100], [100, 200], [200, 200]]
    ])

    H = get_perspective_transform(points_src, points_dst)
    print(H)