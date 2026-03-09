import torch

def _safe_det_3x3(t):
    r0, r1, r2 = t[:, 0, :], t[:, 1, :], t[:, 2, :]
    c0, c1, c2 = r0, r1, r2
    s1 = c0[1] * c2[2] - c0[2] * c2[1]
    s2 = c0[0] * c2[1] - c0[1] * c2[0]
    s3 = c0[0] * c1[2] - c0[2] * c1[0]
    s4 = c0[0] * c1[1] - c0[1] * c1[0]
    s5 = c1[0] * c2[2] - c1[2] * c2[0]
    s6 = c1[0] * c2[1] - c1[1] * c2[0]
    s7 = c2[0] * c1[2] - c2[2] * c1[0]
    s8 = c2[0] * c1[1] - c2[1] * c1[0]
    det = r0 * torch.stack([s1, s2, s3]).T + \
           r1 * torch.stack([s4, s5, s6]).T + \
           r2 * torch.stack([s7, s8, s9]).T
    return det

if __name__ == "__main__":
    t = torch.randn(4, 3, 3)
    det = _safe_det_3x3(t)
    print(det)