import torch
import math

def function_name(k, alpha, beta, gamma):
    l = k
    if l % 2 == 0:
        l += 1
    
    def phi_j(l, m, phi):
        if l % 2 != 0:
            return math.sin(l*phi)/m
        else:
            return math.cos(l*phi)*math.pow(2*m, 0.25)
        
    def C(l, m):
        if l % 2 != 0:
            return math.sqrt((2*l + 1)/(4*math.pi)*math.pow((-1), (l - m)/2)*math.pow(l - m, 0.5))
        else:
            return 0
    
    def phi(alpha, beta, gamma):
        return alpha + beta + gamma
    
    d = torch.zeros((k+1, 2*k+1, 2*k+1))
    
    for l in range(k+1):
        for m in range(-l, l+1):
            d[:, m+l, m+l] += C(l, m) * torch.outer(phi_j(l, m, phi(beta, gamma, l)), phi_j(l, m, phi(alpha, gamma, l)))
            d[:, m+l+1, m+l+1] -= C(l, m) * torch.outer(phi_j(l, m, phi(beta, gamma, l)), phi_j(l, m, phi(alpha, gamma, l)))
            d[:, m+l, m+l+1] = d[:, m+l+1, m+l] = C(l, m) * torch.outer(phi_j(l, m, phi(beta, alpha, l)), phi_j(l, m, phi(gamma, alpha, l)))
    
    return d

if __name__ == "__main__":
    k = 2
    alpha = torch.tensor([math.pi/2])
    beta = torch.tensor([math.pi/4])
    gamma = torch.tensor([math.pi/6])
    d_matrix = function_name(k, alpha, beta, gamma)
    print(d_matrix)