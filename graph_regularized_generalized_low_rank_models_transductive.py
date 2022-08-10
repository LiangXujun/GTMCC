# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:45:12 2022

@author: liangxj
"""

#%%
import pandas as pd
import numpy as np
from scipy.linalg import svd, norm, inv
from sklearn.model_selection import KFold

np.random.seed(1)

#%%
def get_lap(S, nn):
    n = S.shape[0]
    if np.sum(np.diag(S) == 0) != n:
        S = S - np.eye(n)
        
    ind_knn = np.argsort(-S, 1)
    Sknn = np.zeros((n, n))
    for j in range(n):
        ind = ind_knn[j,0:nn]
        Sknn[j,ind] = S[j,ind]
        Sknn[ind,j] = S[ind,j]
    
    D = np.diag(1/(np.sqrt(Sknn.sum(1))+1e-9))
    Sknn = D@Sknn@D
    L = np.eye(n) - Sknn
    return L


def gradW(W, H, Z, test, c, beta):
    Oz = np.ones(Z.shape)
    Oz[test,0:c] = 0
    Pz = np.array(Z!=0, int)
    # Pz = Z
    gW = 2*(1 - beta)*H@((H.T@W - Z.T)*Oz.T) + 2*(2*beta -1)*H@((H.T@W - Z.T)*Pz.T)
    return gW


def gradH(W, H, Z, test, c, beta):
    Oz = np.ones(Z.shape)
    Oz[test,0:c] = 0
    Pz = np.array(Z!=0, int)
    # Pz = Z
    gH = 2*(1 - beta)*W@((W.T@H - Z)*Oz) + 2*(2*beta -1)*(W@(Pz*(W.T@H - Z)))
    return gH


def cost_fun(W, H, Z, test, c, beta):
    Oz = np.ones(Z.shape)
    Oz[test,0:c] = 0
    Pz = np.array(Z!=0, int)
    # Pz = Z
    f = (1 - beta)*norm(Oz*(Z - W.T@H), 'fro')**2 + (2*beta - 1)*norm((Pz*(Z - W.T@H)))**2
    return f

def obj_fun(W, H, Z, L, test, c, beta, gam, lam1, lam2):
    p1 = cost_fun(W, H, Z, test, c, beta)
    p2 = gam*np.trace(W@L@W.T) + lam1*norm(W, 'fro')**2
    p3 = lam2*norm(H, 'fro')**2
    return p1 + p2 + p3


def gtmcc(Z, L, k, test, c, beta, gam, lam1, lam2):
    n = L.shape[0]
    U, s, Vh = svd(Z, full_matrices = False)
    W = U[:,0:k].T
    H = Vh[0:k,:]
    
    obj_old = obj_fun(W, H, Z, L, test, c, beta, gam, lam1, lam2)
    for it in range(100):
        gW = gradW(W, H, Z, test, c, beta)
        tw = 2*beta*norm(H@H.T)
        G = W - 1/tw*gW
        W = tw*G@inv(2*gam*L + 2*lam1*np.eye(n) + tw*np.eye(n))
        
        gH = gradH(W, H, Z, test, c, beta)
        th = 2*beta*norm(W@W.T)
        V = H - 1/th*gH
        H = th*V/(th + 2*lam2)
        
        obj = obj_fun(W, H, Z, L, test, c, beta, gam, lam1, lam2)
        diff_obj = obj_old - obj
        # print(it, obj, diff_obj, diff_obj/obj)
        obj_old = obj
    
    Z_pred = W.T@H
    return Z_pred

#%%
drug2target = pd.read_csv('./drug_to_drugbank_and_stitch_target_uniprot_table.txt', sep = '\t', header = 0, index_col = 0)
drug2side = pd.read_csv('./drug_to_side_cui_table.txt', sep = '\t', header = 0, index_col = 0)
drug2target_new = pd.read_csv('./new_drug_to_drugbank_and_stitch_target_uniprot_table.txt',  sep = '\t', header = 0, index_col = 0)
drug_sim = pd.read_csv('./new_drug_pubchem_stitch_sim.txt',  header = 0, index_col = 0, sep = '\t')


#%%
Y = np.r_[drug2side.values.T, np.zeros((drug2target_new.shape[1], drug2side.shape[0]))]
X = np.r_[drug2target.values.T, drug2target_new.values.T]

S = drug_sim.values
L = get_lap(S, 10)

n, c= Y.shape
d = X.shape[1]

#%%
k = 400
beta =  0.8
gam = 10
lam1 = 2
lam2 = 1

test = np.arange(drug2side.shape[1], n)
Y_ob = Y.copy()
Y_ob[test,] = 0

X_ob = X.copy()

Z = np.hstack((Y_ob, X_ob))
Z_pred = gtmcc(Z, L, k, test, c, beta, gam, lam1, lam2)

Y_pred = Z_pred[:,0:c]
X_pred = Z_pred[:,c:]

#%%
from scipy.io import savemat
savemat('./gtmcc_se_pred_new_stitch.mat', {'X_pred':X_pred, 'Y_pred':Y_pred})
