import numpy as np
from sklearn import svm
from sklearn.metrics import roc_auc_score

def find_gamma(D_matrix):
  assert D_matrix.shape[0] == D_matrix.shape[1]
  best_value = None
  best_gamma = None

  for gamma in np.logspace(-1,1,50):
    D_kernel = np.exp(-gamma*D_matrix)
    D_kernel = -np.sort(-D_kernel,axis=1)
    n=D_kernel.shape[0]
    a = D_kernel[:,:int(0.1*n)].sum()-n
    b = D_kernel.sum()-n
    value = abs(a/b - 0.9)

    if not best_gamma:
      best_gamma = gamma
      best_value = value
    else:
      if value<best_value:
        best_value = value
        best_gamma = gamma
  
  return best_gamma

def find_nu(train_inlier, test_inlier, test_outlier,best_gamma):
  best_nu = None
  best_auc =None

  for nu in np.logspace(-3,-0.3,20):
    clf = svm.OneClassSVM(nu=nu, kernel="rbf",gamma=best_gamma)
    clf.fit(train_inlier)
    value1 = clf.decision_function(test_inlier)
    value2 = clf.decision_function(test_outlier)
    value = np.concatenate((value1,value2),axis=0)
    label = np.concatenate( (np.ones(len(test_inlier)),np.zeros(len(test_outlier)) ),axis=0 )
    auc = roc_auc_score(label,value)
    
    if not best_nu:
      best_nu = nu
      best_auc = auc
    else:
      if auc>best_auc:
        best_auc = auc
        best_nu = nu