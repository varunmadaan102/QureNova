from sklearn.svm import SVC

def build_precomputed_qsvc():
    # Use the decision function for ROC-AUC; sklearn 1.9 deprecates the
    # legacy probability=True implementation on SVC.
    return SVC(kernel="precomputed")
