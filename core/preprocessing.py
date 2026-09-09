from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

def build_preprocessor(pca_components=None):
    steps = [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
    if pca_components is not None:
        steps.append(("pca", PCA(n_components=pca_components, random_state=42)))
    return Pipeline(steps)

def safe_pca_components(requested, n_samples, n_features):
    return max(1, min(int(requested), int(n_features), max(1, int(n_samples) - 1)))
