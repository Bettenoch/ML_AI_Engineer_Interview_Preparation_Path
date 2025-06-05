import numpy as np

class PCFromScratch:
    def __init__(self, n_components = None):
        self.n_components = n_components #show the number of components to keep
        self.components_ = None #total datasets
        self.mean_ = None
        self.explained_variance_ = None #shows the variance of each PC how much info each pc hold
        self.explained_variance_ratio_ = None #shows percentage of detail each PC in relation to variance with whole dataset
        
        
    def fit(self, X):
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        #lets calculate the covariance matric
        
        n_samples = X_centered.shape[0]
        
        covariance_matrix = np.dot(X_centered.T, X_centered)/n_samples - 1
        
        print(f"Covariance Matrix shape: {covariance_matrix.shape}")
        
        print("\nStep 3: Computing eigenvalues and eigenvectors...")
        eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
        
        idx = np.argsort(eigenvalues)[::-1]
        
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        print(f"Eigenvalues (sorted): {eigenvalues}")
        
        if self.n_components is None:
            self.n_components = len(eigenvalues)
            
        print(f"\n Step 5: Selecting {self.n_components} components")
        
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        total_variance = np.sum(eigenvalues)
        
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        print(f"Explained variance ratio: {self.explained_variance_ratio_}")
        return self
    
    def transform(self, X):
        X_centered = X - self.mean_
    
        X_transformed = np.dot(X_centered, self.components_.T)
        print(f"Transformed data shape: {X_transformed.shape}")
        return X_transformed
    
    def fit_transform(self, X):
        return self.fit(X).transform(X)
    def inverse_transform(self, X_transformed):
        X_reconstructed = np.dot(X_transformed, self.components_) + self.mean_
        
        return X_reconstructed