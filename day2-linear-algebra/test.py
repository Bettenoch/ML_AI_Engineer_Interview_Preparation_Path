import numpy as np

arrlist = [3.5, 5.6, 9.0, 2.3]
arrlist.sort()
print (arrlist[::1])
eigenvalues = np.array([4.6, 8.1, 0.2])
eigenvectors = np.array([
    [0.1, 0.8, 0.4],
    [0.2, 0.2, 0.6],
    [0.3, 0.7, 0.2]
])

idx = np.argsort(eigenvalues)[::-1]

print(f"Indices of the eigen values, {idx}")

eigenvalues = eigenvalues[idx] # Only NumPy arrays support indexing with another array (e.g. eigenvalues[idx]).
print(eigenvalues)
print("#" * 30)
print("original array \n", eigenvectors)
print("\n")
print(eigenvectors[:, idx])