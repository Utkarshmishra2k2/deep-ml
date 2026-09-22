import numpy as np

def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	matrix1 = np.array(matrix)
	eigenvalues,eigenvectors = np.linalg.eig(matrix1)

	return eigenvalues