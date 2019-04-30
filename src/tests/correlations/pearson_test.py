import numpy as np


x = np.array([[1, 2, 3, 4, 5, 6], [5, 6, 7, 8, 9, 9]]).T
pcc = PearsonCorrelationCoefficient()
pcc.compute_score(x)