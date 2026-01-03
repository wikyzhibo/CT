import numpy as np


A = np.array([[-1,-2,6],[-1,0,3],[-1,-1,4]],dtype=int)
print(A)

B = A - 3 * np.eye(3)
print(f"B={B}")

C = np.array([[-1,-3,0],
                    [1,0,1],
                    [0,1,1]],dtype=int)
D = np.invert(C)

E = np.matmul(D,A)
E = np.matmul(E,C)
print(f"E={E}")
