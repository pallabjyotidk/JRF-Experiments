import numpy as np

x = np.array([[[1,2],[2,3],[3,4]], [[4,5], [5,6], [6,7]]])
print(x.shape)

y = x[:,None]
print(y.shape)

z = x[None, :]
print(z.shape)

x = x.ravel()
print(x)
print(x.shape)