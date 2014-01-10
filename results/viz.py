from pylab import *

A = fromfile("output.dat", sep=",")
A = A[:-1]
A = reshape(A, [32, 64, 64])
[x, y] = meshgrid(range(64), range(64))
z = A[10, :, :]

contourf(x, y, z)
show()