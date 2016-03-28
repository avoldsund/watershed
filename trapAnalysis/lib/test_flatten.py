import numpy as np
import time

start = time.time()
A = np.random.random((20000,20000))
end = time.time()
print 'Time spent creating random 2d array: '
print (end - start)

start1 = time.time()
D = A.flatten()
end1 = time.time()
print 'Time spent using flatten: '
print (end1 - start1)
#print D

start2 = time.time()
D = np.reshape(A, (1, np.product(A.shape)))[0]
end2 = time.time()
print 'Time spent using reshape: ' 
print (end2 - start2)
#print D

start3 = time.time()
D = np.ravel(A)
end3 = time.time()
print 'Time spent using ravel: '
print (end3 - start3)