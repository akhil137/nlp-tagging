import numpy as np
#from scipy import linalg
from scipy import sparse
from scipy.sparse import linalg as sp_linalg
#Number of samples(ns), features (nd), labels (nl), and desired rank (nk)
ns,nd,nl,nk=21000,900,1,20

#create some sparse test matrices
y=np.random.randint(0,2,size=[ns,nl]) #label matrix has to be dense for cg
#y=sparse.csr_matrix(y)
X=sparse.rand(ns,nd,density=0.1,format='csr')

X1 = sp_linalg.aslinearoperator(X)
coefs = np.empty((y.shape[1], nd))

alpha=0.5; #alpha in cg algo
max_iter=None
tol=1e-3

def mv(x): #define the matrix-vector product needed for cg
	return X1.rmatvec(X1.matvec(x)) + alpha * x
#Note we are solving
#min xT(AT.A+alpha*I)x-2bT.A.x for x, e.g. AT.A.x~AT.b (1)
#akin to solving min xT.A.x-2b.x in usual scipy cg algo A.x~b (2)
for i in range(y.shape[1]):
	y_column = X1.rmatvec(y[:,i]) #define the rhs as in (1)
	C = sp_linalg.LinearOperator((nd, nd), 
			matvec=mv, dtype=X.dtype)
	coefs[i], info = sp_linalg.cg(C, y_column, maxiter=max_iter,tol
			=tol)



