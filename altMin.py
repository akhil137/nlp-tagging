#Solve min sqr loss of Y-X.Z + r(Z) ; dims: nxL=(nxd)(dxL) 
#using lower rank Z=W.(H.t) ; dims dxL=(dxk)(kxL)
#r(Z) translates to frob norm of W and H

#Algo uses alt min:
#1. init H=H0, min for W=W1
#2  min for H=H1 using W1
#3 repeat

#scipy.cg is supposed to solve Ax=b
#but can solve sqr loss with appropriate b and hessian
#which represents the matrix-vector product

#Note for stacking columns vec(W) = np.ravel(W,order='F')
#likewise, given a vec w, build a matrix W, s.t. vec(W)=w
#W=np.reshape(w,(d,k),order='F')
#Note: Fortran order is slower in numpy

#min problem for W
#min||v(Y)-U.v(W)||^2 + alpha/2 ||v(W)||^2 
#where v(.) is vectorization, i.e. stacking columns
#and U is nLxdk matrix made from X and H
import numpy as np
from scipy.sparse import linalg as sp_linalg

#yv=np.ravel(Y,order='F')

def solveW(H):
	bw=np.ravel(Xlo.rmatvec(Y.dot(H)),order='F')
	H2=H.T.dot(H)
	def mvW(s):
		S=np.reshape(s,(nd,nk),order='F')
		return np.ravel(Xlo.rmatvec(Xlo.matmat(S.dot(H2))),order='F') + alpha*s

	Cw=sp_linalg.LinearOperator((nd*nk, nd*nk), matvec=mvW, dtype=Xlo.dtype)
	w, info = sp_linalg.cg(Cw, bw, maxiter=max_iter,tol=tol)
	return np.reshape(w,(nd,nk),order='F')


def solveH(W):
	h=np.empty((nl,nk))
	A=sp_linalg.aslinearoperator(Xlo.matvec(W))
	def mvH(s):
		return A.rmatvec(A.matvec(s))+alpha*s
	for i in range(Y.shape[1]):
		Ch=sp_linalg.LinearOperator((nk,nk),matvec=mvH,dtype=Xlo.dtype)
		bh=A.rmatvec(Y[:,i])
		h[i], info=sp_linalg.cg(Ch,bh,maxiter=max_iter,tol=tol)
	return h

#put in a loop
#initial val for h


def alMin(itr=10):
	i=0
	Hn=H0
	Wn=np.empty((nd,nk),dtype='float64')
	while i<itr:
		Wn=solveW(Hn)
		Hn=solveH(Wn)
		i+=1
	return Hn, Wn



#time scaling results
#nn,nd,nl,nk=1000,700,500,100
#CPU times: user 485.67 s, sys: 14.15 s, total: 499.82 s
#Wall time: 340.22 s
#f1=0.67358



#Initialize
nn,nd,nl,nk=10,7,5,5
X=np.random.randn(nn,nd)
Y=np.random.randint(0,2,(nn,nl))
#Or create a sparse matrix directly
X=scipy.sparse.rand(nn,nd,density=0.01,format='csr',dtype=np.float64)
Y=scipy.sparse.rand(nn,nl,density=0.01,format='csr',dtype=np.float64)

#Define matrix-vector product for W, using Hessian in Fu
Xlo=sp_linalg.aslinearoperator(X)
alpha=0.5
max_iter=20
tol=1e-8

H0=np.random.randn(nl,nk)*0.01
%time Hf,Wf=alMin(itr=100)
Zfact=Wf.dot(Hf.T)
Yhat=X.dot(Zfact)
from sklearn.metrics import f1_score #and hamming_loss if avail
err=np.linalg.norm(Y-Yhat)
#hl=hamming_loss(Y,np.round(Yhat))*nl
f1=f1_score(Y,np.round(Yhat))


#--------------------------------------------
#debug algo
#--------------------------------------------
u,s,v=np.linalg.svd(X,full_matrices=0)
M=u.T.dot(Y)
Z=np.dot(v.T,np.dot(np.diag(np.reciprocal(s)),M)) #solves L2 min analytically w/ alpha=0

#Now do the factorization into W and H analytically
#and check Zc=Wc.dot(Hc.T) is close to the analytic Z above
um,sm,vm=np.linalg.svd(M,full_matrices=0)
Hc=vm
Xpinv=np.linalg.pinv(X)
Wc=Xpinv.dot(Y.dot(Hc))
Zc=c=Wc.dot(Hc.T) #This is true
np.allclose(Z,Zc) #is true

#Now use above cg to solve for H with alpha=0
nk=5
alpha=0
max_iter=20
tol=1e-8
H=solveH(Wc)
np.allclose(H,Hc) #true!!!!
W=solveW(H)
np.allclose(W,Wc) #also true!!


#form the Z matrix from factors and check it agains 
H0=0.1*Hc
Hf,Wf=alMin(itr=10)
Zfact=Wf.dot(Hf.T)
np.allclose(Z,Zfact) #it works!!

#Try again with random initilization for H
np.random.seed(1979)
H0=np.random.randn(nl,nk)*0.01
Hf,Wf=alMin(itr=100)
Zfact=Wf.dot(Hf.T)
np.allclose(Z,Zfact) 

#Now lets plot the convergence (i.e |Z-Zfact|_F vs. altMin iterations)
NumItr=4
normZ=np.empty((NumItr,))
for i in range(NumItr):
	np.random.seed(1979)
	H0=np.random.randn(nl,nk)*0.01
	Hf,Wf=alMin(itr=10**i)
	print i
	Zfact=Wf.dot(Hf.T)
	normZ[i]=np.linalg.norm(Z-Zfact)

#After about 100 iterations, we have very little improvement
#np.log10(normZ)= 
#array([-1.60717082, -2.19423294, -2.35645334, -2.53816142, -2.54171448])













	







