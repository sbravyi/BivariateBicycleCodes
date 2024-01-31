import numpy as np
from mip import Model, xsum, minimize, BINARY
from bposd.css import css_code

# computes the minimum Hamming weight of a binary vector x such that 
# stab @ x = 0 mod 2
# logicOp @ x = 1 mod 2
# here stab is a binary matrix and logicOp is a binary vector
def distance_test(stab,logicOp):
	# number of qubits
	n = stab.shape[1]
	# number of stabilizers
	m = stab.shape[0]

	# we assume that each stabilizer has weight 6
	for i in range(m):
		assert(np.count_nonzero(stab[i,:])==6)
	# Then a minimum weight logical operator overlaps with each stabilizer on 0 or 2 qubits

	# weight of the logical operator
	wlog = np.count_nonzero(logicOp)
	# how many slack variables are needed to express anti-commutation with logicOp
	num_slack = int(np.ceil(np.log2(wlog)))

	model = Model()
	model.verbose = 0
	# total number of variables
	num_var = n + n2 + num_slack 
	# first n variables parameterize a logical operator
	# next n2 variables are slack variables to express commutation with stabilizers 
	# last num_slack variables are slack variables to express anti-commutation with logicOp
	x = [model.add_var(var_type=BINARY) for i in range(num_var)]
	model.objective = minimize(xsum(x[i] for i in range(n)))

	# commutation with stabilizers
	for i in range(n2):
		model+= xsum([x[j] for j in range(n) if stab[i,j]==1]) == 2*x[n+i]

	# anti-commutation with logicOp
	model+= xsum([x[j] for j in range(n) if logicOp[j]==1]) == (1 + xsum([x[-j]*(1<<(j+1)) for j in range(num_slack)])) 

	model.optimize()
	return sum([x[i].x for i in range(n)])



# [[144,12,12]]
ell,m = 12,6
a1,a2,a3 = 3,1,2
b1,b2,b3 = 3,1,2


n = 2*ell*m
n2 = ell*m


# define cyclic shift matrices 
I_ell = np.identity(ell,dtype=int)
I_m = np.identity(m,dtype=int)
I = np.identity(ell*m,dtype=int)
x = {}
y = {}
for i in range(ell):
	x[i] = np.kron(np.roll(I_ell,i,axis=1),I_m)
for i in range(m):
	y[i] = np.kron(I_ell,np.roll(I_m,i,axis=1))

# define check matrices
A = (x[a1] + y[a2] + y[a3]) % 2
B = (y[b1] + x[b2] + x[b3]) % 2
AT = np.transpose(A)
BT = np.transpose(B)
hx = np.hstack((A,B))
hz = np.hstack((BT,AT))

qcode=css_code(hx,hz)
print('Testing CSS code...')
qcode.test()
print('Done')

lz = qcode.lz
lx = qcode.lx
k = lz.shape[0]

print('Computing code distance...')
# We compute the distance only for Z-type logical operators (the distance for X-type logical operators is the same)
# A minimum weight logical-Z overlaps with each X-check on 0 or 2 qubits (since each X-check has weight-6)
# and overlaps with the logical-X operator on at least one logical qubit on odd number of qubits
# For each logical qubit i=1,...,k solve a mixed integer linear program to compute minimum weight of such logical-Z operator
d = n
for i in range(k):
	wt = distance_test(hx,lx[i,:])
	print('Logical qubit=',i,'Distance=',wt)
	d = min(d,wt)

print('Code parameters: n,k,d=',n,k,d)
