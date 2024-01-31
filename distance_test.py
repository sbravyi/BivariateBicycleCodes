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

	# maximum stabilizer weight
	wstab = np.max([np.sum(stab[i,:]) for i in range(m)])
	# weight of the logical operator
	wlog = np.count_nonzero(logicOp)
	# how many slack variables are needed to express orthogonality constraints modulo two
	num_anc_stab = int(np.ceil(np.log2(wstab)))
	num_anc_logical = int(np.ceil(np.log2(wlog)))
	# total number of variables
	num_var = n + m*num_anc_stab + num_anc_logical

	model = Model()
	model.verbose = 0
	x = [model.add_var(var_type=BINARY) for i in range(num_var)]
	model.objective = minimize(xsum(x[i] for i in range(n)))

	# orthogonality to rows of stab constraints
	for row in range(m):
		weight = [0]*num_var
		supp = np.nonzero(stab[row,:])[0]
		for q in supp:
			weight[q] = 1
		cnt = 1
		for q in range(num_anc_stab):
			weight[n + row*num_anc_stab +q] = -(1<<cnt)
			cnt+=1
		model+= xsum(weight[i] * x[i] for i in range(num_var)) == 0

	# odd overlap with logicOp constraint
	supp = np.nonzero(logicOp)[0]
	weight = [0]*num_var
	for q in supp:
		weight[q] = 1
	cnt = 1
	for q in range(num_anc_logical):
			weight[n + m*num_anc_stab +q] = -(1<<cnt)
			cnt+=1
	model+= xsum(weight[i] * x[i] for i in range(num_var)) == 1

	model.optimize()

	opt_val = sum([x[i].x for i in range(n)])
	return int(opt_val)



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
# by solving an integer linear program (ILP). The ILP looks for a minimum weight Pauli Z-type operator which has an even overlap with each X-check 
# and an odd overlap with logical-X operator on the i-th logical qubit. Let w_i be the optimal value of this ILP. 
# Then the code distance for Z-type logical operators is dZ = min(w_1,…,w_k).
d = n
for i in range(k):
	w = distance_test(hx,lx[i,:])
	print('Logical qubit=',i,'Distance=',w)
	d = min(d,w)

print('Code parameters: n,k,d=',n,k,d)
