import numpy as np
import itertools
from ldpc import bposd_decoder
from bposd.css import css_code
import pickle
from scipy.sparse import coo_matrix
from scipy.sparse import hstack


# Takes as input a binary square matrix A
# Returns the rank of A over the binary field F_2
def rank2(A):
    rows,n = A.shape
    X = np.identity(n,dtype=int)

    for i in range(rows):
        y = np.dot(A[i,:], X) % 2
        not_y = (y + 1) % 2
        good = X[:,np.nonzero(not_y)]
        good = good[:,0,:]
        bad = X[:, np.nonzero(y)]
        bad = bad[:,0,:]
        if bad.shape[1]>0 :
            bad = np.add(bad,  np.roll(bad, 1, axis=1) ) 
            bad = bad % 2
            bad = np.delete(bad, 0, axis=1)
            X = np.concatenate((good, bad), axis=1)
    # now columns of X span the binary null-space of A
    return n - X.shape[1]





# depolarizing noise model 
error_rate = 0.003
error_rate_init = error_rate
error_rate_idle = error_rate
error_rate_cnot = error_rate
error_rate_meas = error_rate




# syndrome cycle with 7 CNOT rounds
# sX and sZ define the order in which X-check and Z-check qubit
# is coupled with the neighboring data qubits
# We label the six neighbors of each check qubit in the Tanner graph
# by integers 0,1,...,5
sX= ['idle', 1, 4, 3, 5, 0, 2]
sZ= [3, 5, 0, 1, 2, 4, 'idle']


# number of syndrome measurement cycles 
num_cycles = 12


# Parameters of a Bivariate Bicycle (BB) code
# see Section 4 of https://arxiv.org/pdf/2308.07915.pdf for notations
# The code is defined by a pair of polynomials
# A and B that depends on two variables x and y such that
# x^ell = 1
# y^m = 1
# A = x^{a_1} + y^{a_2} + y^{a_3} 
# B = y^{b_1} + x^{b_2} + x^{b_3}

# [[144,12,12]]
ell,m = 12,6
a1,a2,a3 = 3,1,2
b1,b2,b3 = 3,1,2

# [[784,24,24]]
#ell,m = 28,14
#a1,a2,a3=26,6,8
#b1,b2,b3=7,9,20

# [[72,12,6]]
#ell,m = 6,6
#a1,a2,a3=3,1,2
#b1,b2,b3=3,1,2


# Ted's code [[90,8,10]]
#ell,m = 15,3
#a1,a2,a3 = 9,1,2
#b1,b2,b3 = 0,2,7

# [[108,8,10]]
#ell,m = 9,6
#a1,a2,a3 = 3,1,2
#b1,b2,b3 = 3,1,2

# [[288,12,18]]
#ell,m = 12,12
#a1,a2,a3 = 3,2,7
#b1,b2,b3 = 3,1,2

# code length
n = 2*m*ell


n2 = m*ell

# Compute check matrices of X- and Z-checks


# cyclic shift matrices 
I_ell = np.identity(ell,dtype=int)
I_m = np.identity(m,dtype=int)
I = np.identity(ell*m,dtype=int)
x = {}
y = {}
for i in range(ell):
	x[i] = np.kron(np.roll(I_ell,i,axis=1),I_m)
for i in range(m):
	y[i] = np.kron(I_ell,np.roll(I_m,i,axis=1))


A = (x[a1] + y[a2] + y[a3]) % 2
B = (y[b1] + x[b2] + x[b3]) % 2

A1 = x[a1]
A2 = y[a2]
A3 = y[a3]
B1 = y[b1]
B2 = x[b2]
B3 = x[b3]

AT = np.transpose(A)
BT = np.transpose(B)

hx = np.hstack((A,B))
hz = np.hstack((BT,AT))

# number of logical qubits
k = n - rank2(hx) - rank2(hz)


qcode=css_code(hx,hz)
print('Testing CSS code...')
qcode.test()
print('Done')

lz = qcode.lz
lx = qcode.lx


# Give a name to each qubit
# Define a linear order on the set of qubits
lin_order = {}
data_qubits = []
Xchecks = []
Zchecks = []

cnt = 0
for i in range(n2):
    node_name = ('Xcheck', i)
    Xchecks.append(node_name)
    lin_order[node_name] = cnt
    cnt += 1

for i in range(n2):
    node_name = ('data_left', i)
    data_qubits.append(node_name)
    lin_order[node_name] = cnt
    cnt += 1
for i in range(n2):
    node_name = ('data_right', i)
    data_qubits.append(node_name)
    lin_order[node_name] = cnt
    cnt += 1


for i in range(n2):
    node_name = ('Zcheck', i)
    Zchecks.append(node_name)
    lin_order[node_name] = cnt
    cnt += 1


# compute the list of neighbors of each check qubit in the Tanner graph
nbs = {}
# iterate over X checks
for i in range(n2):
	check_name = ('Xcheck',i)
	# left data qubits
	nbs[(check_name,0)] = ('data_left',np.nonzero(A1[i,:])[0][0])
	nbs[(check_name,1)] = ('data_left',np.nonzero(A2[i,:])[0][0])
	nbs[(check_name,2)] = ('data_left',np.nonzero(A3[i,:])[0][0])
	# right data qubits
	nbs[(check_name,3)] = ('data_right',np.nonzero(B1[i,:])[0][0])
	nbs[(check_name,4)] = ('data_right',np.nonzero(B2[i,:])[0][0])
	nbs[(check_name,5)] = ('data_right',np.nonzero(B3[i,:])[0][0])

# iterate over Z checks
for i in range(n2):
	check_name = ('Zcheck',i)
	# left data qubits
	nbs[(check_name,0)] = ('data_left',np.nonzero(B1[:,i])[0][0])
	nbs[(check_name,1)] = ('data_left',np.nonzero(B2[:,i])[0][0])
	nbs[(check_name,2)] = ('data_left',np.nonzero(B3[:,i])[0][0])
	# right data qubits
	nbs[(check_name,3)] = ('data_right',np.nonzero(A1[:,i])[0][0])
	nbs[(check_name,4)] = ('data_right',np.nonzero(A2[:,i])[0][0])
	nbs[(check_name,5)] = ('data_right',np.nonzero(A3[:,i])[0][0])


# syndrome measurement cycle as a list of operations
cycle = [] 
U = np.identity(2*n,dtype=int)
# round 0: prep xchecks, CNOT zchecks and data
t=0
for q in Xchecks:
	cycle.append(('PrepX',q))
data_qubits_cnoted_in_this_round = []
assert(not(sZ[t]=='idle'))
for target in Zchecks:
	direction = sZ[t]
	control = nbs[(target,direction)]
	U[lin_order[target],:] = (U[lin_order[target],:] + U[lin_order[control],:]) % 2
	data_qubits_cnoted_in_this_round.append(control)
	cycle.append(('CNOT',control,target))
for q in data_qubits:
	if not(q in data_qubits_cnoted_in_this_round):
		cycle.append(('IDLE',q))

# round 1-5: CNOT xchecks and data, CNOT zchecks and data
for t in range(1,6):
	assert(not(sX[t]=='idle'))
	for control in Xchecks:
		direction = sX[t]
		target = nbs[(control,direction)]
		U[lin_order[target],:] = (U[lin_order[target],:] + U[lin_order[control],:]) % 2
		cycle.append(('CNOT',control,target))
	assert(not(sZ[t]=='idle'))
	for target in Zchecks:
		direction = sZ[t]
		control = nbs[(target,direction)]
		U[lin_order[target],:] = (U[lin_order[target],:] + U[lin_order[control],:]) % 2
		cycle.append(('CNOT',control,target))

# round 6: CNOT xchecks and data, measure Z checks
t=6
for q in Zchecks:
	cycle.append(('MeasZ',q))
assert(not(sX[t]=='idle'))
data_qubits_cnoted_in_this_round = []
for control in Xchecks:
	direction = sX[t]
	target = nbs[(control,direction)]
	U[lin_order[target],:] = (U[lin_order[target],:] + U[lin_order[control],:]) % 2
	cycle.append(('CNOT',control,target))
	data_qubits_cnoted_in_this_round.append(target)
for q in data_qubits:
	if not(q in data_qubits_cnoted_in_this_round):
		cycle.append(('IDLE',q))

# round 7: all data qubits are idle, Prep Z checks, Meas X checks
for q in data_qubits:
	cycle.append(('IDLE',q))
for q in Xchecks:
	cycle.append(('MeasX',q))
for q in Zchecks:
	cycle.append(('PrepZ',q))

# full syndrome measurement circuit
cycle_repeated = num_cycles*cycle


# test the syndrome measurement circuit

# implement syndrome measurements using the sequential depth-12 circuit
V = np.identity(2*n,dtype=int)
# first measure all X checks
for t in range(7):
	if not(sX[t]=='idle'):
		for control in Xchecks:
			direction = sX[t]
			target = nbs[(control,direction)]
			V[lin_order[target],:] = (V[lin_order[target],:] + V[lin_order[control],:]) % 2
# next measure all Z checks
for t in range(7):
	if not(sZ[t]=='idle'):
		for target in Zchecks:
			direction = sZ[t]
			control = nbs[(target,direction)]
			V[lin_order[target],:] = (V[lin_order[target],:] + V[lin_order[control],:]) % 2

if np.array_equal(U,V):
	print('circuit test: OK')
else:
	print('circuit test: FAIL')
	exit()


# Compute decoding matrices

print('error rate=',error_rate)
print('Generating noisy circuits with a singe Z-type faulty operation...')
ProbZ = []
circuitsZ = []
head = []
tail = cycle_repeated.copy()
for gate in cycle_repeated:
	assert(gate[0] in ['CNOT','IDLE','PrepX','PrepZ','MeasX','MeasZ'])
	if gate[0]=='MeasX':
		assert(len(gate)==2)
		circuitsZ.append(head + [('Z',gate[1])] + tail)
		ProbZ.append(error_rate_meas)
	# move the gate from tail to head
	head.append(gate)
	tail.pop(0)
	assert(cycle_repeated==(head+tail))
	if gate[0]=='PrepX':
		assert(len(gate)==2)
		circuitsZ.append(head + [('Z',gate[1])] + tail)
		ProbZ.append(error_rate_init)
	if gate[0]=='IDLE':
		assert(len(gate)==2)
		circuitsZ.append(head + [('Z',gate[1])] + tail)
		ProbZ.append(error_rate_idle*2/3)
	if gate[0]=='CNOT':
		assert(len(gate)==3)
		# add error on the control qubit
		circuitsZ.append(head + [('Z',gate[1])] + tail)
		ProbZ.append(error_rate_cnot*4/15)
		# add error on the target qubit
		circuitsZ.append(head + [('Z',gate[2])] + tail)
		ProbZ.append(error_rate_cnot*4/15)
		# add ZZ error on the control and the target qubits
		circuitsZ.append(head + [('ZZ',gate[1],gate[2])] + tail)
		ProbZ.append(error_rate_cnot*4/15)
		
num_errZ=len(circuitsZ)
print('Number of noisy circuits=',num_errZ)
print('Done.')


print('Generating noisy circuits with a singe X-type faulty operation...')
ProbX = []
circuitsX = []
head = []
tail = cycle_repeated.copy()
for gate in cycle_repeated:
	assert(gate[0] in ['CNOT','IDLE','PrepX','PrepZ','MeasX','MeasZ'])
	if gate[0]=='MeasZ':
		assert(len(gate)==2)
		circuitsX.append(head + [('X',gate[1])] + tail)
		ProbX.append(error_rate_meas)
	# move the gate from tail to head
	head.append(gate)
	tail.pop(0)
	assert(cycle_repeated==(head+tail))
	if gate[0]=='PrepZ':
		assert(len(gate)==2)
		circuitsX.append(head + [('X',gate[1])] + tail)
		ProbX.append(error_rate_init)
	if gate[0]=='IDLE':
		assert(len(gate)==2)
		circuitsX.append(head + [('X',gate[1])] + tail)
		ProbX.append(error_rate_idle*2/3)
	if gate[0]=='CNOT':
		assert(len(gate)==3)
		# add error on the control qubit
		circuitsX.append(head + [('X',gate[1])] + tail)
		ProbX.append(error_rate_cnot*4/15)
		# add error on the target qubit
		circuitsX.append(head + [('X',gate[2])] + tail)
		ProbX.append(error_rate_cnot*4/15)
		# add XX error on the control and the target qubits
		circuitsX.append(head + [('XX',gate[1],gate[2])] + tail)
		ProbX.append(error_rate_cnot*4/15)
		
	
num_errX=len(circuitsX)
print('Number of noisy circuits=',num_errX)
print('Done.')



# we only look at the action of the circuit on Z errors; 0 means no error, 1 means error
def simulate_circuitZ(C):
	syndrome_history = []
	# keys = Xchecks, vals = list of positions in the syndrome history array
	syndrome_map = {}
	state = np.zeros(2*n,dtype=int)
	# need this for debugging
	err_cnt = 0
	syn_cnt = 0
	for gate in C:
		if gate[0]=='CNOT':
			assert(len(gate)==3)
			control = lin_order[gate[1]]
			target = lin_order[gate[2]]
			state[control] = (state[target] + state[control]) % 2
			continue
		if gate[0]=='PrepX':
			assert(len(gate)==2)
			q = lin_order[gate[1]]
			state[q]=0
			continue
		if gate[0]=='MeasX':
			assert(len(gate)==2)
			assert(gate[1][0]=='Xcheck')
			q = lin_order[gate[1]]
			syndrome_history.append(state[q])
			if gate[1] in syndrome_map:
				syndrome_map[gate[1]].append(syn_cnt)
			else:
				syndrome_map[gate[1]] = [syn_cnt]
			syn_cnt+=1
			continue
		if gate[0] in ['Z','Y']:
			err_cnt+=1
			assert(len(gate)==2)
			q = lin_order[gate[1]]
			state[q] = (state[q] + 1) % 2
			continue

		if gate[0] in ['ZX', 'YX']:
			err_cnt+=1
			assert(len(gate)==3)
			q = lin_order[gate[1]]
			state[q] = (state[q] + 1) % 2
			continue

		if gate[0] in ['XZ','XY']:
			err_cnt+=1
			assert(len(gate)==3)
			q = lin_order[gate[2]]
			state[q] = (state[q] + 1) % 2
			continue

		if gate[0] in ['ZZ','YY','YZ','ZY']:
			err_cnt+=1
			assert(len(gate)==3)
			q1 = lin_order[gate[1]]
			q2 = lin_order[gate[2]]
			state[q1] = (state[q1] + 1) % 2
			state[q2] = (state[q2] + 1) % 2
			continue
	return np.array(syndrome_history,dtype=int),state,syndrome_map,err_cnt


# we only look at the action of the circuit on X errors; 0 means no error, 1 means error
def simulate_circuitX(C):
	syndrome_history = []
	# keys = Zchecks, vals = list of positions in the syndrome history array
	syndrome_map = {}
	state = np.zeros(2*n,dtype=int)
	# need this for debugging
	err_cnt = 0
	syn_cnt = 0
	for gate in C:
		if gate[0]=='CNOT':
			assert(len(gate)==3)
			control = lin_order[gate[1]]
			target = lin_order[gate[2]]
			state[target] = (state[target] + state[control]) % 2
			continue
		if gate[0]=='PrepZ':
			assert(len(gate)==2)
			q = lin_order[gate[1]]
			state[q]=0
			continue
		if gate[0]=='MeasZ':
			assert(len(gate)==2)
			assert(gate[1][0]=='Zcheck')
			q = lin_order[gate[1]]
			syndrome_history.append(state[q])
			if gate[1] in syndrome_map:
				syndrome_map[gate[1]].append(syn_cnt)
			else:
				syndrome_map[gate[1]] = [syn_cnt]
			syn_cnt+=1
			continue
		if gate[0] in ['X','Y']:
			err_cnt+=1
			assert(len(gate)==2)
			q = lin_order[gate[1]]
			state[q] = (state[q] + 1) % 2
			continue

		if gate[0] in ['XZ', 'YZ']:
			err_cnt+=1
			assert(len(gate)==3)
			q = lin_order[gate[1]]
			state[q] = (state[q] + 1) % 2
			continue

		if gate[0] in ['ZX','ZY']:
			err_cnt+=1
			assert(len(gate)==3)
			q = lin_order[gate[2]]
			state[q] = (state[q] + 1) % 2
			continue

		if gate[0] in ['XX','YY','XY','YX']:
			err_cnt+=1
			assert(len(gate)==3)
			q1 = lin_order[gate[1]]
			q2 = lin_order[gate[2]]
			state[q1] = (state[q1] + 1) % 2
			state[q2] = (state[q2] + 1) % 2
			continue
	return np.array(syndrome_history,dtype=int),state,syndrome_map,err_cnt




HXdict  = {}

# execute each noisy circuit and compute the syndrome
# we add two noiseless syndrome cycles at the end
print('Computing syndrome histories for single-X-type-fault circuits...')
cnt = 0
for circ in circuitsX:
	syndrome_history,state,syndrome_map,err_cnt = simulate_circuitX(circ+cycle+cycle)
	assert(err_cnt==1)
	assert(len(syndrome_history)==n2*(num_cycles+2))
	state_data_qubits = [state[lin_order[q]] for q in data_qubits]
	syndrome_final_logical = (lz @ state_data_qubits) % 2
	# apply syndrome sparsification map
	syndrome_history_copy = syndrome_history.copy()
	for c in Zchecks:
		pos = syndrome_map[c]
		assert(len(pos)==(num_cycles+2))
		for row in range(1,num_cycles+2):
			syndrome_history[pos[row]]+= syndrome_history_copy[pos[row-1]]
	syndrome_history%= 2
	syndrome_history_augmented = np.hstack([syndrome_history,syndrome_final_logical])
	supp = tuple(np.nonzero(syndrome_history_augmented)[0])
	if supp in HXdict:
		HXdict[supp].append(cnt)
	else:
		HXdict[supp]=[cnt]
	cnt+=1
	

first_logical_rowX = n2*(num_cycles+2)
print('Done.')

# if a subset of columns of H are equal, retain only one of these columns
print('Computing effective noise model for the X-decoder...')

num_errX = len(HXdict)
print('Number of distinct X-syndrome histories=',num_errX)
HX = []
HdecX = []
channel_probsX = []
for supp in HXdict:
	new_column = np.zeros((n2*(num_cycles+2)+k,1),dtype=int)
	new_column_short = np.zeros((n2*(num_cycles+2),1),dtype=int)
	new_column[list(supp),0] = 1
	new_column_short[:,0] = new_column[0:first_logical_rowX,0]
	HX.append(coo_matrix(new_column))
	HdecX.append(coo_matrix(new_column_short))
	channel_probsX.append(np.sum([ProbX[i] for i in HXdict[supp]]))
print('Done.')
HX = hstack(HX)
HdecX = hstack(HdecX)

print('Decoding matrix HX sparseness:')
print('max col weight=',np.max(np.sum(HdecX,0)))
print('max row weight=',np.max(np.sum(HdecX,1)))


# execute each noisy circuit and compute the syndrome
# we add two noiseless syndrome cycles at the end

HZdict  = {}

print('Computing syndrome histories for single-Z-type-fault circuits...')
cnt = 0
for circ in circuitsZ:
	syndrome_history,state,syndrome_map,err_cnt = simulate_circuitZ(circ+cycle+cycle)
	assert(err_cnt==1)
	assert(len(syndrome_history)==n2*(num_cycles+2))
	state_data_qubits = [state[lin_order[q]] for q in data_qubits]
	syndrome_final_logical = (lx @ state_data_qubits) % 2
	# apply syndrome sparsification map
	syndrome_history_copy = syndrome_history.copy()
	for c in Xchecks:
		pos = syndrome_map[c]
		assert(len(pos)==(num_cycles+2))
		for row in range(1,num_cycles+2):
			syndrome_history[pos[row]]+= syndrome_history_copy[pos[row-1]]
	syndrome_history%= 2
	syndrome_history_augmented = np.hstack([syndrome_history,syndrome_final_logical])
	supp = tuple(np.nonzero(syndrome_history_augmented)[0])
	if supp in HZdict:
		HZdict[supp].append(cnt)
	else:
		HZdict[supp]=[cnt]
	cnt+=1


first_logical_rowZ = n2*(num_cycles+2)
print('Done.')

# if a subset of columns of HZ are equal, retain only one of these columns
print('Computing effective noise model for the Z-decoder...')
num_errZ = len(HZdict)
print('Number of distinct Z-syndrome histories=',num_errZ)
HZ = []
HdecZ = []
channel_probsZ = []
for supp in HZdict:
	new_column = np.zeros((n2*(num_cycles+2)+k,1),dtype=int)
	new_column_short = np.zeros((n2*(num_cycles+2),1),dtype=int)
	new_column[list(supp),0] = 1
	new_column_short[:,0] = new_column[0:first_logical_rowZ,0]
	HZ.append(coo_matrix(new_column))
	HdecZ.append(coo_matrix(new_column_short))
	channel_probsZ.append(np.sum([ProbZ[i] for i in HZdict[supp]]))
print('Done.')
HZ = hstack(HZ)
HdecZ = hstack(HdecZ)


print('Decoding matrix HZ sparseness:')
print('max col weight=',np.max(np.sum(HdecZ,0)))
print('max row weight=',np.max(np.sum(HdecZ,1)))


# save decoding matrices 

mydata = {}
mydata['HdecX']=HdecX
mydata['HdecZ']=HdecZ
mydata['probX']=channel_probsX
mydata['probZ']=channel_probsZ
mydata['cycle']=cycle
mydata['lin_order']=lin_order
mydata['num_cycles']=num_cycles
mydata['data_qubits']=data_qubits
mydata['Xchecks']=Xchecks
mydata['Zchecks']=Zchecks
mydata['HX']=HX
mydata['HZ']=HZ
mydata['lx']=lx
mydata['lz']=lz
mydata['first_logical_rowZ']=first_logical_rowZ
mydata['first_logical_rowX']=first_logical_rowX
mydata['ell']=ell
mydata['m']=m
mydata['a1']=a1
mydata['a2']=a2
mydata['a3']=a3
mydata['b1']=b1
mydata['b2']=b2
mydata['b3']=b3
mydata['error_rate']=error_rate
mydata['sX']=sX
mydata['sZ']=sZ

title='./TMP/mydata_' + str(n) + '_' + str(k) + '_p_' + str(error_rate) + '_cycles_' + str(num_cycles)


print('saving data to ',title)
with open(title, 'wb') as fp:
	pickle.dump(mydata, fp)

print('Done')


