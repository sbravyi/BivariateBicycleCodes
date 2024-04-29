# compute an upper bound on the circuit-level distance for BB codes

import numpy as np
from ldpc import bposd_decoder
from bposd.css import css_code
import pickle

# number of Monte Carlo trials
num_trials = 100000

# error rate 
p = 0.003


# code parameters
n = 144
k = 12
d = 12
num_cycles = 1


# load code parameters and decoding matrices
title = './TMP/mydata_' + str(n) + '_' + str(k) + '_p_' + str(p) + '_cycles_' + str(num_cycles)
with open(title, 'rb') as fp:
	mydata = pickle.load(fp)


HdecX = mydata['HdecX']
HdecZ = mydata['HdecZ']
assert(mydata['num_cycles']==num_cycles)
HX = mydata['HX']
HZ = mydata['HZ']
first_logical_rowZ=mydata['first_logical_rowZ']
first_logical_rowX=mydata['first_logical_rowX']
ell=mydata['ell']
m=mydata['m']
a1=mydata['a1']
a2=mydata['a2']
a3=mydata['a3']
b1=mydata['b1']
b2=mydata['b2']
b3=mydata['b3']
assert(p==mydata['error_rate'])
sX=mydata['sX']
sZ=mydata['sZ']

# setup BP-OSD decoder parameters
my_bp_method = "ms"
my_max_iter = 10000
my_osd_method = "osd_cs"
my_osd_order = 20
my_ms_scaling_factor = 0

# code length
n = 2*m*ell

n2 = m*ell


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


A = (x[a1] + y[a2] + y[a3]) % 2
B = (y[b1] + x[b2] + x[b3]) % 2
AT = np.transpose(A)
BT = np.transpose(B)
hx = np.hstack((A,B))
hz = np.hstack((BT,AT))


qcode=css_code(hx,hz)
lz = qcode.lz
lx = qcode.lx

HZ = HZ.todense()
HX = HX.todense()
HdecZ = HdecZ.todense()
HdecX = HdecX.todense()

# stores the minimum weight of logical X and Z operators
wminX = HdecX.shape[1]
wminZ = HdecZ.shape[1]

for trial in range(num_trials):


	ec_resultZ = 0
	ec_resultX = 0
	
	# correct Z errors
	random_vector = np.random.randint(2,size=HZ.shape[0]) 
	random_logical_op = (random_vector @ HZ) % 2
	random_logical_op = np.reshape(random_logical_op,(1,HZ.shape[1]))
	HZ1 = np.vstack((HdecZ,random_logical_op))
	syndrome = np.zeros(HZ1.shape[0],dtype=int)
	syndrome[-1]=1

	bpdZ=bposd_decoder(
   		HZ1,
    	error_rate=0.01,
	    max_iter=my_max_iter, #the maximum number of iterations for BP)
	    bp_method=my_bp_method,
	    ms_scaling_factor=my_ms_scaling_factor, #min sum scaling factor. If set to zero the variable scaling factor method is used
	    osd_method="osd_cs", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
	    osd_order=my_osd_order #the osd search depth
	    )
	bpdZ.decode(syndrome)
	low_weight_logical = bpdZ.osdw_decoding
	wt = np.count_nonzero(low_weight_logical)
	if wt<wminZ and wt>0:
		wminZ = wt
	print('Logical Z weight =',wt,'minimum Z weight =',wminZ)

		
	# correct X errors 
	random_vector = np.random.randint(2,size=HX.shape[0]) 
	random_logical_op = (random_vector @ HX) % 2
	random_logical_op = np.reshape(random_logical_op,(1,HX.shape[1]))
	HX1 = np.vstack((HdecX,random_logical_op))
	syndrome = np.zeros(HX1.shape[0],dtype=int)
	syndrome[-1]=1

	bpdX=bposd_decoder(
   		HX1,
    	error_rate=0.01,
	    max_iter=my_max_iter, #the maximum number of iterations for BP)
	    bp_method=my_bp_method,
	    ms_scaling_factor=my_ms_scaling_factor, #min sum scaling factor. If set to zero the variable scaling factor method is used
	    osd_method="osd_cs", #the OSD method. Choose from:  1) "osd_e", "osd_cs", "osd0"
	    osd_order=my_osd_order #the osd search depth
	    )
	bpdX.decode(syndrome)
	low_weight_logical = bpdX.osdw_decoding
	wt = np.count_nonzero(low_weight_logical)
	if wt<wminX and wt>0:
		wminX = wt
	print('Logical X weight =',wt,'minimum X weight =',wminX)

	
		