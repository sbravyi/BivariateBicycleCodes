Simulation software used to generate data reported on Figure 3 of [BCGMRY]

The simulation software consists of two python scripts:

decoder_setup.py is the offline part of the decoder that constructs check matrices,
syndrome measurement circuits, and decoding matrices for a particular quantum code.
This computation can take a few minutes per code. All code data is saved to disk.
One has to call decoder_setup.py only once for each combination (code, error rate, number of syndrome cycles). 

decoder_run.py is the online part of the decoder that simulates error correction circuits. It relies on the software implementation of the Belief Propagation with the Ordered Statistics Decoder due to 
Joschka Roffe
"LDPC: Python tools for low density parity check codes"
https://pypi.org/project/ldpc/

File naming: the working directory that contains  decoder_setup.py and decoder_run.py must contain folders "TMP" and "CODE_n_k_d" for each code [[n,k,d]] to be simulated. Initially these folders are empty. Folder "TMP" stores code data files with
check matrices, syndrome measurement circuits, and decoding matrices. There is a separate data file for each combination (code, error rate, number of syndrome cycles). Create code data files using decoder_setup.py. Folder "CODE_n_k_d" contains a file "result" that stores the simulation results. Each line in the "result" file has four columns:
column 1: physical error rate,
column 2: number of syndrome cycles,
column 3: number of Monte Carlo trials,
column 4: number of failed trials that resulted in a logical error.
Each trial runs the noisy error correction circuit followed by a noiseless syndrome measurement of all stabilizers, decoding, and error correction. A trial is failed if error correction results in a non-identity logical Pauli error. Create "result" files using decoder_run.py

distance_test.py calculates the code distance by solving an integer linear program


[BCGMRY]
Sergey Bravyi, Andrew Cross, Jay Gambetta, Dmitri Maslov, Patrick Rall, Theodore Yoder,
High-threshold and low-overhead fault-tolerant quantum memory
https://arxiv.org/abs/2308.07915
