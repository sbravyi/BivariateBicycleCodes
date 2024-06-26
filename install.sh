python3.10 -m venv venv
source venv/bin/activate

pip install cython==0.29.37
pip install wheels
pip install numpy==1.22.0
pip install scipy==1.8.0
pip install git+https://github.com/quantumgizmos/ldpc.git --no-build-isolation
pip install git+https://github.com/quantumgizmos/bp_osd.git