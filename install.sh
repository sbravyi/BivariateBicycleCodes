# This script assumes Python3.10 or later (replace if needed)
# The script is creating a virtualenv and installing the packages needed to get the ldpcv1 library working

python3.10 -m venv venvs
source venv/bin/activate

pip install cython==0.29.37
pip install wheels
pip install numpy==1.22.0
pip install scipy==1.8.0
pip install git+https://github.com/quantumgizmos/ldpc.git --no-build-isolation
pip install git+https://github.com/quantumgizmos/bp_osd.git