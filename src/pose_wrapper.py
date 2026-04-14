import sys
import os
from .config import OPENPOSE_PATH

# Setup OpenPose System Paths
sys.path.append(os.path.join(OPENPOSE_PATH, 'build', 'python', 'openpose', 'Release'))
os.environ['PATH'] = os.environ['PATH'] + ';' + os.path.join(OPENPOSE_PATH, 'build', 'x64', 'Release') + ';' + os.path.join(OPENPOSE_PATH, 'bin')

try:
    import pyopenpose as op
except ImportError:
    print(f"Error: Could not import pyopenpose from {OPENPOSE_PATH}")
    raise

def start_openpose(params):
    opWrapper = op.WrapperPython()
    opWrapper.configure(params)
    opWrapper.start()
    return opWrapper