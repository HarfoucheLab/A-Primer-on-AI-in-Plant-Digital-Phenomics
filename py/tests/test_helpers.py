import os
import sys
sys.path.append(os.getcwd())
from helpers import *

def test_makedir():
	makedir('/content/unittest')
	assert os.path.isdir('/content/unittest')