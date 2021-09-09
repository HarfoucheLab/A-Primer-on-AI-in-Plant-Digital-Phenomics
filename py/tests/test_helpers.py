import os
import sys
import torch
import numpy as np

sys.path.append(os.getcwd())
from helpers import *

def test_makedir():
	makedir('/content/unittest')
	assert os.path.isdir('/content/unittest')