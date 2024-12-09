"""adding these lines allows you to run experiments on different windows
machines without having to install the library or to change the code to adjust for local folder structure
"""

import os
import sys
import argparse

# Set up the root directory
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
sys.path.append(root_dir)

# now you can import src and other packages.