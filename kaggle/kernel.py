"""Kaggle GPU kernel: runs train.py from the vietvec-code dataset on the vietvec-data dataset."""

import glob
import os
import subprocess
import sys

code_dir = os.path.dirname(glob.glob("/kaggle/input/**/train.py", recursive=True)[0])
data_dir = os.path.dirname(glob.glob("/kaggle/input/**/corpus_ids.npz", recursive=True)[0])
subprocess.run(
    [sys.executable, os.path.join(code_dir, "train.py"), "--data_dir", data_dir, "--out_dir", "/kaggle/working"],
    check=True,
)
