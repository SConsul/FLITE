# Inputs
import torch
import numpy as np
from heuristics import *


# Test bounding box heuristic
def test_bbox_heuristic():
    # Define arguments
    test_paths = [['../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00401.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00411.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00421.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00431.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00441.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00451.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00461.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00471.jpg'],
 ['../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00481.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00491.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00501.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00511.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00521.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00531.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00541.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00551.jpg'],
 ['../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00401.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00411.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00421.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00431.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00441.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00451.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00461.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00471.jpg'],
 ['../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00001.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00011.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00021.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00031.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00041.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00051.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00061.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00071.jpg'],
 ['../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00561.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00571.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00581.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00591.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00591.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00591.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00591.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00591.jpg'],
 ['../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00241.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00251.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00261.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00271.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00281.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00291.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00301.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00311.jpg'],
 ['../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00001.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00011.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00021.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00031.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00041.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00051.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00061.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00071.jpg'],
 ['../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00081.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00091.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00101.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00111.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00121.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00131.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00141.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00151.jpg'],
 ['../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00481.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00491.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00501.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00511.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00521.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00531.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00541.jpg',
  '../dataset/orbit_benchmark_224/train/P665/wallet/clutter/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q/P665--wallet--clutter--6sN1XsQ76GOCaJOabx-5M3zl4qtDQCp3nryHcTW821Q-00551.jpg'],
 ['../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00081.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00091.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00101.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00111.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00121.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00131.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00141.jpg',
  '../dataset/orbit_benchmark_224/train/P665/door/clutter/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc/P665--door--clutter--VAVn7HMPfNinMzRJh301zoqxhbVGDrBbcK9mXjs56gc-00151.jpg'],
 ['../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00561.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00571.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00581.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00591.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00591.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00591.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00591.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00591.jpg'],
 ['../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00481.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00491.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00501.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00511.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00521.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00531.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00541.jpg',
  '../dataset/orbit_benchmark_224/train/P665/bed/clutter/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0/P665--bed--clutter--R8xCSOIpEhfXO5NsSHOPdaYcm0GM_0P4rwImD2WAmP0-00551.jpg']]
    test_paths = np.array(test_paths)
    test_bbox_path = '../dataset/orbit_clutter_bounding_boxes'
    # Compute heuristic
    bbox_filter = BBox(test_paths, test_bbox_path)
    bboxes = bbox_filter.get_batch_bbox()
    ranked_idxs = bbox_filter.get_ranked_bbox_sizes()
    print('BBoxes shape:', bboxes.shape)
    print('Ranked idxs:', ranked_idxs)


# Test blur heuristic
def test_blur_heuristic():
    # Define arguments
    random_tensor = torch.rand((44,4,3,224,224))
    # Compute heuristic
    blur_filter = Blur(random_tensor)
    top_k_idxs = blur_filter.get_least_blurry(4)
    print(top_k_idxs)


if __name__ == '__main__':
    test_bbox_heuristic()
