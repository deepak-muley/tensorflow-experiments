import numpy as np
import os
import os.path
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile
import pathlib
import cv2
import argparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

from tf_object_detection_common import *

if __name__ == "__main__":

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", type=str,
        help="path to optional input video file")
    args = vars(ap.parse_args())

    # if a video path was not supplied, grab a reference to the webcam
    if not args.get("input", False):
        print("[INFO] starting video stream...")
        cap = cv2.VideoCapture(0)

    # otherwise, grab a reference to the video file
    else:
        print("[INFO] opening video file...")
        cap = cv2.VideoCapture(args["input"])
        
    start_detection(cap)