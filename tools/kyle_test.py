from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import os

import cv2
import torch
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker
from pysot.utils.bbox import get_axis_aligned_bbox
from pysot.utils.model_load import load_pretrain
from toolkit.datasets import DatasetFactory
from toolkit.utils.region import vot_overlap, vot_float2str


# Set Paths
curr_file_base_path = os.getcwd()
experiments_base_path = os.path.join(os.path.dirname(curr_file_base_path), "experiments")

model_tracker_names_dict = {
    # Alexnet-based
    "siamrpn_alex_dwxcorr_otb": {
        "basepath": os.path.join(experiments_base_path, "siamrpn_alex_dwxcorr_otb"),
        "config": os.path.join(experiments_base_path, "siamrpn_alex_dwxcorr_otb", "config.yaml"),
        "snapshot": os.path.join(experiments_base_path, "siamrpn_alex_dwxcorr_otb", "model.pth"),
        "trackers": [
            "siamrpn_tracker",
            "siamrpn_tracker_kyle",
        ],
    },

    # Resnet50-based
    "siamrpn_r50_l234_dwxcorr_otb": {
        "basepath": os.path.join(experiments_base_path, "siamrpn_r50_l234_dwxcorr_otb"),
        "config": os.path.join(experiments_base_path, "siamrpn_r50_l234_dwxcorr_otb", "config.yaml"),
        "snapshot": os.path.join(experiments_base_path, "siamrpn_r50_l234_dwxcorr_otb", "model.pth"),
        "trackers": [
            "siamrpn_tracker",
            "siamrpn_tracker_kyle",
        ],
    },

    # Resnet-34, Resnet-20, etc. (add later on)
}

dataset_name = "OTB100"
is_visualization = True

















































