import argparse
import re
from io import BytesIO
import os, os.path as osp

import requests
from PIL import Image
import numpy as np
import sys

# load this directory
sys.path.append(os.getcwd())
from remembr.captioners.vila_captioner import VILACaptioner
from remembr.utils.util import get_frames
import pickle as pkl
from PIL import Image as PILImage

from langchain_huggingface import HuggingFaceEmbeddings
import glob
from scipy.spatial.transform import Rotation
import shutil
import json

import tqdm

# ROS
import rosbag
from cv_bridge import CvBridge

def run_video_in_segs(args):
    bridge = CvBridge()
    bagfile = os.path.join(args.data_path, f"{args.bagname}.bag")

    embedder = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
    vila_model = VILACaptioner(args)

    with rosbag.Bag(bagfile, 'r') as bag:
        pass

if __name__ == "__main__":
    default_query = "<video>\n You are a wandering around a household kitchen/work area.\
        Please describe in detail what you see in the few seconds of the video. \
        Specifically focus on the people, objects, environmental features, events/ectivities, and other interesting details. Think step by step about these details and be very specific."
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/VILA1.5-13b")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--bagname", type=str, default="")
    parser.add_argument("--data_path", type=str, default="./coda_data")
    parser.add_argument("--out_path", type=str, default="./data/captions")
    parser.add_argument("--captioner_name", type=str, default="VILA1.5-13b")

    parser.add_argument("--seconds_per_caption", type=int, default=3)

    parser.add_argument("--video-file", type=str, default=None)
    parser.add_argument("--num-video-frames", type=int, default=6)
    parser.add_argument("--query", type=str, default=default_query)
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()

    # add some rules here
    if 'Efficient-Large-Model/VILA1.5-40b' in args.model_path:
        args.conv_mode = 'hermes-2'
    elif 'Efficient-Large-Model/VILA1.5' in args.model_path:
        args.conv_mode = 'vicuna_v1'
    elif 'Llama' in args.model_path:
        args.conv_mode = 'llama_3'
    else:
        # trust the default conv_mode
        args.conv_mode = args.conv_mode

    run_video_in_segs(args)