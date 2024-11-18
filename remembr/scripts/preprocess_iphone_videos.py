import argparse
import re
from io import BytesIO
import os, os.path as osp

import requests
from PIL import Image
import numpy as np
import sys
import cv2

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

from tqdm import tqdm
import cv2

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
def run_video_in_segs(args):
    args.device_map = 'auto'
    vila_model = VILACaptioner(args)
    
    cap = cv2.VideoCapture(args.data_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {args.data_path}")
    
    images, captions = [], []
    for _ in range(1000000):
        ret, frame = cap.read()
        if not ret:
            break
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if len(images) >= 90:
            images = images[::30//args.num_video_frames]
            out_text = vila_model.caption(images)
            captions.append(out_text)
            images = [pil_image]
        else:
            images.append(pil_image)
            
    t = 1.5
    outputs = []
    for caption in captions:
        entiry = {
            "time": t,
            "base_position": [0.0,0.0,0.0], # dummy data
            "base_caption": caption,
            "base_caption_embedding": [],
            "wrist_caption": "",
            "wrist_caption_embedding": []
        }
    
if __name__ == "__main__":
    default_query = "<video>\n You are a wandering around a household kitchen/work area.\
        Please describe in detail what you see in the few seconds of the video. \
        Specifically focus on the objects, environmental features, events/ectivities, people (as well as their actions), and other interesting details. Think step by step about these details and be very specific."
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/Llama-3-VILA1.5-8B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--captioner_name", type=str, default="VILA1.5-8b")
    parser.add_argument("--seconds_per_caption", type=int, default=5)
    
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