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

from tqdm import tqdm
import cv2

# ROS
import rosbag
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def run_video_in_segs(args):
    bridge = CvBridge()
    bagfile = os.path.join(args.data_path, f"{args.bagname}.bag")

    embedder = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')
    args.device_map = 'auto'
    vila_model = VILACaptioner(args)

    with rosbag.Bag(bagfile, 'r') as bag:
        n_base_images = bag.get_message_count(topic_filters=['/camera2/color/image_raw/compressed'])
        n_wrist_images = bag.get_message_count(topic_filters=['/camera/color/image_raw/compressed'])

        base_positions = []
        base_captions, wrist_captions = [], []
        segments = []

        # caption base images
        start_time = None
        images = []
        for topic, msg, t in tqdm(bag.read_messages(topics=["/camera2/color/image_raw/compressed"]), total=n_base_images, desc="Captioning base images"):
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            images.append(pil_image)
            if start_time is None:
                start_time = t.to_sec()
            elif t.to_sec() - start_time > args.seconds_per_caption:
                images = images[::30//args.num_video_frames]
                out_text = vila_model.caption(images)
                # store time window & captions for base camera
                segments.append((start_time, t.to_sec()))
                base_captions.append(out_text)
                start_time = t.to_sec()
                images = [pil_image]

        # process base localization
        sid = 0
        positions = []
        for topic, msg, t in bag.read_messages(topics=["/localization"]):
            start_t, end_t = segments[sid]
            if t.to_sec() < start_t:
                continue
            elif t.to_sec() > end_t:
                pos = np.mean(positions, axis=0)
                base_positions.append(pos)
                sid += 1
                positions = [(msg.pose.x, msg.pose.y)]
            else:
                positions.append((msg.pose.x, msg.pose.y))
            if sid >= len(segments):
                break
            
        # caption wrist images
        sid = 0
        images = []
        for topic, msg, t in tqdm(bag.read_messages(topics=["/camera/color/image_raw/compressed"]), total=n_wrist_images, desc="Captioning wrist images"):
            start_t, end_t = segments[sid]
            if t.to_sec() < start_t:
                continue
            np_arr = np.frombuffer(msg.data, np.uint8)
            cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            pil_image = Image.fromarray(cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB))
            if t.to_sec() > end_t:
                images = images[::30//args.num_video_frames]
                out_text = vila_model.caption(images)
                wrist_captions.append(out_text)
                sid += 1
                images = [pil_image]
            else:
                images.append(pil_image)
            if sid >= len(segments):
                break

        
        outputs = []
        for base_caption, wrist_caption, base_position, (start_t, end_t) in zip(base_captions, wrist_captions, base_positions, segments):
            base_caption_embedding = embedder.embed_query(base_caption)
            wrist_caption_embedding = embedder.embed_query(wrist_caption)
            entity = {
                "time": np.mean([start_t, end_t]),
                "base_position": base_position,
                "base_caption": base_caption,
                "base_caption_embedding": base_caption_embedding,
                "wrist_caption": wrist_caption,
                "wrist_caption_embedding": wrist_caption_embedding
            }
            outputs.append(entity)

        os.makedirs(args.out_path, exist_ok=True)
        captions_location = os.path.join(args.out_path, args.bagname)
        os.makedirs(captions_location, exist_ok=True)
        filepath = os.path.join(captions_location, f'captions_{args.captioner_name}_{args.seconds_per_caption}_secs.json')
        with open(filepath, 'w') as f:
            print(f"Writing data to {filepath}")
            json.dump(outputs, f, cls=NumpyEncoder)


if __name__ == "__main__":
    default_query = "<video>\n You are a wandering around a household kitchen/work area.\
        Please describe in detail what you see in the few seconds of the video. \
        Specifically focus on the people, objects, environmental features, events/ectivities, and other interesting details. Think step by step about these details and be very specific."
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/Llama-3-VILA1.5-8B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--bagname", type=str, default="")
    parser.add_argument("--data_path", type=str, default="./coda_data")
    parser.add_argument("--out_path", type=str, default="./data/captions")
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
    
    # python remembr/scripts/preprocess_cobot_captions.py --seconds_per_caption 5 --data_path /robodata/taijing/RobotMem/data/bags/ --out_path /robodata/taijing/RobotMem/data/captions/ --bagname 2024-11-06-16-12-12