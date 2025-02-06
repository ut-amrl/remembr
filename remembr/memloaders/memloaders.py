from typing import List
import json

import sys, os
sys.path.append(os.getcwd())
from remembr.memory.memory import MemoryItem
from remembr.memory.milvus_memory import MilvusMemory

MAIN_MEM = "main_memory"

def remember_caption_once(inpath: str) -> MilvusMemory:
    default_query = "<video>\n You are witnessing an event from a human perspective in a household area. The human is likely doing household chores.\
        Please describe in detail what you see in the video. \
        Specifically focus on the objects and human activities (e.g. their interactions with objects, their hand movements, etc.) \
        importantly, you should pay attention to event and action seqeunces in details \
        Think step by step about these details and be very specific."
    
    import cv2
    from PIL import Image
    from remembr.captioners.vila_captioner import VILACaptioner
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="Efficient-Large-Model/Llama-3-VILA1.5-8B")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--captioner_name", type=str, default="VILA1.5-8b")
    parser.add_argument("--seconds_per_caption", type=int, default=3)
    parser.add_argument("--num-video-frames", type=int, default=5)
    parser.add_argument("--query", type=str, default=default_query)
    parser.add_argument("--conv-mode", type=str, default="llama_3")
    parser.add_argument("--sep", type=str, default=",")
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    args = parser.parse_args()
    
    args.device_map = 'auto'
    vila_model = VILACaptioner(args)
    
    cap = cv2.VideoCapture(inpath)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video file: {inpath}")
    images = []
    for _ in range(10000000):
        ret, frame = cap.read()
        if not ret:
            break
        pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        images.append(pil_image)
    images = images[::60]
    out_text = vila_model.caption(images)
    print(out_text)
    
def remember_dummy_video(inpath: str, time_offset: float = 0, pos_offset: float = 0, reset: bool = True) -> MilvusMemory:
    memory = MilvusMemory(MAIN_MEM, db_ip='127.0.0.1')
    if reset:
        memory.reset()
    pos_offset = 0
    with open(inpath, 'r') as f:
        for entry in json.load(f):
            t, pos, caption, start_frame, end_frame = entry["time"], entry["base_position"], entry["base_caption"], entry["start_frame"], entry["end_frame"]
            # If pos only contains (x,y), append a dummy 0.0 as the z coordinate
            from pathlib import Path
            # TODO
            vidpath = "/home/tiejean/RobotMem/data/images/"
            if "iphones" in inpath:
                vidpath = os.path.join(vidpath, "iphones")
            vidpath = os.path.join(vidpath, Path(inpath).stem)
            if len(pos) == 2:
                pos += [0.0]
            pos[0] += pos_offset # TODO DELETE ME
            
            from time import strftime, localtime
            time = t+time_offset
            timestr = strftime('%Y-%m-%d %H:%M:%S', localtime(time))
            capstr = f"At time={timestr}, the robot was at an average position of {pos}. "
            capstr += f"The robot saw the following: {caption}"
            memory_item = MemoryItem(
                caption=capstr,
                time=time,
                position=pos,
                theta=0,
                vidpath=vidpath,
                start_frame=start_frame,
                end_frame=end_frame
            )
            memory.insert(memory_item)
            pos_offset += 1
    return memory

def remember_cobot(inpath: str) -> MilvusMemory:
    memory = MilvusMemory(MAIN_MEM, db_ip='127.0.0.1')
    memory.reset()
    with open(inpath, 'r') as f:
        for entry in json.load(f):
            t, pos, caption = entry["time"], entry["base_position"], entry["base_caption"]
            # If pos only contains (x,y), append a dummy 0.0 as the z coordinate
            if len(pos) == 2:
                pos += [0.0]
            memory_item = MemoryItem(
                caption=caption,
                time=t,
                position=pos,
                theta=0
            )
            memory.insert(memory_item)
    return memory

def remember_demo(observations: List[str] = [], positions: list = None)-> MilvusMemory:
    memory = MilvusMemory(MAIN_MEM, db_ip='127.0.0.1')
    memory.reset()
    
    if len(observations) == 0:
        observations = ["apple in refridge", 
             "orange in refridge", 
             "milk in refridge", 
             "banana on tablecounter", 
             "cherry in refridge", 
             "someone opened the cabinet and there's cereal in it",
             "knife",
             "someone is drinking milk",
             "milk box in trashcan",
             "someone opened the drawer and take out a staple and some paper",
             "bowls on table"]
    
    if positions is None:
        positions = []
        pos = [0.0, 0.0 ,0.0]
        for _ in observations:
            positions.append(pos)
            pos[0] += 0.1
            
    assert len(observations) == len(positions)
            
    t = 1.0; dt = 0.1
    for obs, pos in zip(observations, positions):
        memory_item = MemoryItem(
            caption=obs,
            time=t,
            position=pos,
            theta=0
        )
        memory.insert(memory_item)
        t += dt
    return memory