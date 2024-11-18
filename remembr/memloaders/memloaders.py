from typing import List
import json

import sys, os
sys.path.append(os.getcwd())
from remembr.memory.memory import MemoryItem
from remembr.memory.milvus_memory import MilvusMemory

def remember_cobot(inpath: str) -> MilvusMemory:
    memory = MilvusMemory("main_memory", db_ip='127.0.0.1')
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
    memory = MilvusMemory("main_memory", db_ip='127.0.0.1')
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