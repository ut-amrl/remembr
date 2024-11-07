from remembr.memory.memory import MemoryItem
from remembr.memory.milvus_memory import MilvusMemory
from remembr.agents.remembr_agent import ReMEmbRAgent

import argparse
import json

def remember(args):
    memory = MilvusMemory("test_collection", db_ip='127.0.0.1')
    memory.reset()
    with open(args.data_path, 'r') as f:
        for entry in json.load(f):
            t, pos, caption = entry["time"], entry["base_position"] + [0.0], entry["base_caption"]
            memory_item = MemoryItem(
                caption=caption,
                time=t,
                position=pos,
                theta=0
            )
            memory.insert(memory_item)
    return memory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    memory = remember(args)

    agent = ReMEmbRAgent(llm_type='gpt-4o')
    agent.set_memory(memory)
    while True:
        user_input = input("Ask me a question (Enter 'q' to exit): ")
        if user_input.lower() == 'q':
            print("Exiting the program.")
            exit(0)
        response = agent.query(user_input)
        print(response.position)
        print(response.text)

