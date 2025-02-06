from remembr.memory.memory import MemoryItem
from remembr.memory.milvus_memory import MilvusMemory
from remembr.planners.remember_video_planner import ReMEmbRVideoPlanner
from remembr.memloaders.memloaders import *

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--uid", type=str, default="tabletop_simple")
    args = parser.parse_args()
    
    memory = None
    import time
    t_offset = 1738816277.9279537; p_offset = 0
    if args.uid == "tabletop_simple":
        inpaths = [
            "/home/tiejean/RobotMem/data/captions/iphones/taijing_table_1_VILA1.5-8b_3_secs.json",
            "/home/tiejean/RobotMem/data/captions/iphones/taijing_table_3_VILA1.5-8b_3_secs.json",
            "/home/tiejean/RobotMem/data/captions/iphones/taijing_table_4_VILA1.5-8b_3_secs.json"
        ]
        for i, inpath in enumerate(inpaths):
            memory = remember_dummy_video(inpath, time_offset=t_offset, pos_offset=p_offset, reset=(i == 0))
            t_offset += 86400; p_offset += 1
    # memory.search_by_datetime("2025-02-06 04:30:56")
    # memory.search_by_text("book")
    # import pdb; pdb.set_trace()
    agent = ReMEmbRVideoPlanner(llm_type='gpt-4o')
    agent.set_memory(memory)
    while True:
        user_input = input("\nGive me a task (Enter 'q' to exit): ")
        if user_input.lower() == 'q':
            print("Exiting the program.")
            exit(0)
        agent.query(user_input)
    # Today is 2025-02-08. Where is the book that was on the table yesterday?
    # Today is 2025-02-07. Where is the book that was on the table yesterday?