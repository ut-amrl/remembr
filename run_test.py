from remembr.planners.remember_video_planner import ReMEmbRVideoPlanner
from remembr.memloaders.memloaders import *

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--uid", type=str, default="tabletop_simple")
    args = parser.parse_args()
    
    memory = None
    import time
    
    if args.uid == "tabletop_simple":
        inpaths = [
            "/home/tiejean/RobotMem/data/captions/iphones/taijing_table_1_VILA1.5-8b_3_secs.json",
            "/home/tiejean/RobotMem/data/captions/iphones/taijing_table_3_VILA1.5-8b_3_secs.json",
            "/home/tiejean/RobotMem/data/captions/iphones/taijing_table_4_VILA1.5-8b_3_secs.json"
        ]
        user_input = "Today is 2025-02-08. Where is the book that was on the table the day before yesterday?"
        t_offset = 1738952666.5530548-len(inpaths)*86400 + 86400; p_offset = 0
        for i, inpath in enumerate(inpaths):
            memory = remember_dummy_video(inpath, time_offset=t_offset, pos_offset=p_offset, reset=(i == 0))
            t_offset += 86400; p_offset += 1
    elif args.uid == "cobot1":
        inpaths = [
            "/home/tiejean/RobotMem/data/captions/cobot/2025-02-08-16-39-19_VILA1.5-8b_3_secs.json",
            "/home/tiejean/RobotMem/data/captions/cobot/2025-02-08-17-07-39_VILA1.5-8b_3_secs.json",
            # "/home/tiejean/RobotMem/data/captions/cobot/2025-02-08-16-49-07_VILA1.5-8b_3_secs.json",
            # "/home/tiejean/RobotMem/data/captions/cobot/2025-02-08-17-07-39_VILA1.5-8b_3_secs.json"
        ]
        user_input = "Today is 2025-02-07. Where is the bottle that was on a shelf yesterday?"
        t_offset = 1738952666.5530548-len(inpaths)*86400 + 86400; p_offset = 0
        for i, inpath in enumerate(inpaths):
            memory = remember_cobot_vid(inpath, time_offset=t_offset, reset=(i == 0))
            t_offset += 86400
    elif args.uid == "cobot2":
        inpaths = [
            "/home/tiejean/RobotMem/data/captions/cobot/2025-02-08-16-49-07_VILA1.5-8b_3_secs.json",
            "/home/tiejean/RobotMem/data/captions/cobot/2025-02-08-16-39-19_VILA1.5-8b_3_secs.json",
            "/home/tiejean/RobotMem/data/captions/cobot/2025-02-08-17-07-39_VILA1.5-8b_3_secs.json",
        ]
        user_input = "Today is 2025-02-07. Where is the bottle that I used to drink water the day before yesterday?"
        t_offset = 1738952666.5530548-len(inpaths)*86400 + 86400; p_offset = 0
        for i, inpath in enumerate(inpaths):
            memory = remember_cobot_vid(inpath, time_offset=t_offset, reset=(i == 0))
            t_offset += 86400
    # memory.search_by_datetime("2025-02-06 04:30:56")
    # memory.search_by_text("book")
    # import pdb; pdb.set_trace()
    agent = ReMEmbRVideoPlanner(llm_type='gpt-4o')
    agent.set_memory(memory)
    while True:
        # user_input = input("\nGive me a task (Enter 'q' to exit): ")
        
        if user_input.lower() == 'q':
            print("Exiting the program.")
            exit(0)
        agent.query(user_input)
    # Today is 2025-02-08. Where is the book that was on the table yesterday?
    # Today is 2025-02-07. Where is the book that was on the table yesterday?