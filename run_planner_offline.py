from remembr.memory.memory import MemoryItem
from remembr.memory.milvus_memory import MilvusMemory
from remembr.planners.remember_planner import ReMEmbRPlanner
from remembr.memloaders.memloaders import *

if __name__ == "__main__":
    # memory = remember_demo()
    memory = remember_cobot(inpath="/home/tiejean/Workspace/RobotMem/data/captions/iphones/taijing_kitchen_VILA1.5-8b_5_secs.json")
    agent = ReMEmbRPlanner(llm_type='gpt-4o')
    # agent = ReMEmbRAgent(llm_type='gpt-4o')
    agent.set_memory(memory)
    while True:
        user_input = input("\nGive me a task (Enter 'q' to exit): ")
        if user_input.lower() == 'q':
            print("Exiting the program.")
            exit(0)
        response = agent.query(user_input)
        response.print_plan(show_reasons=True)