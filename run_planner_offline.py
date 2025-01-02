from remembr.memory.memory import MemoryItem
from remembr.memory.milvus_memory import MilvusMemory
from remembr.planners.remember_planner import ReMEmbRPlanner
from remembr.memloaders.memloaders import *

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None)
    args = parser.parse_args()
    if args.data_path is None:
        # memory = remember_demo(observations=["The video starts with a shot of a kitchen counter with various cooking ingredients and utensils scattered around. The person's hand is seen reaching for a bottle of olive oil and a jar of spices. The camera then pans to a white cabinet where the person takes out a frying pan and places it on the stove. The person then opens the refrigerator and takes out a bag of cauliflower. The person places the cauliflower on the frying pan and begins to cook it. The camera focuses on the cauliflower as it cooks, with the person occasionally stirring it. The video ends with the person taking the cooked cauliflower out of the frying pan."])
        memory = remember_demo()
        # memory = remember_demo(observations=["take out a frying pan from a white cabinet", "someone bring a califlower from refridge"])
    else:
        memory = remember_cobot(inpath=args.data_path)
    agent = ReMEmbRPlanner(llm_type='gpt-4o')
    agent.set_memory(memory)
    while True:
        user_input = input("\nGive me a task (Enter 'q' to exit): ")
        if user_input.lower() == 'q':
            print("Exiting the program.")
            exit(0)
        agent.query(user_input)
        # response = agent.query(user_input)
        # response.print_plan(show_reasons=True)