from remembr.memory.memory import MemoryItem
from remembr.memory.milvus_memory import MilvusMemory
from remembr.planners.remember_planner import ReMEmbRPlanner
from remembr.agents.remembr_agent import ReMEmbRAgent

def remember():
    memory = MilvusMemory("test_collection", db_ip='127.0.0.1')
    memory.reset()

    t = 1.0
    dt = 0.1
    pos = [0.0, 0.0 ,0.0]
    items = ["apple in refridge", 
             "orange in refridge", 
             "milk in refridge", 
             "banana on tablecounter", 
             "cherry in refridge", 
             "cereal in a cabinet", 
             "knife",
             "someone is drinking milk",
             "milk box in trashcan",
             "bowls on table"]

    for item in items:
        caption = f"I see {item}"
        pos[0] += 1.0
        memory_item = MemoryItem(
            caption=caption,
            time=t,
            position=pos,
            theta=0
        )
        memory.insert(memory_item)
        t += dt
    return memory


if __name__ == "__main__":
    memory = remember()
    agent = ReMEmbRPlanner(llm_type='gpt-4o')
    # agent = ReMEmbRAgent(llm_type='gpt-4o')
    agent.set_memory(memory)
    while True:
        user_input = input("Ask me a question (Enter 'q' to exit): ")
        if user_input.lower() == 'q':
            print("Exiting the program.")
            exit(0)
        response = agent.query(user_input)
        print(response)
        # print(response.positions)
        # print(response.plans)
        # print(response.text)
        # print(response.question)