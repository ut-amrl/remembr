from remembr.memory.memory import MemoryItem
from remembr.memory.milvus_memory import MilvusMemory
from remembr.agents.remembr_agent import ReMEmbRAgent

memory = MilvusMemory("test_collection", db_ip='127.0.0.1')
memory.reset()

t = 1.0
dt = 0.1
items = ["apple(s)", "banana(s)", "cherry(s)", "orange(s)"]
Nt = 50; Nitems = len(items)
cabinets = [([i, 0.0, 0.0], 3.14, items[i]) for i in range(Nitems)]

import random
for i in range(Nt):
    iCabinet = random.randint(0, Nitems-1)
    pos, theta, item = cabinets[iCabinet]
    noise = [random.randint(0, 2) for _ in range(Nitems)]
    noise = [f", {i} {items[i]}"  for i in noise if i != 0]
    caption = f"I see a cabinet with {random.randint(3, 8)} {items[iCabinet]}"
    caption = caption.join(noise)
    memory_item = MemoryItem(
        caption=caption,
        time=t,
        position=pos,
        theta=theta
    )
    memory.insert(memory_item)
    t += dt


# agent = ReMEmbRAgent(llm_type='command-r')
agent = ReMEmbRAgent(llm_type='gpt-4o')
agent.set_memory(memory)

response = agent.query("I have a bag of cherries. Where should I put it?")
print(response.position)
print(response.text)