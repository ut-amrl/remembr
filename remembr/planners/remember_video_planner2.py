from typing import Annotated, Literal, Sequence, TypedDict
import traceback
import sys, re

# from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

from langchain_openai import ChatOpenAI
from langchain_community.chat_models import ChatOllama
from langchain_nvidia_ai_endpoints import ChatNVIDIA

from langchain_core.prompts import PromptTemplate
from langchain.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder
)
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.utils.function_calling import convert_to_openai_function

from langchain.tools import StructuredTool
from langchain_core.pydantic_v1 import BaseModel, Field
import cv2
import json
import sys, os
sys.path.append(os.getcwd())
import copy

from remembr.utils.util import file_to_string
from remembr.tools.tools import *
from remembr.tools.functions_wrapper import FunctionsWrapper

from remembr.memory.milvus_video_memory import MilvusVideoMemory

from remembr.planners.planner import Planner, PlannerOutput

class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [1] * n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        if root_x != root_y:
            if self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            elif self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1

### Print out state of the system
def inspect(state):
    """Print the state passed between Runnables in a langchain and pass it on"""
    for k,v in state.items():
        if type(v) == str:
            print(v)

        elif type(v) == list:
            for item in v:
                if type(item) == str:
                    print(item)
                else:
                    print(item)
        else:
            print(item)

    # print(state)
    return state


def parse_json(string):
    parsed = re.search(r"```json(.*?)```", string, re.DOTALL| re.IGNORECASE).group(1).strip()
    return eval(parsed)

def parse_response(response: str, special_keys: list = [], messages = None):
    # let us parse and check the output is a dictionary. raise error otherwise
    response = ''.join(response.content.splitlines())
    
    try:
        if '```json' not in response:
            # try parsing on its own since we cannot always trust llms
            parsed = eval(response) 
        else:
            parsed = parse_json(response)
            
        for key in special_keys:
            if key not in parsed.keys():
                 raise ValueError("Generate call failed. Retrying...")
            if type(parsed[key]) == str:
                parsed[key] = eval(parsed[key])

    except:
        import pdb; pdb.set_trace()
        raise ValueError("Generate call failed. Retrying...")
    return parsed

def replace_messages(current: Sequence, new: Sequence):
    """Custom update strategy to replace the previous value with the new one."""
    return new

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
    identified_instance: Annotated[Sequence, replace_messages]
    instance_text: Annotated[Sequence, replace_messages]
    filtered_messages: Annotated[Sequence, replace_messages]

def filter_retrieved_record(messages: list):
    records = [msg.content for msg in filter(lambda x: isinstance(x, ToolMessage), messages)]
    return sorted(list(set(records)))

def filter_toolcalls(messages: list):
    toolcalls = [msg for msg in filter(lambda x: isinstance(x, AIMessage) and len(x.tool_calls) > 0, messages)]
    return toolcalls

def from_object_to(state: AgentState):
    messages = state["messages"]

    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        # TODO FIX ME
        print(last_message.content)
        import pdb; pdb.set_trace()
        return "next"
    else:
        return "call_tools"
    
def from_backward_search_to(state: AgentState):
    pass
    
def try_except_continue(state, func):
    while True:
        try:
            ret = func(state)
            return ret
        except Exception as e:
            print("I crashed trying to run:", func)
            print("Here is my error")
            print(e)
            traceback.print_exception(*sys.exc_info())
            continue

class ReMEmbRVideoPlanner2(Planner):

    def __init__(self, llm_type='gpt-4o', num_ctx=8192, temperature=0):
        # TODO read in from some config file
        self.config_max_object_query_call_cnt = 3

        # Wrapper that handles everything
        llm = self.llm_selector(llm_type, temperature, num_ctx)
        chat = FunctionsWrapper(llm)

        self.num_ctx = num_ctx
        self.temperature = temperature

        self.chat = chat
        self.llm_type = llm_type
        ### Load vectorstore
        self.embeddings = HuggingFaceEmbeddings(model_name='mixedbread-ai/mxbai-embed-large-v1')

        top_level_path = str(os.path.dirname(__file__)) + '/../'
        self.instance_txt_prompt = file_to_string(top_level_path+'prompts/video_planner/instance_txt_prompt.txt')
        self.instance_txt_terminate_prompt = file_to_string(top_level_path+'prompts/video_planner/instance_txt_terminate_prompt.txt')
        self.instance_vid_prompt = file_to_string(top_level_path+'prompts/video_planner/instance_vid_prompt.txt')
        self.instance_location_prompt = file_to_string(top_level_path+'prompts/video_planner/instance_location_prompt.txt')
        self.instance_desc_prompt = file_to_string(top_level_path+'prompts/video_planner/instance_desc_prompt.txt')
        self.txt_search_prompt = file_to_string(top_level_path+'prompts/video_planner/txt_search_prompt.txt')
        self.same_instance_prompt = file_to_string(top_level_path+'prompts/video_planner/same_instance_prompt.txt')

        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.object_query_call_cnt = 0

        self.chat_history = ChatMessageHistory()

    def llm_selector(self, llm_type, temperature, num_ctx):
        llm = None
        # Support for LLM Gateway
        if 'gpt-4' in llm_type:
            import os
            llm = ChatOpenAI(model='gpt-4o', api_key=os.environ.get("OPENAI_API_KEY"))

        # Support for NIMs
        elif 'nim/' in llm_type:
            llm_name = llm_type[4:]
            llm = ChatNVIDIA(model=llm_name)

        # Support for Ollama functions
        elif llm_type == 'command-r':
            llm = ChatOllama(model=llm_type, temperature=temperature, num_ctx=num_ctx)
        else:
            llm = ChatOllama(model=llm_type, format="json", temperature=temperature, num_ctx=num_ctx)

        if llm is None:
            raise Exception("No correct LLM provided")

        return llm

    def set_memory(self, memory: MilvusVideoMemory):
        self.memory = memory
        self.create_tools(memory)
        self.build_graph()

    def create_tools(self, memory):

        class TextRetrieverInput(BaseModel):
            x: str = Field(description="The query that will be searched by the vector similarity-based retriever.\
                                Text embeddings of this description are used. There should always be text in here as a response! \
                                Based on the question and your context, decide what text to search for in the database. \
                                This query argument should be a phrase such as 'a crowd gathering' or 'a green car driving down the road'.\
                                The query will then search your memories for you.")

        self.retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.video_search_by_text(x),
            name="retrieve_from_text",
            description="Search and return information from your video memory in the form of captions",
            args_schema=TextRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        class PositionRetrieverInput(BaseModel):
            x: tuple = Field(description="The query that will be searched by finding the nearest memories at this (x,y,z) position.\
                                The query must be an (x,y,z) array with floating point values \
                                Based on the question and your context, decide what position to search for in the database. \
                                This query argument should be a position such as (0.5, 0.2, 0.1). They should NOT be a string. \
                                The query will then search your memories for you.")
        # position-based tool
        self.position_retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_by_position(x),
            name="retrieve_from_position",
            description="Search and return information from your video memory by using a position array such as (x,y,z)",
            args_schema=PositionRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        class TimeRetrieverInput(BaseModel):
            x: str = Field(description="The query that will be searched by finding the nearest memories at a specific time in H:M:S format.\
                                The query must be a string containing only time. \
                                Based on the question and your context, decide what time to search for in the database. \
                                This query argument should be an YYYY-MM-DD HH:MM:SS time such as 2024-01-01 08:02:03 with leading zeros. \
                                The query will then search your memories for you.")

        # position-based tool
        self.time_retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_by_datetime(x),
            name="retrieve_from_time",
            description="Search and return information from your video memory by using an YYYY-MM-DD HH:MM:SS time.",
            args_schema=TimeRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        self.tool_list = [self.retriever_tool, self.position_retriever_tool, self.time_retriever_tool]
        self.tool_definitions = [convert_to_openai_function(t) for t in self.tool_list]
        
    def instance_txt(self, state):
        
        # self._group_records("someone using or drinking water with a cup/mug/bottle")
        # import pdb; pdb.set_trace
        
        messages = state["messages"]
        model = self.chat
        if self.object_query_call_cnt < self.config_max_object_query_call_cnt:
            model = model.bind_tools(tools=self.tool_definitions)
            prompt = self.instance_txt_prompt
        else:
            prompt = self.instance_txt_terminate_prompt
            # TODO this is the WRONG place to clear count. FIX ME!!!
            self.object_query_call_cnt = 0
        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder("chat_history"),
                (("human"), self.previous_tool_requests),
                ("system", prompt),
                ("human", "{question}"),
            ]
        )
        
        model = prompt | model
        question = messages[0].content
        question = f"The question from the user is: {question}"
        # Convert all ToolMessages into AI Messages since Ollama cann't handle ToolMessage
        if ('gpt-4' not in self.llm_type) and ('nim' not in self.llm_type):
            for i in range(len(messages)):
                if type(messages[i]) == ToolMessage:
                    messages[i] = AIMessage(id=messages[i].id, content=messages[i].content) # ignore tool_call_id
        if "the day before yesterday" in question:
            date = "02-05"
        elif "yesterday" in question:
            date = "02-06"
        else:
            date = ""
            
        retrieved_messages = filter_retrieved_record(messages=messages)
        retrieved_records = []
        for records in retrieved_messages:
            # retrieved_records += json.loads(record)
            for r in json.loads(records):
                if date in r["text"]:
                    retrieved_records.append(r)
                
        processed_records = [{"record_id": record["id"], "text": record["text"]} for record in retrieved_records]
        parsed_processed_records = [json.dumps(record) for record in processed_records]
        response = model.invoke({"question": question, "chat_history": parsed_processed_records})
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call['name'] != "__conversational_response":
                    args = re.sub("\{.*?\}", "", str(tool_call['args'])) # remove curly braces
                    self.previous_tool_requests += f"I previously used the {tool_call['name']} tool with the arguments: {args}.\n"
            self.object_query_call_cnt += 1
            
        if not response.tool_calls:
            response_content = eval(response.content)
            record_ids = response_content["record_ids"]
            if type(record_ids) is str:
                record_ids = eval(record_ids)
            debugdir = "/home/tiejean/RobotMem/remembr/debug/top1_txt_past"
            for record in retrieved_records:
                if record["id"] == record_ids[0]:
                    self._debug_vid(record, debugdir)
            return {"messages": [response], "instance_text": response, "filtered_messages": retrieved_records}
        else:
            return {"messages": [response], "filtered_messages": retrieved_records}
        
    def instance_vid(self, state):
        messages = state["messages"]
        last_message = messages[-1]
        model = self.chat
        prompt = self.instance_vid_prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder("chat_history"),
                ("ai", prompt),
                ("human", "{question}"),
            ]
        )
        question = f"The question from the user is: {messages[0]}"
        last_message_content = eval(last_message.content)
        if "text" in last_message_content.keys():
            question += f"This object is likely to match the following description: {last_message_content['text']}"
        model = prompt | model
        
        # if ('gpt-4' not in self.llm_type) and ('nim' not in self.llm_type):
        #     for i in range(len(messages)):
        #         if type(messages[i]) == ToolMessage:
        #             messages[i] = AIMessage(id=messages[i].id, content=messages[i].content) # ignore tool_call_id
        # retrieved_messages = filter_retrieved_record(messages=messages)
        # retrieved_records = []
        # for record in retrieved_messages:
        #     retrieved_records += json.loads(record)
        # processed_records = [{"record_id": record["id"], "text": record["text"]} for record in retrieved_records]
        # parsed_processed_records = [json.dumps(record) for record in processed_records]
        retrieved_records = state["filtered_messages"]
        processed_records = [{"record_id": record["id"], "text": record["text"]} for record in retrieved_records]
        parsed_processed_records = [json.dumps(record) for record in processed_records]
        response = model.invoke({"question": question, "chat_history": parsed_processed_records})
        
        identified_records = self._parse_vid_response(response, retrieved_records)
        
        # TODO handle the logics here
        if True:
            response_content = eval(response.content)
            if "top_1_record" not in response_content.keys():
                raise ValueError("LLM couldn't identify the video clip")
            
            # TODO debug
            debugdir = "/home/tiejean/RobotMem/remembr/debug/top1_past"
            for record in retrieved_records:
                if record["id"] == response_content["top_1_record"]:
                    self._debug_vid(record, debugdir)
                        
        return {"identified_instance": identified_records, "messages": [response]}
    
    def _debug_vid(self, vid, debugdir: str):
        os.makedirs(debugdir, exist_ok=True)
        for filename in os.listdir(debugdir):
                filepath = os.path.join(debugdir, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
        images = self._get_images(vid)
        for i, img in enumerate(images):
                imgpath = os.path.join(debugdir, f"{i:06d}.png")
                import cv2; cv2.imwrite(imgpath, img)
                
    def _debug_grouped_vids(self, vids, debugdir: str):
        os.makedirs(debugdir, exist_ok=True)
        for filename in os.listdir(debugdir):
                filepath = os.path.join(debugdir, filename)
                if os.path.isfile(filepath):
                    os.remove(filepath)
        for i, vid in enumerate(vids):
            img = self._get_image(vid)
            imgpath = os.path.join(debugdir, f"{i:06d}.png")
            import cv2; cv2.imwrite(imgpath, img)
    
    def _parse_vid_response(self, response, retrieved_records):
        vids = [None, None, None]
        vids_meta = eval(response.content)
        for record in retrieved_records:
            idx = None
            if record["id"] == vids_meta["top_1_record"]:
                idx = 0
                reason = vids_meta["reasoning_for_top_1_record"]
            elif record["id"] == vids_meta["top_2_record"]:
                idx = 1
                reason = vids_meta["reasoning_for_top_2_record"]
            elif record["id"] == vids_meta["top_3_record"]:
                idx = 2
                reason = vids_meta["reasoning_for_top_3_record"]
            if idx is not None:
                import copy; vids[idx] = copy.copy(record)
                vids[idx]["reason"] = reason
        return vids
    
    def _get_image(self, vid_info: dict, type: str = "opencv"):
        vidpath, start_frame, end_frame = vid_info["vidpath"], vid_info["start_frame"], vid_info["end_frame"]
        start_frame, end_frame = int(start_frame), int(end_frame)
        frame = (start_frame + end_frame) // 2
        imgpath = os.path.join(vidpath, f"{frame:06d}.png")
        if type.lower() == "opencv":
            img = cv2.imread(imgpath)
            return img
        elif type.lower() == "pil":
            from PIL import Image; img = Image.open(imgpath)
            return img
        elif type.lower() == "utf-8":
            import base64
            with open(imgpath, "rb") as imgfile:
                data = imgfile.read()
                img = base64.b64encode(data)
                encoded_img = copy.copy(img.decode("utf-8"))
            return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_img}"}}
        else:
            raise ValueError("Invalid image data type: only support opencv or PIL")
    
    def _get_images(self, vid_info: dict, type: str = "opencv", sample_rate: int = 1):
        vidpath, start_frame, end_frame = vid_info["vidpath"], vid_info["start_frame"], vid_info["end_frame"]
        start_frame, end_frame = int(start_frame), int(end_frame)
        images = []
        for frame in range(start_frame, end_frame+1, sample_rate):
            imgpath = os.path.join(vidpath, f"{frame:06d}.png")
            if type.lower() == "opencv":
                img = cv2.imread(imgpath)
                images.append(img)
            elif type.lower() == "pil":
                from PIL import Image; img = Image.open(imgpath)
                images.append(img)
            elif type.lower() == "utf-8":
                import base64
                with open(imgpath, "rb") as imgfile:
                    data = imgfile.read()
                    img = base64.b64encode(data)
                    encoded_img = copy.copy(img.decode("utf-8"))
                images.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{encoded_img}"}}) 
                break # TODO remove me
            else:
                raise ValueError("Invalid image data type: only support opencv or PIL")
        return images
    
    def _is_same_instance(self, img1, img2, instance_txt: str):
        image_messages = [img1, img2]
        from langchain_core.messages import HumanMessage
        prompt = self.same_instance_prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                ("ai", prompt),
                HumanMessage(content=image_messages),
                ("human", "{question}")
            ]
        )
        question = "Does these two images contain the same mug, cup, or bottle?" # TODO
        
        model = self.chat
        model = prompt | model
        response = model.invoke({"question": question})
        response_content = eval(response.content)
        is_same_instance = response_content["same_instance"] == "yes"
        return (is_same_instance, response)
    
    def _group_records(self, instance_txt: str):
        retrieved_records = self._retrieve_all_records(instance_txt)
        merged_records = self.__merge_records(retrieved_records)
        self._debug_grouped_vids(merged_records, debugdir="/home/tiejean/RobotMem/remembr/debug/segments")
        import pdb; pdb.set_trace()
        n = len(merged_records)
        uf = UnionFind(n)
        
        responses = []
        import numpy as np; same_instances = np.zeros((n, n))
        for i in range(n):
            img1 = self._get_image(merged_records[i], type="utf-8")
            for j in range(i+1, n):
                img2 = self._get_image(merged_records[j], type="utf-8")
                (same_instance, detailed_response) = self._is_same_instance(img1, img2, "")
                responses.append((i, j, detailed_response))
                if same_instance:
                    same_instances[i,j] = 1
                    # uf.union(i, j)
        import pdb; pdb.set_trace()
                
        # groups = {}
        # for i in range(n):
        #     root = uf.find(i)
        #     if root not in groups:
        #         groups[root] = []
        #     groups[root].append(merged_records[i])
        
        return responses
        
    def __merge_records(self, records: list):
        if len(records) == 0:
            return records
        records = sorted(records, key=lambda x: x["text"])
        merged_records = []
        curr_record = copy.copy(records[0])
        for record in records[1:]:
            t1 = float(curr_record["id"])
            t2 = float(record["id"])
            if t2 - t1 < 0.2: # TODO
                curr_record["id"] = t2
                curr_record["end_frame"] = record["end_frame"]
                curr_record["text"] += record["text"]
            else:
                merged_records.append(curr_record)
                curr_record = copy.copy(record)
        return merged_records
    
    def _retrieve_all_records(self, instance_txt: str, date: str = None):
        retrieved_records = json.loads(self.memory.searck_k_by_text(instance_txt, k=100))
        if date is not None:
            filtered_records = []
            for record in retrieved_records:
                if date in record["text"]:
                    filtered_records.append(record)
            retrieved_records = filtered_records
        return retrieved_records
    
    def _retrieve_records_by_time(self, time: float, instance_txt: str = ""):
        now = strftime("%Y-%m-%d %H:%M:%S", localtime(time))
        retrieved_records_by_time = json.loads(self.memory.search_k_by_datetime(now, k=200))
        today = strftime("%Y-%m-%d", localtime(time))
        retrieved_records_by_text = []
        if len(instance_txt) > 0:
            retrieved_records_by_text = json.loads(self.memory.video_search_by_text(f"{instance_txt} on {today}"))
        all_retrieved_records = retrieved_records_by_time + retrieved_records_by_text
        record_ids = set()
        for record in all_retrieved_records:
            if today in record["text"]:
                record_ids.add(record["id"])
        retrieved_records = []
        for record in all_retrieved_records:
            if record["id"] in record_ids:
                retrieved_records.append(record)
                record_ids.remove(record["id"])
        return retrieved_records
    
    def _find_vids(self, records: list, question: str):
        model = self.chat
        prompt = self.instance_vid_prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder("chat_history"),
                ("ai", prompt),
                ("human", "{question}"),
            ]
        )
        model = prompt | model
        response = model.invoke({"question": question, "chat_history": records})
        return response
    
    def _find_locs(self, records: list, question: str):
        model = self.chat
        prompt = self.instance_location_prompt
        prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder("chat_history"),
                ("ai", prompt),
                ("human", "{question}"),
            ]
        )
        model = prompt | model
        response = model.invoke({"question": question, "chat_history": records})
        return response
    
    def instance_desc(self, state):
        instance_txt = eval(state["instance_text"].content)["text"]
        identified_vids = state["identified_instance"]
        
        identified_instances = []
        for vid in identified_vids:
            if vid is None:
                break
            imgs = self._get_images(vid, type="utf-8", sample_rate=5)
            
            human_message_content = imgs
            question = f"The user is looking for a detailed, contextualized description of an instance appearing in a video. The instance is likely to match the following description: {instance_txt}"
            from langchain_core.messages import HumanMessage
            
            prompt = self.instance_desc_prompt
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("ai", prompt),
                    HumanMessage(content=human_message_content),
                    ("human", "{question}")
                ]
            )
            model = self.chat
            model = prompt | model
            response = model.invoke({"question": question})
            entry = copy.copy(vid)
            entry["desc"] = response.content
            identified_instances.append(entry)
        return {"identified_instance": identified_instances, "messages": [response]}
    
    def backward_search(self, state):
        # last_message = state["messages"][-1]
        # instance_txt_query = eval(last_message.content)
        # instance_txt_query = instance_txt_query["text"]
        
        identified_instances = state["identified_instance"]
        for identified_instance in identified_instances:
            instance_txt_query = identified_instance["desc"]
        
            import time; from time import strftime, localtime
            
            # TODO FIX ME
            currtime = time.time()
            currtime = 1738900977.4448056 # TODO delete me
            
            retrieved_records = self._retrieve_all_records(instance_txt_query, date="02-07") # TODO FIX ME
            import pdb; pdb.set_trace()
            # retrieved_records = self._retrieve_records_by_time(time=currtime, instance_txt=instance_txt_query)
            processed_records = []
            for record in retrieved_records:
                processed_record = {"record_id": record["id"], "text": record["text"]}
                processed_records.append(processed_record)
            parsed_processed_records = [json.dumps(record) for record in processed_records]
            
            vid_question = f"I want to find an object that match this desciption: {instance_txt_query}. When is the last time I saw this object?"
            vid_response_msg = self._find_vids(records=parsed_processed_records, question=vid_question)
            vid_response = eval(vid_response_msg.content)
            loc_question = f"I want to find an object that match this desciption: {instance_txt_query}. Where did I observe this object last time?"
            loc_response_msg = self._find_locs(records=parsed_processed_records, question=loc_question)
            loc_response = eval(loc_response_msg.content)
            
            debugdir = "/home/tiejean/RobotMem/remembr/debug/top1_now"
            for record in retrieved_records:
                if record["id"] == vid_response["top_1_record"]:
                    self._debug_vid(record, debugdir)
            if len(vid_response["top_2_record"]) > 0:
                debugdir = "/home/tiejean/RobotMem/remembr/debug/top2_now"
                for record in retrieved_records:
                    if record["id"] == vid_response["top_2_record"]:
                        self._debug_vid(record, debugdir)
            import pdb; pdb.set_trace()
            
        return {"messages": [vid_response, loc_response]}

    def instances(self, state):
        import pdb; pdb.set_trace()
        self._group_records("someone using or drinking water with a cup/mug/bottle")
        import pdb; pdb.set_trace()

    def build_graph(self):

        from langgraph.graph import END, StateGraph
        from langgraph.prebuilt import ToolNode

        # Define a new graph
        workflow = StateGraph(AgentState)
        workflow.add_node("instances", lambda state: try_except_continue(state, self.instances))
        workflow.add_node("instance_desc", lambda state: try_except_continue(state, self.instance_desc))
        workflow.add_node("backward_search", lambda state: try_except_continue(state, self.backward_search))
        
        workflow.add_edge("instances", "instance_desc")
        workflow.add_edge("instance_desc", "backward_search")
        workflow.add_edge("backward_search", END)
        
        workflow.set_entry_point("instances")
        # Compile
        self.graph = workflow.compile()

    def query(self, question: str):

        inputs = { "messages": [
                                (("user", question)),
            ]
        }
        self.graph.invoke(inputs)
        # out = self.graph.invoke(inputs)
        # # out = self.graph.invoke(inputs, config={"recursion_limit": 50})
        # response = out['messages'][-1]
        # response = ''.join(response.content.splitlines())

        # if '```json' not in response:
        #     # try parsing on its own since we cannot always trust llms
        #     parsed = eval(response) 
        # else:
        #     parsed = parse_json(response)

        # response = PlannerOutput.from_dict(parsed)

        # return response

if __name__ == "__main__":
    pass