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


import sys, os
sys.path.append(os.getcwd())

from remembr.utils.util import file_to_string
from remembr.tools.tools import *
from remembr.tools.functions_wrapper import FunctionsWrapper

from remembr.memory.memory import Memory

from remembr.planners.planner import Planner, PlannerOutput

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
        raise ValueError("Generate call failed. Retrying...")
    return parsed

def replace_messages(current: Sequence, new: Sequence):
    """Custom update strategy to replace the previous value with the new one."""
    return new

class AgentState(TypedDict):
    # The add_messages function defines how an update should be processed
    # Default is to replace. add_messages says "append"
    messages: Annotated[Sequence[BaseMessage], add_messages]
    context: Annotated[Sequence, replace_messages]
    object_level_plans: Annotated[Sequence, replace_messages]
    records: Annotated[Sequence, replace_messages] # TODO delete me
    toolcalls: Annotated[Sequence, replace_messages] # TODO delete me

def filter_retrieved_record(messages: list):
    records = [msg.content for msg in filter(lambda x: isinstance(x, ToolMessage), messages)]
    return sorted(list(set(records)))

def filter_toolcalls(messages: list):
    toolcalls = [msg for msg in filter(lambda x: isinstance(x, AIMessage) and len(x.tool_calls) > 0, messages)]
    return toolcalls

def from_objects_to(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    if not last_message.tool_calls:
        return "object_plans"
    else:
        return "objects_action"
    
def from_object_plans_to(state: AgentState):
    object_plans = state["object_level_plans"]
    for _, v in object_plans.items():
        if not v["has_planned"]:
            return "object_plans_action"
        else:
            import pdb; pdb.set_trace()
            return "end" # TODO

# Define the function that determines whether to continue or not
def from_agent_to(state: AgentState):
    messages = state["messages"]

    last_message = messages[-1]
    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "generate"
    else:
        return "agent_action"
    
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

class ReMEmbRPlanner(Planner):

    def __init__(self, llm_type='gpt-4o', num_ctx=8192, temperature=0):
        # TODO read in from some config file
        self.config_max_objects_call_cnt = 3
        self.config_max_object_plans_call_cnt = 1
        self.config_max_agent_call_cnt = 3

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
        self.objects_prompt = file_to_string(top_level_path+'prompts/planner/planner_objects_prompt.txt')
        self.objects_terminate_prompt = file_to_string(top_level_path+'prompts/planner/planner_objects_terminate_prompt.txt')
        self.object_plans_prompt = file_to_string(top_level_path+'prompts/planner/planner_object_plans_prompt.txt')
        self.object_plans_terminate_prompt = file_to_string(top_level_path+'prompts/planner/planner_object_plans_terminate_prompt.txt')
        self.agent_prompt = file_to_string(top_level_path+'prompts/planner/planner_system_prompt.txt')
        self.generate_prompt = file_to_string(top_level_path+'prompts/planner/generate_system_prompt.txt')
        self.agent_terminate_prompt = file_to_string(top_level_path+'prompts/planner/planner_agent_terminate_prompt.txt')
        self.critic_prompt = file_to_string(top_level_path+'prompts/planner/critic_system_prompt.txt')

        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.objects_call_count = 0
        self.object_plans_call_count = 0
        self.agent_call_count = 0

        self.chat_history = ChatMessageHistory()

    def llm_selector(self, llm_type, temperature, num_ctx):
        llm = None
        # Support for LLM Gateway
        if 'gpt-4' in llm_type:
            import os
            llm = ChatOpenAI(model=llm_type, api_key=os.environ.get("OPENAI_API_KEY"))

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

    def set_memory(self, memory: Memory):
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
            func=lambda x: memory.search_by_text(x),
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
                                This query argument should be an HMS time such as 08:02:03 with leading zeros. \
                                The query will then search your memories for you.")

        # position-based tool
        self.time_retriever_tool = StructuredTool.from_function(
            func=lambda x: memory.search_by_time(x),
            name="retrieve_from_time",
            description="Search and return information from your video memory by using an H:M:S time.",
            args_schema=TimeRetrieverInput
            # coroutine= ... <- you can specify an async method if desired as well
        )

        self.tool_list = [self.retriever_tool, self.position_retriever_tool, self.time_retriever_tool]
        self.tool_definitions = [convert_to_openai_function(t) for t in self.tool_list]

    def objects(self, state):
        messages = state["messages"]
        model = self.chat
        if self.objects_call_count < self.config_max_objects_call_cnt:
            model = model.bind_tools(tools=self.tool_definitions)
            prompt = self.objects_prompt
        else:
            prompt = self.objects_terminate_prompt
            self.objects_call_count = 0
        
        objects_prompt = ChatPromptTemplate.from_messages(
            [
                # ("system", prompt),
                MessagesPlaceholder("chat_history"),
                (("human"), self.previous_tool_requests),
                ("ai", prompt),
                ("human", "{question}"),
            ]
        )
        model = objects_prompt | model
        question = f"User-specified request: {messages[0]}"
        # Convert all ToolMessages into AI Messages since Ollama cann't handle ToolMessage
        if ('gpt-4' not in self.llm_type) and ('nim' not in self.llm_type):
            for i in range(len(messages)):
                if type(messages[i]) == ToolMessage:
                    messages[i] = AIMessage(id=messages[i].id, content=messages[i].content) # ignore tool_call_id
        response = model.invoke({"question": question, "chat_history": messages[:]})
        
        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call['name'] != "__conversational_response":
                    args = re.sub("\{.*?\}", "", str(tool_call['args'])) # remove curly braces
                    self.previous_tool_requests += f"I previously used the {tool_call['name']} tool with the arguments: {args}.\n"

        self.objects_call_count += 1
        
        return {"messages": [response]}
    
    def object_plans(self, state):
        messages = state["messages"]
        object_plans = state["object_level_plans"]
        context = state["context"]
        if messages[-1].content and "objects" in messages[-1].content and "answer_reasoning" in messages[-1].content:
            last_response = parse_response(messages[-1], special_keys=["objects"])
            object_plans = {obj:{"has_planned":False, "plans":[]} for obj in last_response["objects"]}
            context = last_response["answer_reasoning"]
            
        object = None
        for k,v in object_plans.items():
            if not v["has_planned"]:
                object = k
        if object is None:
            return None # FIXME
        
        model = self.chat
        if self.object_plans_call_count < self.config_max_object_plans_call_cnt:
            model = model.bind_tools(tools=self.tool_definitions)
            prompt = self.object_plans_prompt
        else:
            prompt = self.object_plans_terminate_prompt
            object_plans[k]["has_planned"] = True
            self.object_plans_call_count = 0
            
        object_plans_prompt = ChatPromptTemplate.from_messages(
            [
                MessagesPlaceholder("chat_history"),
                (("human"), self.previous_tool_requests),
                ("ai", prompt),
                ("human", "{question}"),
            ]
        )
        question = f"Context: {context}.\n Question: How can you retrieve {object}? Where are you likely to find this object? What actions should you take?"
        model = object_plans_prompt | model
        
        # Convert all ToolMessages into AI Messages since Ollama cann't handle ToolMessage
        if ('gpt-4' not in self.llm_type) and ('nim' not in self.llm_type):
            for i in range(len(messages)):
                if type(messages[i]) == ToolMessage:
                    messages[i] = AIMessage(id=messages[i].id, content=messages[i].content) # ignore tool_call_id
        retrieved_records = filter_retrieved_record(messages=messages)
        response = model.invoke({"question": question, "chat_history": retrieved_records})
        
        if not response.tool_calls and response.content:
            parsed_response = parse_response(response, special_keys=["plans"])
            object_plans[object]["plans"] = parsed_response["plans"]
            object_plans[object]["has_planned"] = True
            
        self.object_plans_call_count += 1
            
        import copy
        return {"messages": [response], "context": context, "object_level_plans": copy.deepcopy(object_plans)}

    ### Nodes
    def agent(self, state):
        """
        Invokes the agent model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply end.

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with the agent response appended to messages
        """
        messages = state["messages"]

        model = self.chat


        # limit tool calls.
        if self.agent_call_count < self.config_max_agent_call_cnt:
            model = model.bind_tools(tools=self.tool_definitions)
            prompt = self.agent_prompt
        else:
            prompt = self.agent_terminate_prompt


        agent_prompt = ChatPromptTemplate.from_messages(
            [
                # ("system", prompt),
                MessagesPlaceholder("chat_history"),
                (("human"), self.previous_tool_requests),
                ("ai", prompt),
                ("human", "{question}"),

            ]
        )


        model = agent_prompt | model

        question = f"The question is: {messages[0]}"

        # Convert all ToolMessages into AI Messages since Ollama cann't handle ToolMessage
        if ('gpt-4' not in self.llm_type) and ('nim' not in self.llm_type):
            for i in range(len(messages)):
                if type(messages[i]) == ToolMessage:
                    messages[i] = AIMessage(id=messages[i].id, content=messages[i].content) # ignore tool_call_id


        response = model.invoke({"question": question, "chat_history": messages[:]})

        if response.tool_calls:
            for tool_call in response.tool_calls:
                if tool_call['name'] != "__conversational_response":
                    args = re.sub("\{.*?\}", "", str(tool_call['args'])) # remove curly braces
                    self.previous_tool_requests += f"I previously used the {tool_call['name']} tool with the arguments: {args}.\n"

        self.agent_call_count += 1


        return {"messages": [response]}

    def generate(self, state):
        """
        Generate answer

        Args:
            state (messages): The current state

        Returns:
            dict: The updated state with re-phrased question
        """
        messages = state["messages"]
        question = messages[0].content \
                + "\n Please responsed in the desired format."

        prompt = PromptTemplate(
            template=self.generate_prompt,
            input_variables=["context", "question"],
        )
        filled_prompt = prompt.invoke({'question':question})

        gen_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", filled_prompt.text),
                MessagesPlaceholder("chat_history"),
                ("human", "{question}"),

            ]
        )

        model = gen_prompt | self.chat
        retrieved_record = filter_retrieved_record(messages=messages)
        # response = model.invoke({"question": question, "chat_history": messages[1:]})
        response = model.invoke({"question": question, "chat_history": retrieved_record})

        # let us parse and check the output is a dictionary. raise error otherwise
        response = ''.join(response.content.splitlines())
        
        try:
            if '```json' not in response:
                # try parsing on its own since we cannot always trust llms
                parsed = eval(response) 
            else:
                parsed = parse_json(response)
                
            # then check it has all the required keys
            keys_to_check_for = ["answer_reasoning", "question", "plans"]

            for key in keys_to_check_for:
                if key not in parsed:
                    raise ValueError("Missing all the required keys during generate. Retrying...")
                
            if type(parsed['plans']) == str:
                parsed['plans'] = eval(parsed['plans'])

        except:
            raise ValueError("Generate call failed. Retrying...")
        
        self.previous_tool_requests = "These are the tools I have previously used so far: \n"
        self.agent_call_count = 0
        return {"messages": [str(parsed)]}

    def build_graph(self):

        from langgraph.graph import END, StateGraph
        from langgraph.prebuilt import ToolNode

        # Define a new graph
        workflow = StateGraph(AgentState)
        
        # Testing
        workflow.add_node("objects", lambda state: try_except_continue(state, self.objects))
        workflow.add_node("objects_action", ToolNode(self.tool_list))
        workflow.add_node("object_plans", lambda state: try_except_continue(state, self.object_plans))
        workflow.add_node("object_plans_action", ToolNode(self.tool_list))
        
        workflow.add_edge('objects_action', 'objects')
        workflow.add_conditional_edges(
            "objects",
            from_objects_to,
            {
                "objects_action": "objects_action",
                "object_plans": "object_plans",
            },
        )
        workflow.add_edge('object_plans_action', 'object_plans')
        workflow.add_conditional_edges(
            "object_plans",
            from_object_plans_to,
            {
                "object_plans_action": "object_plans_action",
                "end": END,
            }
        )
        workflow.set_entry_point("objects")
        # # Define the nodes we will cycle between
        # workflow.add_node("agent", lambda state: try_except_continue(state, self.agent))  # agent
        # workflow.add_node("agent_action", ToolNode(self.tool_list))

        # workflow.add_node(
        #     "generate", lambda state: try_except_continue(state, self.generate)
        # )  # Generating a response after we know the documents are relevant
        # # Call agent node to decide to retrieve or not
        # workflow.set_entry_point("agent")

        # workflow.add_edge('agent_action', 'agent')
        # # Decide whether to retrieve
        # workflow.add_conditional_edges(
        #     "agent",
        #     # Assess agent decision
        #     from_agent_to,
        #     {
        #         # Translate the condition outputs to nodes in our graph
        #         "agent_action": "agent_action",
        #         "generate": "generate",
        #     },
        # )
        # workflow.add_edge("generate", END)

        # Compile
        self.graph = workflow.compile()


    def query(self, question: str):

        inputs = { "messages": [
                                (("user", question)),
            ]
        }
        out = self.graph.invoke(inputs)
        # out = self.graph.invoke(inputs, config={"recursion_limit": 50})
        response = out['messages'][-1]
        response = ''.join(response.content.splitlines())

        if '```json' not in response:
            # try parsing on its own since we cannot always trust llms
            parsed = eval(response) 
        else:
            parsed = parse_json(response)

        response = PlannerOutput.from_dict(parsed)

        return response

if __name__ == "__main__":
    pass