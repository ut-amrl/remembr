from dataclasses import dataclass
import inspect

@dataclass
class PlannerOutput:
    answer_reasoning: str
    positions: list
    plans: list
    text: str
    
    @classmethod
    def from_dict(cls, dict_input):      
        return cls(**{
            k: v for k, v in dict_input.items() 
            if k in inspect.signature(cls).parameters
        })

class Planner:
    def query(self, query: str) -> PlannerOutput:
        raise NotImplementedError
    
    def query_positions(self, query: str) -> list:
        return self.query(query).positions