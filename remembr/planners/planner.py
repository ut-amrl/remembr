from dataclasses import dataclass, field, asdict
from typing import List, Any, Dict
import inspect

@dataclass
class Plan:
    reason: str
    action: str
    object: str
    position: List[float]

    @classmethod
    def from_dict(cls, dict_input):      
        return cls(**{
            k: v for k, v in dict_input.items() 
            if k in inspect.signature(cls).parameters
        })

@dataclass
class PlannerOutput:
    answer_reasoning: str
    question: str
    plans: List[Plan] = field(default_factory=list)
    
    @classmethod
    def from_dict(cls, dict_input):      
        # Extract plans and convert each entry to a Plan object
        plans = [Plan.from_dict(plan) for plan in dict_input.get("plans", [])]
        # Filter and construct the PlannerOutput object
        init_args = {
            k: (plans if k == "plans" else v)
            for k, v in dict_input.items()
            if k in inspect.signature(cls).parameters
        }
        return cls(**init_args)
    
    def print_plan(self, show_reasons=False):
        for plan in self.plans:
            s = f"{plan.action} {plan.object} at {plan.position}"
            if show_reasons:
                s += f" because {plan.reason}"
            print(s)

class Planner:
    def query(self, query: str) -> PlannerOutput:
        raise NotImplementedError
    
    def query_plans(self, query: str) -> list:
        return self.query(query).plans