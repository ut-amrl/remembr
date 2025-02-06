from dataclasses import dataclass
import inspect 

@dataclass
class MemoryItem:
    caption: str
    time: float
    position: list
    theta: float
    vidpath: str
    start_frame: int
    end_frame: int

    @classmethod
    def from_dict(cls, dict_input):      
        return cls(**{
            k: v for k, v in dict_input.items() 
            if k in inspect.signature(cls).parameters
        })
    
    def __post_init__(self):
        # Not every method will use a caption, so we set it to none in those cases
        if self.caption is None:
            self.caption = ''
        if self.vidpath is None:
            self.vidpath = ''
        if self.start_frame is None:
            self.start_frame = 0
        if self.end_frame is None:
            self.end_frame = 0

class Memory:

    def insert(self, item: MemoryItem):
        raise NotImplementedError

    def get_working_memory(self) -> list[MemoryItem]:
        raise NotImplementedError

    def search_by_position(self, query: tuple) -> list[MemoryItem]:
        raise NotImplementedError

    def search_by_time(self, hms_time_query: str) -> list[MemoryItem]:
        raise NotImplementedError

    def search_by_text(self, query: str) -> list[MemoryItem]:
        raise NotImplementedError

    def memory_to_string(self, memory_list: list[MemoryItem]) -> str:
        raise NotImplementedError

