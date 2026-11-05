from dataclasses import dataclass, field
from chorus import ActorID, spawn, terminate, print_actors
import importlib
import json

from .geometry import Polygon 


PROGRAM_LOOKUP_FILE =  "./data/program_lookup.json"


@dataclass
class PaperRecord:
    encoding: int 
    tags: list[Polygon]

def load_program_names(filename: str) -> list[str]:
    """Loads and returns a list of the names of existing programs.

    This list is ordered such that a program can be indexed by its encoding.

    Args:
        filename: Filename to load from.
    """
    with open(filename) as f:
        programs = json.load(f)
    assert type(programs) is list
    assert all(type(p) is str for p in programs)
    return programs

program_lookup = load_program_names(PROGRAM_LOOKUP_FILE)


@dataclass 
class SpatialModel:
    encoding_to_paper_record: dict[int, PaperRecord] = field(default_factory=dict)
    actor_id_to_paper_record: dict[ActorID, PaperRecord] = field(default_factory=dict)
    encoding_to_actor_id: dict[int, ActorID] = field(default_factory=dict)
    
    def update(self, visible_papers:list[PaperRecord]):
        actor_id_to_seen: dict[ActorID, bool] = {aid:False for aid in self.actor_id_to_paper_record}
        for paper in visible_papers:
            if paper.encoding not in self.encoding_to_paper_record:
                module_name = "puck.programs." + program_lookup[paper.encoding]
                module = importlib.import_module(module_name)
                aid = spawn(target=module.run)
                print(f"I've made actor: {aid}")
                self.actor_id_to_paper_record[aid] = paper
                self.encoding_to_paper_record[paper.encoding] = paper
                self.encoding_to_actor_id[paper.encoding] = aid
                actor_id_to_seen[aid] = True
            else:
                aid= self.encoding_to_actor_id[paper.encoding]
                actor_id_to_seen[aid] = True
        for aid in [aid for aid, seen in actor_id_to_seen.items() if not seen]:
            terminate(aid)
            print(f"Die, {aid}")
    
    def kill_all(self):
        for aid in self.actor_id_to_paper_record.keys():
            terminate(aid)
            print(f"Die, {aid}")

    def dummy(self):
        print("THIS IS TO TEST")
            
            




