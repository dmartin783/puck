import importlib
import json
import logging
from dataclasses import dataclass, field

from chorus import ActorID, Proxy, me, spawn, terminate
from PIL import Image

from ..geometry import Polygon

KILL_TIME = 100
PROGRAM_LOOKUP_FILE = "./data/program_lookup.json"
logger = logging.getLogger(__name__)


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


def invert_dict(my_dict):
    inverted_dict = {v: k for k, v in my_dict.items()}
    return inverted_dict


@dataclass
class SpatialModel:
    """
    Not actually a spatial model, simply a wrapper around program spawning
    """

    main: ActorID
    encoding_to_paper_record: dict[int, PaperRecord] = field(default_factory=dict)
    actor_id_to_paper_record: dict[ActorID, PaperRecord] = field(default_factory=dict)
    encoding_to_actor_id: dict[int, ActorID] = field(default_factory=dict)

    actor_id_time_since_seen: dict[ActorID, int] = field(default_factory=dict)

    def get_encoding_to_actor_id(self) -> dict[int, ActorID]:
        return self.encoding_to_actor_id

    def update(self, visible_papers: list[PaperRecord]):
        actor_id_to_seen: dict[ActorID, bool] = {
            aid: False for aid in self.actor_id_to_paper_record
        }
        for paper in visible_papers:
            if paper.encoding not in self.encoding_to_paper_record:
                module_name = "puck.programs." + program_lookup[paper.encoding]
                module = importlib.import_module(module_name)
                aid = spawn(module.run, Proxy(me()), Proxy(self.main))
                self.encoding_to_actor_id[paper.encoding] = aid
            else:
                aid = self.encoding_to_actor_id[paper.encoding]

            actor_id_to_seen[aid] = True
            self.actor_id_to_paper_record[aid] = paper
            self.encoding_to_paper_record[paper.encoding] = paper
            self.actor_id_time_since_seen[aid] = 0

        for aid in [aid for aid, seen in actor_id_to_seen.items() if not seen]:
            self.actor_id_time_since_seen[aid] += 1
            if self.actor_id_time_since_seen[aid] > KILL_TIME:
                terminate(aid)
                actor_id_to_encoding = invert_dict(self.encoding_to_actor_id)
                encoding = actor_id_to_encoding[aid]
                del self.actor_id_to_paper_record[aid]
                del self.encoding_to_actor_id[encoding]
                del self.encoding_to_paper_record[encoding]
                del self.actor_id_time_since_seen[aid]
                logger.info(f"{aid} terminated because not seen")

    def kill_all(self):
        for aid in self.actor_id_to_paper_record:
            terminate(aid)
            logger.info(f"{aid} terminated because puck ended")

    def corners(self, aid):
        paper_record = self.actor_id_to_paper_record[aid]
        return paper_record.tags


@dataclass
class SlideSpatial(SpatialModel):
    """
    main: ActorID is now slides.py main loop
    encoding_to_paper_record is now dict from slide id to Slide object

    """
