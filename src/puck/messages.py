from dataclasses import dataclass
from enum import Enum
from queue import Queue
from typing import Any, TYPE_CHECKING

lazy from .actor import Actor
from .geometry import Polygon


@dataclass
class Message:
    pass


@dataclass
class KillMessage(Message):
    pass


@dataclass
class DrawingQueueMessage(Message):
    sender: Actor[Message]


@dataclass
class PaperLocationMessage(Message):
    paper_location: Polygon


@dataclass
class AddPolygonMessage(DrawingQueueMessage):
    shape: Polygon


@dataclass
class UpdatePolygonMessage(DrawingQueueMessage):
    shape_id: int
    shape: Polygon


@dataclass
class CanvasIdMessage(Message):
    canvas_id: int


@dataclass
class DrawingQueueReferenceMessage(Message):
    drawing_queue: Queue[DrawingQueueMessage]

@dataclass
class InformationMessage(Message):
    name: str
    data: Any
