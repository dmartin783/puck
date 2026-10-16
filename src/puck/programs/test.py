import logging

# Relative import of the Actor class.
from ..actor import Actor
from ..messages import (
    AddPolygonMessage,
    DrawingQueueReferenceMessage,
    PaperLocationMessage,
    KillMessage,
    CanvasIdMessage,
    UpdatePolygonMessage,
)

# Every file has a logger.
logger = logging.getLogger(__name__)


def run(self: Actor) -> None:
    """Target function for an actor that just tracks a paper on the webcam.

    Args:
        self: the Actor whose target function is 'run'
    """
    logger.info("test_run has been called sucessfully")

    # First message should establish a reference to the drawing queue.
    first_message = self.recieve()
    assert type(first_message) is DrawingQueueReferenceMessage
    drawing_queue = first_message.drawing_queue

    # These are lists that are for this actor to hold onto whatever graphical objects and actors it spawns off.
    # TODO: use sentinels in future Python?
    paper_highlight_shape_id: int | None = None
    spawned_actors: list[Actor] = []

    # Handle incoming messages
    while True:
        message = self.recieve()
        logger.debug("run has been gotten {message}")
        match message:
            case KillMessage():
                break
            case PaperLocationMessage(paper_location):
                # Highlight the paper in white
                if paper_highlight_shape_id is None:
                    drawing_queue.put(AddPolygonMessage(self, paper_location))
                else:
                    drawing_queue.put(
                        UpdatePolygonMessage(
                            self, paper_highlight_shape_id, paper_location
                        )
                    )
            case CanvasIdMessage(canvas_id):
                # This is the id Tkinter has assigned to our paper shape
                paper_highlight_shape_id = canvas_id
            case _:
                logger.error(f"Received unexpected message {message}")

    logger.info("run finished")
