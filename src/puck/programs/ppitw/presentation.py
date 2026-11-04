import logging

import cv2 as cv
from cv2.typing import MatLike

from puck.actor import Actor
from puck.messages import *
from puck.programs.ppitw.slide import Slide

logger = logging.getLogger(__name__)


@dataclass
class UpdateSlideMessage(DrawingQueueMessage):
    slide: Slide


def display(self: Actor):
    """
    This will handle drawing to the presentation canvas.
    Is spun up from the main ppitw thread.
    """

    # cv.namedWindow("Presentation View")

    # I thnik I want a unique draw queue for this wondow that is not drawing to the overlay view
    # first_message = self.recieve()
    # assert type(first_message) == DrawingQueueReferenceMessage
    # draw_queue: Queue[DrawingQueueMessage] = first_message.drawing_queue

    while True:
        logger.info(f"Running Presentation View")
        message = self.recieve()

        logger.debug(f"recieved {message = }")

        match message:
            case UpdateSlideMessage(_sender, slide):
                assert draw_slide(slide, "Presentation View")


def draw_slide(slide: Slide, window: str):
    if slide.slide_content is None:
        return False

    cv.imshow(window, slide.slide_content)
    return True
