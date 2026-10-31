import logging
from dataclasses import dataclass
from typing import Sequence

import cv2 as cv
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

from puck.actor import Actor
from puck.calibration.calibrate import CalibrationInfo, calibrate
from puck.geometry import Point, Polygon, ordered_rectangle
from puck.messages import *  # TODO :: Avoid

from .slide import Slide
from .slide_detection import align_and_crop_slide, detect_slide_tags, get_slides

logger = logging.getLogger(__name__)

__all__ = ["run"]


def run(self: Actor):
    """
    Main thread in PPitW program, will look for slides
        (identified with two tags), and extra functionaity

    If program 3 is seen, the below will run.
    """

    logger.info("Running PPitW")

    first_message = self.recieve()
    assert type(first_message) == DrawingQueueReferenceMessage
    draw_queue = first_message.drawing_queue

    # TODO :: This is very hacky
    second_message = self.recieve()  # This should be the homograohy matrix
    assert type(second_message) == InformationMessage
    assert second_message.name == "homography_matrix"
    homography_matrix = second_message.data

    frame = None

    slide_id = None
    addPolySent = False

    slides = {}  # TODO :: Dict from id to slide object?
    ### --- SubProcesses --- #

    # Handle messages
    count = 0
    while True:
        logger.info("Detecting slides")

        message = self.recieve()
        if type(message) == InformationMessage and message.name == "frame":
            logger.info("Recieved frame")
        else:
            logger.info(f"{message = }")

        match message:
            case KillMessage():
                break
            case PaperLocationMessage(
                paper_location
            ):  # This is where the program paper is, not where a slide is.
                pass
            case InformationMessage("frame", data):
                frame = data
            case CanvasIdMessage(id):
                slide_id = id
            case _:
                pass

        if frame is None:
            continue

        slides = get_slides(frame)

        logger.info(f"{slides = }")

        if slides == False:
            continue

        assert type(slides) == list

        for slide in slides:
            assert slide.slide_content is not None
            cv.imwrite(f"content_{slide.id}.png", slide.slide_content)
