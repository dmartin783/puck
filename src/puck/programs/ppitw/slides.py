import logging

import chorus
import cv2 as cv
from PIL import Image, ImageTk

from ...spatial.api import (
    corners,
    create_image,
    create_polygon,
    update_image,
    update_polygon,
)
from . import presentation
from .slide import Slide
from .slide_detection import get_slides

logger = logging.getLogger(__name__)


def run(spatial: chorus.Proxy, main_thread):
    """
    Main thread in PPitW program, will look for slides
        (identified with two tags), and extra functionaity

    If program 3 is seen, the below will run.
    """

    logger.info("Running PPitW")

    pres_slide_img_id = None

    # Handle messages
    while True:
        logger.info("Detecting slides")

        _, frame = chorus.receive(lambda m: m[0] == "frame")

        if frame is None:
            continue

        slides = get_slides(frame)

        if slides == False:
            logger.warning("Error in get_slides")
            continue

        assert type(slides) == list
        if len(slides) == 0:
            logger.info("no slides")
            continue

        logger.info("adding image")

        if pres_slide_img_id is None:
            pres_slide_img_id = create_image(main_thread, slides[0])
        else:
            update_image(main_thread, slides[0], pres_slide_img_id)

        for slide in slides:
            assert slide.slide_content is not None
            cv.imwrite(f"content_{slide.id}.png", slide.slide_content)
