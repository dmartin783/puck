import logging

import chorus
import cv2 as cv
from PIL import Image, ImageTk

from puck.geometry import Point, Polygon, ordered_rectangle

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
    active_id = None

    height = main_thread.ask.winfo_reqheight()
    width = main_thread.ask.winfo_reqwidth()

    active_slide_area, _indicies = ordered_rectangle(
        Polygon(
            [
                Point(width // 2, height // 2),
                Point(width, height // 2),
                Point(width, height),
                Point(width // 2, height),
            ]
        ),
        Point(width // 2, height // 2),
    )

    # Handle messages
    while True:
        logger.info("Drawing Active Area")
        if active_id is None:
            active_id = create_polygon(main_thread, active_slide_area)

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

        logger.info("Drawing Slide")

        # When do I show a slide? How do I know if its in the "active area"
        if slides[0].area_camera.is_in(active_slide_area):
            if pres_slide_img_id is None:
                pres_slide_img_id = create_image(main_thread, slides[0])
            else:
                update_image(main_thread, slides[0], pres_slide_img_id)

        for slide in slides:
            assert slide.slide_content is not None
            cv.imwrite(f"content_{slide.id}.png", slide.slide_content)
