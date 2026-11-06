import logging

import chorus
import cv2 as cv

from . import presentation
from .slide import Slide
from .slide_detection import get_slides

logger = logging.getLogger(__name__)


def run(spatial: chorus.Proxy, canvas):
    """
    Main thread in PPitW program, will look for slides
        (identified with two tags), and extra functionaity

    If program 3 is seen, the below will run.
    """

    logger.info("Running PPitW")

    # first_message: Message = self.recieve()
    # assert type(first_message) == DrawingQueueReferenceMessage
    # draw_queue: Queue[DrawingQueueMessage] = first_message.drawing_queue

    # third_message = self.recieve()
    # assert type(third_message) == DrawingQueueReferenceMessage
    # pres_view_queue = third_message.drawing_queue

    # # TODO :: This is very hacky
    # second_message: Message = self.recieve()  # This should be the homograohy matrix
    # assert type(second_message) == InformationMessage
    # assert second_message.name == "homography_matrix"
    # homography_matrix = second_message.data

    frame = None

    slides = {}  # TODO :: Dict from id to slide object?

    ### --- SubProcesses --- #
    # Spawn Presentation Thread
    # pres_draw_queue = Queue()
    # pres_thread = Actor(target=presentation.display)
    # pres_thread.start()
    # pres_thread.send(DrawingQueueReferenceMessage(drawing_queue=pres_draw_queue))

    # Handle messages
    while True:
        logger.info("Detecting slides")

        frame = chorus.receive(lambda m: m[0] == "frame")

        logger.info(f"{frame = }")
        frame = frame[1]
        # message = self.recieve()
        # if type(message) == InformationMessage and message.name == "frame":
        #     ...  # logger.info("Recieved frame")
        # else:
        #     ...  # logger.info(f"{message = }")

        # match message:
        #     case KillMessage():
        #         break
        #     case PaperLocationMessage(
        #         paper_location
        #     ):  # This is where the program paper is, not where a slide is.
        #         pass
        #     case InformationMessage("frame", data):
        #         frame = data
        #     case CanvasIdMessage(id):
        #         slide_id = id
        #     case _:
        #         pass

        if frame is None:
            continue

        slides = get_slides(frame)

        # # logger.info(f"{slides = }")

        if slides == False:
            logger.warning("Error in get_slides")
            continue

        assert type(slides) == list
        if len(slides) == 0:
            logger.info("no slides")
            continue

        # # logger.info("Sending slide")
        # # pres_view_queue.put(presentation.UpdateSlideMessage(self, slides[0]))

        for slide in slides:
            assert slide.slide_content is not None
            cv.imwrite(f"content_{slide.id}.png", slide.slide_content)
