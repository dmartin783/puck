# Setup tkinter early (necessary on MacOS).
import tkinter as tk
from dataclasses import dataclass
from queue import Queue
from tkinter import Canvas, Tk

from puck.programs.ppitw.presentation import PresView

base = Tk()
pres_topLevel = tk.Toplevel(base)

# Standard packages
import logging

import chorus

# External packages
import cv2 as cv
import numpy as np
import typer

# Local packages and modules
from .camera_processor import camera_setup, process_frame, show_camera_pov
from .spatial.spatial_model import PaperRecord, SpatialModel

# Global variables
logger = logging.getLogger(__name__)
TICK_LENGTH = 1
CANVAS_HEIGHT, CANVAS_WIDTH = 1080, 1920


def logging_setup(log: bool, log_level: int) -> None:
    """Configure the logger.

    Args:
        log: To log or not to log that is the question. (Turns on logging)
        log_level: log level to use with the logging package.
    """
    if log:
        logging.basicConfig(level=log_level)
        logger.info(f"Logging working at level {log_level}")


def tk_setup() -> None:
    """Sets up the tkinter window where it will draw graphics."""
    base.tk.call("tk", "scaling", 2.0)
    base.title("Tkinter Widget Size")
    base.wm_attributes("-fullscreen", True)

    # Assume 1080p double-monitor setup.
    base.geometry("1920x1080+0+-1080")


def canvas_setup(base: Tk) -> Canvas:
    """Set up a Tk.Canvas object on the base object.

    Args:
        base: an instance of Tk(), also commonly referred to as "root".
    """
    canvas = Canvas(base, height=CANVAS_HEIGHT, width=CANVAS_WIDTH, background="black")

    # Allows canvas to be seen by the user.
    canvas.pack()
    return canvas


def update(
    cam: cv.VideoCapture,
    canvas: Canvas,
    pres_view: PresView,
    homography_matrix: np.ndarray,
    spatial_model: SpatialModel,
) -> None:
    """Updates the system with event using visual input from the camera."""
    logger.debug("Called the Update Function")

    # Get a frame
    _, frame = cam.read()
    show_camera_pov(frame)

    assert frame is not None

    # Recognise and execute papers
    paper_records = [
        PaperRecord(program_encoding, ordered_shape_list)
        for ordered_shape_list, program_encoding in process_frame(
            frame, homography_matrix
        )
        if program_encoding
    ]

    spatial_model.tell.update(paper_records)

    aids = spatial_model.ask.get_encoding_to_actor_id()
    logger.info(f"{aids = }")
    if aids:
        ppitw_aid = aids[3]
        logger.info(f"{ppitw_aid = }")
        chorus.send(ppitw_aid, ("frame", frame))

    chorus.process_pending([canvas, pres_view])

    # Handle quitting
    if cv.waitKey(1) == ord("q"):
        logger.info("In the stopping condition")
        spatial_model.tell.kill_all()
        # If crashing due to thread conditions, please consider .join
        base.quit()

    # Run this function again shortly
    base.after(
        TICK_LENGTH,
        update,
        cam,
        canvas,
        pres_view,  # TODO :: HACKY
        homography_matrix,
        spatial_model,
    )


def start_puck(
    log: bool = True,
    log_level: int = 20,
    camera_id: int = 0,
    camera_calibration: str = "data/camera_cal.npy",
) -> None:
    """Runs the puck recognition system"""

    log = True
    camera_id = 1

    # Generic print to make sure that everything is working
    print("Hello from puck!")
    logging_setup(log, log_level)
    tk_setup()
    encoding_to_actor_id: dict[str, chorus.ActorID] = {}
    canvas = canvas_setup(base)
    print(chorus.me())
    pres_view = PresView(canvas_setup(pres_topLevel))

    logger.info(f"{pres_view = }")
    #    return
    homography_matrix, cam = camera_setup(camera_id, camera_calibration)
    spatial_model = chorus.spawn_server_with_proxy([SpatialModel(main=chorus.me())])

    chorus.process_pending([canvas, pres_view])

    pres_draw_queue = Queue()

    # Start the main update loop soons
    base.after(
        TICK_LENGTH,
        update,
        cam,
        canvas,
        pres_view,  # TODO :: HACKY
        homography_matrix,
        spatial_model,
    )

    # Start the event loop
    # Blocks until stopping condition is met
    base.mainloop()
    cam.release()
    cv.destroyAllWindows()


# @chorus.actors TODO: update so we can take in typer stuff.
def main_like():
    chorus.run_main(start_puck)


def main() -> None:
    """Runs the 'start_puck' function.

    Wrapped up in typer.run so it can act as a CLI and take in CL args."""
    typer.run(main_like)
