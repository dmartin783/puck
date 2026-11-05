# Setup tkinter early (necessary on MacOS).
from tkinter import Tk, Canvas

base = Tk()

# Standard packages
import importlib
import json
import logging
from pickle import UnpicklingError
from queue import Queue

# External packages
import cv2 as cv
import numpy as np
import typer
from datetime import datetime
from numpy.typing import NDArray
import pprint
import chorus

# Local packages and modules
from .geometry import Point, Polygon, ordered_rectangle
from .calibration.calibrate import calibrate, CalibrationInfo
from .camera_processor import camera_setup, show_camera_pov,process_frame
from .spatial_model import SpatialModel, PaperRecord

# Global variables
logger = logging.getLogger(__name__)
TICK_LENGTH = 16
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
    canvas = Canvas(
        base, height=CANVAS_HEIGHT, width=CANVAS_WIDTH, background="black"
    )

    # Allows canvas to be seen by the user.
    canvas.pack()
    return canvas


def create_and_update_actors(
    program_encoding: int,
    ordered_shape_list: list[Polygon],
    encoding_to_actor_id: dict[str, chorus.ActorID],
    program_lookup: list[str],
) -> None:
    """Creates or updates Actors associated with a given program encoding"""
    if program_encoding >= len(program_lookup) or program_encoding < 0:
        print(
            f"There is no associated program with the encoding: {program_encoding}"
        )
        return

    module_name = "puck.programs." + program_lookup[program_encoding]
    module = importlib.import_module(module_name)

    if not (aid := encoding_to_actor_id.get(str(program_encoding))):
        # This program is not yet running
        aid= chorus.spawn(module.run, chorus.Proxy(chorus.me()))
        encoding_to_actor_id[str(program_encoding)] = aid

    for shape in ordered_shape_list:
        print(shape)
        


# def draw(drawing_queue: Queue[Message], canvas: Canvas) -> None:
#     """Draws the commands on the drawing queue onto the canvas."""
#     if not drawing_queue.empty():  # TODO: change to 'while'
#         message = drawing_queue.get()
#         logger.debug(f"draw loop got message {message}")

#         assert issubclass(type(message), DrawingQueueMessage)

#         match message:
#             case AddPolygonMessage(sender, shape):
#                 fill = "white"
#                 outline = "blue"
#                 width = 2
#                 canvas_id = canvas.create_polygon(
#                     shape.unwrap(),
#                     fill=fill,
#                     outline=outline,
#                     width=width,
#                 )
#                 sender.send(CanvasIdMessage(canvas_id))
#             case UpdatePolygonMessage(_, shape_id, shape):
#                 canvas.coords(shape_id, shape.flat_unwrap())
#             case _:
#                 logger.error(f"Unexpected message {message}")

#         logger.debug("drawing loop finished")
#     canvas.pack()


def update(
    cam: cv.VideoCapture,
    encoding_to_actor_id: dict[str,chorus.ActorID],
    canvas: Canvas,
    homography_matrix: np.ndarray,
    spatial_model: SpatialModel
) -> None:
    """Updates the system with event using visual input from the camera."""
    logger.debug("Called the Update Function")

    # Get a frame
    _, frame = cam.read()
    show_camera_pov(frame)
    
#     # Recognise and execute papers
    paper_records = [PaperRecord(program_encoding,ordered_shape_list) 
                     for ordered_shape_list, program_encoding in 
                     process_frame(frame, homography_matrix) if program_encoding]
    spatial_model.tell.update(paper_records)

    # draw(drawing_queue, canvas) ## LOOK INTO

#     # Handle quitting
    if cv.waitKey(1) == ord("q"):
        logger.info("In the stopping condition") 
        spatial_model.tell.dummy()
        spatial_model.tell.kill_all()
        spatial_model.kill()
        # If crashing due to thread conditions, please consider .join
        base.quit()

#     # Run this function again shortly
    base.after(
        TICK_LENGTH,
        update,
        cam,
        encoding_to_actor_id,
        canvas,
        homography_matrix,
        spatial_model
    )

def start_puck(
    log: bool = True,
    log_level: int = 20,
    camera_id: int = 0,
    camera_calibration: str = "data/camera_cal.npy",
) -> None:
    """Runs the puck recognition system"""

    # Generic print to make sure that everything is working
    print("Hello from puck!")
    logging_setup(log, log_level)
    tk_setup()
    encoding_to_actor_id: dict[str, chorus.ActorID ] = {}
    canvas = canvas_setup(base)
    print(chorus.me())
    homography_matrix,cam = camera_setup(camera_id,camera_calibration)
    spatial_model = chorus.spawn_server_with_proxy([SpatialModel()])




    # Start the main update loop soons
    base.after(
        TICK_LENGTH,
        update,
        cam,
        encoding_to_actor_id,
        canvas,
        homography_matrix,
        spatial_model
    )

    # Start the event loop
    # Blocks until stopping condition is met
    base.mainloop()
    cam.release()
    cv.destroyAllWindows()

# @chorus.actors
def main_like():
    chorus.run_main(start_puck)

def main() -> None:
    """Runs the 'start_puck' function.

    Wrapped up in typer.run so it can act as a CLI and take in CL args."""
    typer.run(main_like)
