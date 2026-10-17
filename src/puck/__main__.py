# Setup tkinter early (necessary on MacOS).
from tkinter import Canvas, Tk

base = Tk()

# Standard packages
import importlib
import json
import logging
from datetime import datetime
from queue import Queue

# External packages
import cv2 as cv
import numpy as np
import typer
from numpy.typing import NDArray

from .actor import Actor
from .calibration.calibrate import calibrate

# Local packages and modules
from .geometry import Point, Polygon, ordered_rectangle
from .messages import *

# Global variables
logger = logging.getLogger(__name__)
DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_16H5)
APRIL_LIMIT = 4
TICK_LENGTH = 16
CANVAS_HEIGHT, CANVAS_WIDTH = 1080, 1920
CAMERA_PERSPECTIVE_WINDOW_NAME = "Camera perspective"


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


def canvas_setup(base: Tk) -> Canvas:
    """Set up a Tk.Canvas object on the base object.

    Args:
        base: an instance of Tk(), also commonly referred to as "root".
    """
    canvas = Canvas(base, height=CANVAS_HEIGHT, width=CANVAS_WIDTH, background="black")

    # Allows canvas to be seen by the user.
    canvas.pack()
    return canvas


def camera_setup(camera_id: int) -> cv.VideoCapture:
    """Set up your camera to get a videofeed input.

    Args:
        camera_id (int): camera id, typically 0 or 1

    Returns:
        cv.VideoCapture: video feed
    """
    cam = cv.VideoCapture(camera_id)

    # Get one frame to verify camera is working (will throw if broken)
    _, frame = cam.read()
    cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    return cam


def camera_perspective_window_setup(window_name: str) -> None:
    """Set up a window that will show you what the camera is seeing."""
    cv.namedWindow(
        window_name,
        cv.WINDOW_FREERATIO,
    )
    cv.moveWindow(window_name, 0, 300)
    cv.resizeWindow(window_name, 600, 500)


def average_pt(tag: Polygon) -> Point:
    """Takes the AprilTag shape and finds a point in the middle of it."""
    sum_x = 0
    sum_y = 0
    for x, y in tag:
        sum_x += x
        sum_y += y
    return Point(int(sum_x / 4), int(sum_y / 4))


def transform_point(point: Point, homography_matrix: np.ndarray) -> Point:
    """Transforms point from a detected point to a point in projection space."""
    assert homography_matrix.shape == (3, 3)

    arr = np.array([[[point.x, point.y]]], dtype=np.float32)
    assert arr.shape == (1, 1, 2)

    transformed = cv.perspectiveTransform(arr, homography_matrix)
    assert transformed.shape == (1, 1, 2)

    transformed_coords = [int(c) for c in list(transformed[0, 0])]
    return Point(transformed_coords[0], transformed_coords[1])


def transform_shape(shape: Polygon, homography_matrix: np.ndarray) -> Polygon:
    """Transforms a shapes points from detected to projected."""
    points = []
    for point in shape:
        points.append(transform_point(point, homography_matrix))
    return Polygon(points)


def detect_paper_tags(
    frame: np.ndarray, homography_matrix: np.ndarray
) -> list[tuple[Polygon, list[int]]]:
    """Determines what papers are within a frame of video.

    Currently we are just looking for one paper at a time.
    This needs to be increased in newer implementations.

    Args:
        frame: a frame of the video feed.
        homography_matrix: a matrix to account for image distortion

    Returns:
        A list of papers and their associated list of four tags
    """
    april_tag_detector = cv.aruco.ArucoDetector(dictionary=DICT)
    tags, ids, _ = april_tag_detector.detectMarkers(frame)
    if ids is None:
        return []
    else:
        # Check for invalid/bad ids
        bad_indices = [i for i in range(len(ids)) if ids[i] > APRIL_LIMIT]
        if len(bad_indices) > 0:
            # Save problematic frame
            bad_ids = [ids[i] for i in bad_indices]
            logger.warning(f"Invalid AprilTag Value(s): {bad_ids} detected")
            copy = cv.aruco.drawDetectedMarkers(frame, tags, ids)
            cv.imwrite(f"invalid_AprilTag_{datetime.now().isoformat()}.png", copy)

        # Assemble required data in lists, filtering out any bad IDs
        ids_list = [int(ids[i]) for i in range(len(ids)) if i not in bad_indices]
        tag_shapes: list[Polygon] = [
            Polygon.from_array(tags[i][0])
            for i in range(len(tags))
            if i not in bad_indices
        ]
        transformed_tag_shapes = [
            transform_shape(shape, homography_matrix) for shape in tag_shapes
        ]

        assert len(ids_list) == len(transformed_tag_shapes)

        if len(ids_list) == 4:
            averaged_paper = Polygon(
                [average_pt(shape) for shape in transformed_tag_shapes]
            )
            ordered_averaged_paper, indices = ordered_rectangle(
                averaged_paper, averaged_paper[0]
            )

            # TODO: Allow for multiple papers.
            ids_list = [int(i) for i in list(ids)]
            ordered_ids_list = [ids_list[i] for i in indices]

            return [(ordered_averaged_paper, ordered_ids_list)]
        else:
            return []


def tags_to_paper_encoding(tags: list[int]) -> int | None:
    """Transforms the 4 AprilTags IDs on a paper into that paper's encoding.

    Args:
        tags list[int]: a list of the April Tag ids given

    Returns:
        int: the program encoding as an integer

    """
    if len(tags) != 4:
        logger.warning(f"{tags} length is not 4")
        return None

    if APRIL_LIMIT not in tags:
        logger.warning(f"{APRIL_LIMIT} not found in {tags}")
        return None

    for t in tags:
        if t > APRIL_LIMIT:
            logger.warning(f"{t} is greater than the april limit")
            return None

    # Reorder based on the position of the APRIL_LIMIT tag
    pos = tags.index(APRIL_LIMIT)
    ordered_tags = tags[pos + 1 :] + tags[:pos]
    print(f"Unordered Tags: {tags}")
    print(f"Ordered Tags: {ordered_tags}")

    # Convert into int by using tags as base-APRIL_LIMIT digits
    program_encoding = sum(tag * APRIL_LIMIT**i for i, tag in enumerate(ordered_tags))
    assert type(program_encoding) is int

    logger.debug(f"Raw id: {tags} to interpreted id: {program_encoding}")
    return program_encoding


def create_and_update_actors(
    program_encoding: int,
    current_coords: Polygon,
    encoding_to_actor: dict[str, Actor],
    program_lookup: list[str],
    drawing_queue: Queue[Message],
) -> None:
    """Creates or updates Actors associated with a given program encoding"""
    if program_encoding >= len(program_lookup) or program_encoding < 0:
        print(f"There is no associated program with the encoding: {program_encoding}")
        return

    # TODO: use this to load modules dynamically
    module_name = "puck.programs." + program_lookup[program_encoding]
    module = importlib.import_module(module_name)

    if not (t := encoding_to_actor.get(str(program_encoding))):
        # This program is not yet running
        t = Actor(target=module.run)
        encoding_to_actor[str(program_encoding)] = t
        t.start()
        # TODO: pass as environment on construction instead
        t.send(DrawingQueueReferenceMessage(drawing_queue))

    t.send(PaperLocationMessage(current_coords))


def draw(drawing_queue: Queue[Message], canvas: Canvas) -> None:
    """Draws the commands on the drawing queue onto the canvas."""
    if not drawing_queue.empty():  # TODO: change to 'while'
        message = drawing_queue.get()
        logger.debug(f"draw loop got message {message}")

        assert type(message) is DrawingQueueMessage

        match message:
            case AddPolygonMessage(sender, shape):
                fill = "white"
                outline = "blue"
                width = 2
                canvas_id = canvas.create_polygon(
                    shape.unwrap(),
                    fill=fill,
                    outline=outline,
                    width=width,
                )
                sender.send(CanvasIdMessage(canvas_id))
            case UpdatePolygonMessage(_, shape_id, shape):
                canvas.coords(shape_id, shape.flat_unwrap())
            case _:
                logger.error(f"Unexpected message {message}")

        logger.debug("drawing loop finished")
    canvas.pack()


def update(
    cam: cv.VideoCapture,
    encoding_to_actor: dict[str, Actor],
    program_lookup: list[str],
    drawing_queue: Queue[Message],
    canvas: Canvas,
    window_name: str,
    homography_matrix: np.ndarray,
) -> None:
    """Updates the system with event using visual input from the camera."""
    logger.debug("Called the Update Function")

    # Get a frame
    _, frame = cam.read()
    cv.imshow(window_name, frame)
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # Recognise and execute papers
    for coords, tags in detect_paper_tags(frame, homography_matrix):
        program_encoding = tags_to_paper_encoding(tags)
        if program_encoding:
            logger.debug(f"Saw program encoding, {program_encoding}")
            create_and_update_actors(
                program_encoding,
                coords,
                encoding_to_actor,
                program_lookup,
                drawing_queue,
            )

    draw(drawing_queue, canvas)

    # Handle quitting
    if cv.waitKey(1) == ord("q"):
        logger.info("In the stopping condition")
        for encoding, a in encoding_to_actor.items():
            a.end()
            logger.info(f"encoding asscoiated is : {encoding}")
            logger.info("Got past the end, onto Join now")
            a.join()
        logger.info("finished the joining and ending")
        base.quit()

    # Run this function again shortly
    base.after(
        TICK_LENGTH,
        update,
        cam,
        encoding_to_actor,
        program_lookup,
        drawing_queue,
        canvas,
        window_name,
        homography_matrix,
    )


def start_puck(
    log: bool = False,
    log_level: int = 0,
    camera_id: int = 0,
    program_lookup_file: str = "data/program_lookup.json",
) -> None:
    """Runs the puck recognition system"""

    # Generic print to make sure that everything is working
    print("Hello from puck!")
    logging_setup(log, log_level)

    tk_setup()
    program_lookup = load_program_names(program_lookup_file)
    encoding_to_actor: dict[str, Actor] = {}
    canvas = canvas_setup(base)
    drawing_queue: Queue[Message] = Queue()
    calibration_info = calibrate()
    cam = camera_setup(camera_id)
    camera_perspective_window_setup(CAMERA_PERSPECTIVE_WINDOW_NAME)
    homography_matrix = calibration_info.camera_to_projector_homography
    if homography_matrix.all() == None:
        logger.fatal(msg="Homography Matrix failed to generate, nothing will work.")

    # Start the main update loop soons
    base.after(
        TICK_LENGTH,
        update,
        cam,
        encoding_to_actor,
        program_lookup,
        drawing_queue,
        canvas,
        CAMERA_PERSPECTIVE_WINDOW_NAME,
        homography_matrix,
    )

    # Start the event loop
    # Blocks until stopping condition is met
    base.mainloop()
    cam.release()
    cv.destroyAllWindows()


def main() -> None:
    """Runs the 'start_puck' function.

    Wrapped up in typer.run so it can act as a CLI and take in CL args."""
    typer.run(start_puck)
