import logging
from dataclasses import dataclass
from typing import Sequence

import cv2 as cv
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

from ..actor import Actor
from ..calibration.calibrate import CalibrationInfo, calibrate
from ..geometry import Point, Polygon, ordered_rectangle
from ..messages import *  # TODO :: Avoid

logger = logging.getLogger(__name__)

SLIDE_TAGS = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_25H9)
N_TAGS = 3


@dataclass
class Slide:
    """
    Will hold all information relating to slides, including but by no
        means limited to their position, linked multimedia, and anything
        else I feel like adding
    """

    tags: Sequence[MatLike]  # Both tags that make up the slide corners
    ids: MatLike  # Corrosponding ids


def transform_point(point: Point, homography_matrix: NDArray) -> Point:
    """Transforms point from a detected point to a point in projection space."""
    assert homography_matrix.shape == (3, 3)

    arr = np.array([[[point.x, point.y]]], dtype=np.float32)
    assert arr.shape == (1, 1, 2)

    transformed = cv.perspectiveTransform(arr, homography_matrix)
    assert transformed.shape == (1, 1, 2)

    transformed_coords = [int(c) for c in list(transformed[0, 0])]
    return Point(transformed_coords[0], transformed_coords[1])


def transform_shape(shape: Polygon, homography_matrix: NDArray) -> Polygon:
    """Transforms a shapes points from detected to projected."""
    points = []
    for point in shape:
        points.append(transform_point(point, homography_matrix))
    return Polygon(points)


def average_pt(tag: Polygon) -> Point:
    """Takes the AprilTag shape and finds a point in the middle of it."""
    sum_x = 0
    sum_y = 0
    for x, y in tag:
        sum_x += x
        sum_y += y
    return Point(int(sum_x / 4), int(sum_y / 4))


def detect_slide_tags(
    frame: NDArray | None, homography_matrix: NDArray
) -> (
    list[Slide] | bool
):  # tuple[Slide, Sequence[MatLike], list[int]] | bool | list[Slide] | Any:
    """Determines what papers are within a frame of video.

    Currently we are just looking for one paper at a time.
    This needs to be increased in newer implementations.

    Args:
        frame: a frame of the video feed.
        homography_matrix: a matrix to account for image distortion

    Returns:
        A list of slides and their associated list of three tags
    """
    if frame is None:
        logger.debug(f"{frame = }")
        return False

    april_tag_detector = cv.aruco.ArucoDetector(dictionary=SLIDE_TAGS)
    cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

    tags, ids, _ = april_tag_detector.detectMarkers(
        frame,
    )  # tags an Nx4 array where N is the number of tags
    logger.debug(f"{ids = }")

    if ids is None or len(ids) != N_TAGS:
        return False

    return [Slide(tags, ids)]

    # # Check for invalid/bad ids
    # bad_indices = [i for i in range(len(ids)) if ids[i] > APRIL_LIMIT]
    # if len(bad_indices) > 0:
    #     # Save problematic frame
    #     bad_ids = [ids[i] for i in bad_indices]
    #     logger.warning(f"Invalid AprilTag Value(s): {bad_ids} detected")
    #     copy = cv.aruco.drawDetectedMarkers(frame, tags, ids)
    #     cv.imwrite(f"invalid_AprilTag_{datetime.now().isoformat()}.png", copy)

    # Assemble required data in lists, filtering out any bad IDs
    ids_list = [int(ids[i]) for i in range(len(ids))]  # if i not in bad_indices]
    tag_shapes: list[Polygon] = [
        Polygon.from_array(tags[i][0])
        for i in range(len(tags))  # if i not in bad_indices
    ]
    transformed_tag_shapes = [
        transform_shape(shape, homography_matrix) for shape in tag_shapes
    ]  # -- list of all tags (polygons)

    assert len(ids_list) == len(transformed_tag_shapes)

    if len(ids_list) != N_TAGS:
        return False

    # averaged_paper = Polygon([average_pt(shape) for shape in transformed_tag_shapes])
    # ordered_averaged_paper, indices = ordered_rectangle(
    #    averaged_paper, averaged_paper[0]
    # )

    # TODO: Allow for multiple papers.
    ids_list = [int(i) for i in list(ids)]
    # ordered_ids_list = [ids_list[i] for i in indices]


def detect_slide_area(slide: Slide, homography_matrix: NDArray) -> Polygon | None:
    tags = slide.tags
    if len(tags) != N_TAGS:
        return None

    ids = slide.ids
    tags = slide.tags

    tags_pos: list[tuple[int, Point]] = []

    for id, tag in zip(ids, tags):
        logger.info(tag)
        tag = tag[
            0
        ]  # -- tags are given in a wierd format wrapped in an outer, single element array TODO :: fix this in an elegant manner
        tag_shape = Polygon.from_array(tag)
        transformed_tag_shape = transform_shape(tag_shape, homography_matrix)
        tags_pos.append((id, average_pt(transformed_tag_shape)))

    top_left = [(tag_pos[0], tag_pos[1]) for tag_pos in tags_pos if tag_pos[0] >= 1]
    if len(top_left) != 1:
        return None
    p0 = top_left[0][1]

    others = [tag_pos for tag_pos in tags_pos if tag_pos[0] == 0]
    if len(others) != 2:
        return None
    logger.info(f"{top_left = } {others = }")

    p1 = others[0][1]
    p2 = others[1][1]

    alpha = p1 - p0  # vector from p0 (top left) to p1 (either other tag, doesnt matter)

    p3 = p1 + alpha

    return Polygon([p0, p1, p2, p3])


def estimate_tag_pose(tag: MatLike, homography_matrix: NDArray):
    square_length = 5
    object_points: NDArray = np.array(
        [
            [-square_length / 2, square_length / 2, 0],
            [square_length / 2, square_length / 2, 0],
            [square_length / 2, -square_length / 2, 0],
            [-square_length / 2, -square_length / 2, 0],
        ]
    )

    ret, rvec, tvec = cv.solvePnP(
        object_points,
        tag,
        homography_matrix,
        None,
        flags=cv.SOLVEPNP_IPPE_SQUARE,
    )

    logger.info(rvec)

    return ret, rvec, tvec


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
        slides = detect_slide_tags(frame, homography_matrix)
        if slides == False:
            continue

        assert type(slides) == list

        slide = slides[0]
        logger.info(f"{slide.ids = }")

        area = detect_slide_area(slides[0], homography_matrix)
        logger.info(f"{area = }")
        if area is None:
            continue

        if slide_id is None or not addPolySent:
            addPolySent = True
            draw_queue.put(AddPolygonMessage(self, area))
            logger.info("Added slide")
        elif slide_id is not None:
            draw_queue.put(UpdatePolygonMessage(self, slide_id, area))
            logger.info("Updated slide")

        # tags, ids = slides[0].tags, slides[0].ids

        # rvecs, tvecs = [], []

        # for tag in tags:
        #     _, rvec, tvec = estimate_tag_pose(tag, homography_matrix)
        #     rvecs.append(rvec)
        #     tvecs.append(tvec)

        # copy = frame.copy()
        # cv.aruco.drawDetectedMarkers(copy, tags, ids)
        # # for rvec, tvec in zip(rvecs, tvecs):
        # cv.drawFrameAxes(copy, homography_matrix, None, rvecs[0], tvecs[0], 10, 2)
        # cv.imwrite(f"PnP_output.png_{count}", copy)
        # # count += 1
