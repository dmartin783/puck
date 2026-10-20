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


@dataclass
class Slide:
    """
    Will hold all information relating to slides, including but by no
        means limited to their position, linked multimedia, and anything
        else I feel like adding
    """

    pos: Polygon
    id: int


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
    return Point(int(sum_x / 2), int(sum_y / 2))


def detect_slide_tags(
    frame: NDArray | None, homography_matrix: NDArray
) -> (
    tuple[Sequence[MatLike], MatLike] | bool
):  # tuple[Slide, Sequence[MatLike], list[int]] | bool | list[Slide] | Any:
    """Determines what papers are within a frame of video.

    Currently we are just looking for one paper at a time.
    This needs to be increased in newer implementations.

    Args:
        frame: a frame of the video feed.
        homography_matrix: a matrix to account for image distortion

    Returns:
        A list of papers and their associated list of four tags
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

    if ids is None:
        return False

    return (tags, ids)

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

    if len(ids_list) != 2:
        return False

    # averaged_paper = Polygon([average_pt(shape) for shape in transformed_tag_shapes])
    # ordered_averaged_paper, indices = ordered_rectangle(
    #    averaged_paper, averaged_paper[0]
    # )

    # TODO: Allow for multiple papers.
    ids_list = [int(i) for i in list(ids)]
    # ordered_ids_list = [ids_list[i] for i in indices]

    # -- Sequence of tags that make up a slide
    return [()]

    # return (Slide(pos=transformed_tag_shapes), tags, ids_list)
    # return [(ordered_averaged_paper, ordered_ids_list)]


def estimate_tag_pose(tag: MatLike, homography_matrix: NDArray):
    object_points: NDArray = np.array(
        [
            [-5 / 2, 5 / 2, 0],
            [5 / 2, 5 / 2, 0],
            [5 / 2, -5 / 2, 0],
            [-5 / 2, -5 / 2, 0],
        ]
    )

    return cv.solvePnP(
        object_points, tag, homography_matrix, None, flags=cv.SOLVEPNP_IPPE_SQUARE
    )


# def estimate_slide_outline(slide: Slide, homography_matrix: NDArray):
#     slide_tag_points: list[tuple[int, int]] = slide.pos.unwrap()
#     tag_corners: NDArray = np.array(
#         [[coord[0], coord[1]] for coord in slide_tag_points]
#     )
#     object_points: NDArray = np.array(
#         [
#             [-5 / 2, 5 / 2, 0],
#             [5 / 2, 5 / 2, 0],
#             [5 / 2, -5 / 2, 0],
#             [-5 / 2, -5 / 2, 0],
#         ]
#     )
#
#     return cv.solvePnP(
#         object_points,
#         tag_corners,
#         homography_matrix,
#         None,
#         flags=cv.SOLVEPNP_IPPE_SQUARE,
#     )


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

    slides = {}  # TODO :: Dict from id to slide object?
    ### --- SubProcesses --- #

    # Handle messages
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
            case _:
                pass

        if frame is None:
            continue
        slides = detect_slide_tags(frame, homography_matrix)
        logger.info(f"{type(slides) = }")
        if slides == False:
            continue

        assert type(slides) == tuple

        tags, ids = slides

        # assert type(slides) == list
        # logger.debug(f"{slides = }")

        # logger.debug(f"{slides[0].pos = }")

        rvecs, tvecs = [], []

        for tag in tags:
            # _, rvec, tvec = estimate_slide_outline(slide, homography_matrix)
            _, rvec, tvec = estimate_tag_pose(tag, homography_matrix)
            rvecs.append(rvec)
            tvecs.append(tvec)

        copy = frame.copy()
        # corners = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=np.float32)
        # # for pos in [slide.pos.unwrap() for slide in slides]:
        # # corners = [slide.pos.unwrap() for slide in slides]
        # ids = [slide.id for slide in slides]
        cv.aruco.drawDetectedMarkers(copy, tags, ids)
        for rvec, tvec in zip(rvecs, tvecs):
            cv.drawFrameAxes(copy, homography_matrix, None, rvec, tvec, 10, 2)
        cv.imwrite("PnP_output.png", copy)
        break
