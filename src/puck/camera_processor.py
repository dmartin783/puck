# Standard packages
import logging

# External Packages
import cv2 as cv
from pickle import UnpicklingError
import numpy as np
from datetime import datetime

# Local packages and modules
from .geometry import Point, Polygon, ordered_rectangle
from .calibration.calibrate import calibrate, CalibrationInfo

# Global Variables
logger = logging.getLogger(__name__)
DICT = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_16H5)
APRIL_LIMIT = 4

CAMERA_PERSPECTIVE_WINDOW_NAME = "Camera perspective"


def camera_perspective_window_setup(window_name: str) -> None:
    """Set up a window that will show you what the camera is seeing."""
    cv.namedWindow(
        window_name,
        cv.WINDOW_FREERATIO,
    )
    cv.moveWindow(window_name, 0, 300)
    cv.resizeWindow(window_name, 600, 500)


def show_camera_pov(frame):
    cv.imshow(CAMERA_PERSPECTIVE_WINDOW_NAME, frame)


def camera_setup(
    camera_id: int, camera_calibration: str = "data/camera_cal.npy"
) -> cv.VideoCapture:
    """Set up your camera to get a videofeed input.

        Args:
            camera_id (int): camera id, typically 0 or 1
    s
        Returns:
            cv.VideoCapture: video feed
    """
    cam = cv.VideoCapture(camera_id)

    # Get one frame to verify camera is working (will throw if broken)
    _, frame = cam.read()
    cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    camera_perspective_window_setup(CAMERA_PERSPECTIVE_WINDOW_NAME)
    try:
        calibration_info = CalibrationInfo.load(camera_calibration)
    except OSError:
        calibration_info = calibrate()
        calibration_info.save(camera_calibration)
    except UnpicklingError:
        logger.fatal(
            f"{camera_calibration} does not contain a calibration matrix.\
            Terminating."
        )
        return
    homography_matrix = calibration_info.camera_to_projector_homography
    if homography_matrix.all() == None:
        logger.fatal("Homography Matrix failed to generate. Terminating.")

    return homography_matrix, cam


def average_pt(tag: Polygon) -> Point:
    """Takes the AprilTag shape and finds a point in the middle of it."""
    sum_x = 0
    sum_y = 0
    for x, y in tag:
        sum_x += x
        sum_y += y
    return Point(int(sum_x / 4), int(sum_y / 4))


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
            cv.imwrite(
                f"invalid_AprilTag_{datetime.now().isoformat()}.png", copy
            )

        # Assemble required data in lists, filtering out any bad IDs
        ids_list = [
            int(ids[i]) for i in range(len(ids)) if i not in bad_indices
        ]
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
            average_to_shape = {}
            average_paper_list = []
            for shape in transformed_tag_shapes:
                avg = average_pt(shape)
                average_paper_list.append(avg)
                average_to_shape.update({avg: shape})
            averaged_paper = Polygon(average_paper_list)

            ordered_averaged_paper, indices = ordered_rectangle(
                averaged_paper, averaged_paper[0]
            )

            average_point_list = [
                Point(x, y) for (x, y) in ordered_averaged_paper.unwrap()
            ]

            ordered_shape_list = [None] * 4
            for avg, shape in average_to_shape.items():
                i = average_point_list.index(avg)
                ordered_shape_list[i] = shape

            # TODO: Allow for multiple papers.
            ids_list = [int(i) for i in list(ids)]
            ordered_ids_list = [ids_list[i] for i in indices]
            return [(ordered_shape_list, ordered_ids_list)]
        else:
            return []


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
    logger.debug(f"Unordered Tags: {tags}")
    logger.debug(f"Ordered Tags: {ordered_tags}")

    # Convert into int by using tags as base-APRIL_LIMIT digits
    program_encoding = sum(
        tag * APRIL_LIMIT**i for i, tag in enumerate(ordered_tags)
    )
    assert type(program_encoding) is int

    logger.debug(f"Raw id: {tags} to interpreted id: {program_encoding}")
    return program_encoding


def process_frame(frame, homography_matrix):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    return [
        (ordered_shape_list, tags_to_paper_encoding(tags))
        for ordered_shape_list, tags in detect_paper_tags(
            frame, homography_matrix
        )
    ]
