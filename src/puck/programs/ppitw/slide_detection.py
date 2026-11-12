import logging

import cv2 as cv
import numpy as np
from cv2.typing import MatLike
from numpy.typing import NDArray

from puck.geometry import Point, Polygon, ordered_rectangle

from .slide import Slide

logger = logging.getLogger(__name__)
logger.level = 0

SLIDE_TAGS = cv.aruco.getPredefinedDictionary(cv.aruco.DICT_APRILTAG_25H9)
N_TAGS = 3

__all__ = ["get_slides"]


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


def detect_slide_tags(frame: NDArray | None) -> list[Slide] | bool:
    """Determines what slides are within a frame of video.

    Currently we are just looking for one slide at a time.
    This needs to be increased in newer implementations.

    Args:
        frame: a frame of the video feed.

    Returns:
        A list of slides
    """
    if frame is None:
        logger.debug(f"{frame = }")
        return False

    april_tag_detector = cv.aruco.ArucoDetector(dictionary=SLIDE_TAGS)
    cv.cvtColor(frame, cv.COLOR_RGB2GRAY)

    tags, ids, _ = april_tag_detector.detectMarkers(
        frame,
    )  # tags an Nx4 array where N is the number of tags
    # logger.debug(f"{ids = }")

    if ids is None or len(ids) % N_TAGS != 0:
        logger.warning("Not enough IDs")
        return False

    # ids[i] <=> tag_shapes[i]
    tag_shapes = [Polygon.from_array(tag[0]) for tag in tags]
    # ids[i] <=> ordered_tag_shapes[i]
    ordered_tag_shapes_and_indicies: list[tuple[Polygon, list[int]]] = [
        ordered_rectangle(tag_shape, tag_shape.points[0]) for tag_shape in tag_shapes
    ]

    ordered_tag_shapes: list[Polygon] = [
        shape for shape, indicies in ordered_tag_shapes_and_indicies
    ]

    # TODO :: something about ids being included in `ordered_tag_shapes`
    #         use to differentiate slides.

    if len(tags) % N_TAGS != 0:
        logger.warning("Not enough tags")
        return False

    tags_pos: list[tuple[int, Point]] = []

    for id, shape in zip(ids, ordered_tag_shapes):
        tags_pos.append((id, average_pt(shape)))

    # top left tag is defined as being only tag in slide with id > 0
    top_left = [(tag_pos[0], tag_pos[1]) for tag_pos in tags_pos if tag_pos[0] >= 1]

    # every other tag in the slide should have an id of 0
    others = [tag_pos for tag_pos in tags_pos if tag_pos[0] == 0]
    # logger.info(f"{top_left = } {others = }")

    # should figure out here which point belongs to which slide.

    # TODO :: Figure out better epsilon
    ANGLE_EPSILON = 10  # degrees

    valid_rects = []
    for tl_id, tl in top_left:
        for o1_id, o1 in others.copy():
            for o2_id, o2 in others.copy():
                # logger.debug(f"{o1 = } {o2 = }")
                # logger.debug(f"{others = }")
                if o1 == o2:
                    # logger.info("o1 is o2, skipping")
                    continue
                edge_one = o1 - tl
                edge_two = o2 - tl
                # from geometric definition of dot product
                angle = np.arccos(
                    np.dot(edge_one.to_numpy(), edge_two.to_numpy())
                    / (
                        np.linalg.norm(edge_one.to_numpy())
                        * np.linalg.norm(edge_two.to_numpy())
                    )
                )

                # logger.info(f"{np.rad2deg(angle) = }")

                if abs(np.rad2deg(angle) - 90) < ANGLE_EPSILON:
                    valid_rects.append((tl_id, tl, o1, o2))
                    others.remove((o1_id, o1))
                    others.remove((o2_id, o2))
                    # -- Once i find valid rect for given top left, stop looking
                    break
            break

    # logger.info(f"{valid_rects = }")

    slides = []
    for rect in valid_rects:
        id, p0, p1, p2 = rect

        alpha = (
            p1 - p0
        )  # vector from p0 (top left) to p1 (either other tag, doesnt matter)

        p3 = p2 + alpha

        ordered_area, _new_points_order = ordered_rectangle(
            Polygon([p0, p1, p2, p3]), p0
        )

        slides.append(Slide(id, area_camera=ordered_area))

    # area_camera now contains clockwise ordered list of all points that make up area from top left.
    # points are center points of tags + one that was inferred as being the center of the excluded tag.
    return slides


# TODO :: TEST THIS !!!!!
def align_and_crop_slide(slide: Slide, frame: MatLike) -> Slide:
    slide_area = slide.area_camera
    if type(slide_area) != Polygon:
        raise ValueError(f"Slide area must have been found, area is {slide_area}")

    sketch = None
    if logger.level > 0:
        sketch = np.copy(frame)

    corners = slide_area.points  # Points clockwise from top left corner

    # -- These 2 define the top edge (i.e. +X direction)
    p0 = corners[0]
    p1 = corners[1]

    # logger.info(f"{p0.to_numpy()[0:2] = } {p1.to_numpy()[0:2] = }")

    if logger.level > 0:
        assert sketch is not None
        cv.line(sketch, list(p0.to_numpy()[0:2]), list(p1.to_numpy()[0:2]), 0, 3)
    size = (1920, 1080)

    d = p1 - p0  # top edge
    d_norm = d.to_numpy() / np.linalg.norm(d.to_numpy())

    corners = np.array([p.to_numpy() for p in corners])
    # logger.info(f"{corners = }")
    corners = np.apply_along_axis(lambda p: p[0:2], 1, corners)
    # logger.info(f"{corners = }")

    angle = np.arctan2(d_norm[1], d_norm[0])  # np.dot(d_norm, [1, 0, 0]))
    # logger.info(f"{np.rad2deg(angle) = }")
    # -- negative to correct angle TODO :: convince self this is correct

    center = [int(x) for x in average_pt(slide_area).to_numpy()]

    # logger.info(f"{center = }")

    # logger.info(f"{frame.shape = }")

    # TODO :: Angle maybe -ve angle to rotate back to 0?
    rotate_translate_mat = cv.getRotationMatrix2D(
        (center[0], center[1]), np.rad2deg(angle), 1
    )

    # logger.info(f"{rotate_translate_mat = }")

    rotated = cv.warpAffine(frame, rotate_translate_mat, size)

    # logger.debug(f"{corners = }")

    rot_mat = rotate_translate_mat[:, :2]  # -- Remove extra opencv scale crap
    translation = rotate_translate_mat[:, 2]
    # # TODO :: Verify that changes made are why it "works" now
    # # TODO :: Translation matrix
    # logger.info(rot_mat)

    p0_rot = np.dot(rot_mat, corners[0])
    p2_rot = np.dot(rot_mat, corners[2])

    p0_rot = np.add(p0_rot, translation)
    p2_rot = np.add(p2_rot, translation)
    # # Crop to just slide
    # ## Rotate slide points
    # p0_rot = np.linalg.matmul(corners[0], rot_mat)[0:2]  # - top left
    # p1_rot = np.linalg.matmul(corners[1], rot_mat)[0:2]  # - top right
    # p2_rot = np.linalg.matmul(corners[2], rot_mat)[0:2]  # - bottom right
    # p3_rot = np.linalg.matmul(corners[3], rot_mat)[0:2]  # - bottom left

    if logger.level > 0:
        assert sketch is not None
        cv.rectangle(
            sketch,
            [int(x) for x in corners[0][0:2]],
            [int(x) for x in corners[2][0:2]],
            (0, 0, 255),
            1,
        )

        cv.rectangle(
            rotated,
            [int(x) for x in p0_rot[0:2]],
            [int(x) for x in p2_rot[0:2]],
            (0, 0, 255),
            1,
        )

    # logger.info(f"{p0_rot = } {p2_rot = }")

    cropped = rotated[
        int(p0_rot[1]) : int(p2_rot[1]), int(p0_rot[0]) : int(p2_rot[0])
    ]  # TODO :: I think this is right

    x, y, _colour = cropped.shape
    aspect_ratio = max(x / y, y / x)  # Get aspect ratio as (long slide / short side)

    NEW_HEIGHT = 1080
    new_width = int(NEW_HEIGHT * aspect_ratio)

    scaled = cv.resize(cropped, (new_width, NEW_HEIGHT))

    # logger.info(f"{scaled.shape = } {cropped.shape = }")

    if logger.level > 0:
        assert sketch is not None
        cv.imwrite(f"before_{slide.id}.png", sketch)
        cv.imwrite(f"rotated_{slide.id}.png", rotated)
        cv.imwrite(f"scaled_{slide.id}.png", scaled)
    # TODO :: REFACTOR AND UPSCALE
    slide.slide_content = scaled
    return slide


def get_slides(frame: MatLike) -> list[Slide] | bool:
    slides = detect_slide_tags(frame)
    if slides == False:
        return False

    assert type(slides) == list

    slides_and_content = [align_and_crop_slide(slide, frame) for slide in slides]
    return slides_and_content
