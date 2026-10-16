from dataclasses import dataclass
import logging

import cv2
import numpy as np
from numpy.typing import NDArray
from screeninfo import Monitor, get_monitors

from .chessboard import make_chessboard


logger = logging.getLogger(__name__)

WINDOW_PROJECTOR = "projector"
WINDOW_BACKGROUND = "background"
WINDOW_CHESSBOARD = "chessboard"
WINDOW_DIFFERENCE = "difference"
WINDOW_MASK = "mask"


@dataclass
class CalibrationInfo:
    camera_to_projector_homography: np.ndarray
    background_frame: NDArray[np.uint8]


def get_projector() -> Monitor:
    monitors = get_monitors()

    for i, monitor in enumerate(monitors):
        logger.info(
            "Monitor %d: %dx%d at (%d, %d)",
            i,
            monitor.width,
            monitor.height,
            monitor.x,
            monitor.y,
        )

    if len(monitors) < 2:
        raise RuntimeError("No second monitor/projector found")

    return monitors[0]


def make_fullscreen_projector_window(
    window_name: str,
    projector: Monitor,
) -> None:
    """Create and show a fullscreen black window on the projector display."""
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # OpenCV does not reliably create the native window until the first image
    # is shown and GUI events have been processed. Materialise it before asking
    # the window manager to make it fullscreen.
    black_screen = np.zeros(
        (projector.height, projector.width, 3),
        dtype=np.uint8,
    )
    cv2.moveWindow(window_name, projector.x, projector.y)
    cv2.resizeWindow(window_name, projector.width, projector.height)
    cv2.imshow(window_name, black_screen)
    cv2.waitKey(100)

    # Move once more now that the native window exists, then fullscreen it on
    # the display containing that position.
    cv2.moveWindow(window_name, projector.x, projector.y)
    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )
    # Refresh after changing the native window style. Some backends otherwise
    # defer fullscreen until the next frame is rendered.
    cv2.imshow(window_name, black_screen)
    cv2.waitKey(250)


def discard_frames(
    camera: cv2.VideoCapture,
    count: int = 5,
) -> None:
    """
    Discard several frames from the camera.

    This is useful after changing the projector image because
    camera/video pipelines may contain buffered frames.
    """
    for _ in range(count):
        ok, _ = camera.read()

        if not ok:
            raise RuntimeError("Could not read camera frame")


def capture_average(
    camera: cv2.VideoCapture,
    count: int = 10,
) -> np.ndarray:
    """
    capture several frames and return their pixel-wise average.
    averaging suppresses random sensor noise.
    """
    frames = []

    for _ in range(count):
        ok, frame = camera.read()

        if not ok:
            raise RuntimeError("Could not read camera frame")

        frames.append(frame.astype(np.float32))

    average = np.mean(
        frames,
        axis=0,
    )

    return average.astype(np.uint8)


def show_image_on_projector_and_wait(
    window_name: str,
    image: np.ndarray,
    settle_ms: int = 200,
) -> None:
    """
    display an image on the projector and allow some time for the display to settle.
    """
    cv2.imshow(window_name, image)
    cv2.waitKey(settle_ms)


def calibrate(debug: bool = True) -> CalibrationInfo:
    """Calibrate the camera-to-projector mapping.

    Args:
        debug: Show intermediate capture and detection windows when true.
            The projector window is always shown because it is required for
            calibration.
    """
    # TODO: refactor to handle exceptions and return types more gracefully. Possibility of returning None?
    
    logger.info("Running calibration tool")

    homography = None

    # Projector
    logger.info("Setting up projector")
    projector = get_projector()

    make_fullscreen_projector_window(
        WINDOW_PROJECTOR,
        projector,
    )

    # The fullscreen helper initially displays black.
    black_screen = np.zeros(
        (projector.height, projector.width, 3), dtype=np.uint8
    )

    # camera
    logger.info("Setting up camera")
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open camera")

    # make the chessboard
    logger.info("Generating calibration board")
    board_size = (projector.width, projector.height)
    board_shape = (8, 6)
    board, board_points = make_chessboard(board_size, board_shape)
    logger.info(
        "Generated calibration chessboard with %d calibration points",
        len(board_points),
    )

    # Note: the section from here to the end of the function might throw an
    # exception of some kind, which should be caught higher up the call chain.
    logger.info("Capturing background")
    show_image_on_projector_and_wait(WINDOW_PROJECTOR, black_screen)
    discard_frames(camera, count=5)
    background_frame = capture_average(camera, count=10)

    logger.info("Capturing projected chessboard")
    show_image_on_projector_and_wait(WINDOW_PROJECTOR, board)
    discard_frames(camera, count=5)
    chessboard_frame = capture_average(camera, count=10)

    # compute the grayscale difference image
    gray_background = cv2.cvtColor(background_frame, cv2.COLOR_BGR2GRAY)
    gray_chessboard = cv2.cvtColor(chessboard_frame, cv2.COLOR_BGR2GRAY)
    difference = cv2.subtract(gray_chessboard, gray_background)

    # Stretch the projected-light signal to the full display range. Camera
    # exposure, projector brightness, and surface reflectance make a fixed
    # threshold unreliable across different hardware.
    difference_normalized = np.zeros(shape=difference.shape, dtype=difference.dtype)
    difference_normalized = cv2.normalize(
        difference,
        difference_normalized,
        alpha=0,
        beta=255,
        norm_type=cv2.NORM_MINMAX,
    )
    threshold_value, mask = cv2.threshold(
        difference_normalized,
        0,
        255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )

    if debug:
        cv2.imshow(WINDOW_BACKGROUND, gray_background)
        cv2.imshow(WINDOW_DIFFERENCE, difference_normalized)
        cv2.imshow(WINDOW_MASK, mask)

    corner_shape = (
        board_shape[0] - 1,
        board_shape[1] - 1,
    )
    print(difference_normalized)
    detector_flags = (
        cv2.CALIB_CB_NORMALIZE_IMAGE
        | cv2.CALIB_CB_EXHAUSTIVE
        | cv2.CALIB_CB_ACCURACY
    )
    # The raw camera frame contains the surface and ambient scene. Detect on
    # the background-subtracted projection first, with the raw frame only as
    # a fallback.
    ret = False
    corners = None
    detection_source = "none"
    detector_inputs = (
        ("background difference", difference_normalized),
        ("Otsu mask", mask),
        ("raw camera frame", gray_chessboard),
    )
    for source_name, detector_image in detector_inputs:
        found, candidate_corners = cv2.findChessboardCornersSB(
            detector_image,
            corner_shape,
            flags=detector_flags,
        )
        logger.info("Detection using %s: %s", source_name, found)
        if found:
            ret = True
            corners = candidate_corners
            detection_source = source_name
            break

    percentiles = np.percentile(difference, (50, 90, 99))
    logger.info(
        "Projection difference percentiles (50/90/99%%): %s",
        ", ".join(f"{value:.1f}" for value in percentiles),
    )
    logger.info("Automatic threshold: %.1f", threshold_value)
    logger.info("Detect captured board: %s", ret)
    logger.info("Detection source: %s", detection_source)
    logger.info("Expected corners: %s", corner_shape)
    logger.info("Detected: %d", 0 if corners is None else len(corners))
    logger.info("Board points: %d", len(board_points))

    # if found, add object points, image points (after refining them)
    if ret and corners is not None:
        # findChessboardCornersSB already returns sub-pixel corner positions.
        if debug:
            display_image = gray_chessboard.copy()
            cv2.drawChessboardCorners(
                display_image, corner_shape, corners, ret
            )
            cv2.imshow(WINDOW_CHESSBOARD, display_image)

        # Compute a mapping from camera-image coordinates to projector
        # coocalrdinates. Both arrays must use the same row-major corner order.
        camera_points = corners.reshape(-1, 2).astype(np.float32)
        if camera_points[0].sum() > camera_points[-1].sum():
            # A chessboard is symmetric under a 180-degree rotation, so the
            # detector may return its corners in reverse order. Anchor the
            # first corner at the camera image's top-left to match the
            # projector point ordering produced by make_chessboard().
            camera_points = camera_points[::-1].copy()
        projector_points = np.asarray(board_points, dtype=np.float32)
        homography, inlier_mask = cv2.findHomography(
            camera_points,
            projector_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=5.0,
        )

        if homography is None or inlier_mask is None:
            raise RuntimeError(
                "Could not compute the camera-to-projector homography"
            )

        mapped_points = cv2.perspectiveTransform(
            camera_points.reshape(-1, 1, 2),
            homography,
        ).reshape(-1, 2)
        reprojection_errors = np.linalg.norm(
            mapped_points - projector_points,
            axis=1,
        )
        inliers = inlier_mask.ravel().astype(bool)

        logger.info(
            "Camera-to-projector homography:\n%s",
            np.array2string(homography, precision=8, suppress_small=True),
        )
        logger.info(
            "Homography inliers: %d/%d", inliers.sum(), len(inliers)
        )
        logger.info(
            "Mean inlier reprojection error: %.3f projector pixels",
            reprojection_errors[inliers].mean(),
        )
        logger.info(
            "Maximum inlier reprojection error: %.3f projector pixels",
            reprojection_errors[inliers].max(),
        )
    else:
        raise RuntimeError("Could not detect chessboard points")

    if debug:
        logger.info("Press q to close the debug windows")
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    return CalibrationInfo(
        camera_to_projector_homography=homography, 
        background_frame=background_frame,
    )
