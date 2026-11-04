from dataclasses import dataclass

from cv2.typing import MatLike

from puck.geometry import Point, Polygon


@dataclass
class Slide:
    """
    Will hold all information relating to slides, including but by no
        means limited to their position, linked multimedia, and anything
        else I feel like adding
    """

    # tags: Sequence[MatLike]  # Both tags that make up the slide corners
    # ids: MatLike  # Corrosponding ids
    id: int  # slide id is id of tag in top left corner

    area_camera: Polygon  # slide area in camera space
    slide_content: MatLike | None = None  # slide content image
