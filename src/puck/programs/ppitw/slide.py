from dataclasses import dataclass

import cv2
from cv2.typing import MatLike
from PIL import Image, ImageTk

from puck.geometry import Point, Polygon


class Slide:
    """
    Will hold all information relating to slides, including but by no
        means limited to their position, linked multimedia, and anything
        else I feel like adding
    """

    def __init__(
        self, id: int, area_camera: Polygon, slide_content: MatLike | None = None
    ):
        # tags: Sequence[MatLike]  # Both tags that make up the slide corners
        # ids: MatLike  # Corrosponding ids
        self.id: int = id  # slide id is id of tag in top left corner

        self.area_camera: Polygon = area_camera  # slide area in camera space
        self._slide_content: MatLike | None = slide_content  # slide content image
        self.slide_content_tk: ImageTk | None = None

    @property
    def slide_content(self) -> MatLike | None:
        return self._slide_content

    @slide_content.setter
    def slide_content(self, img: MatLike):
        self._slide_content = img
        self.slide_content_tk = ImageTk.PhotoImage(
            Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        )
