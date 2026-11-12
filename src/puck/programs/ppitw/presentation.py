import logging
import tkinter as tk
from dataclasses import dataclass
from tkinter import Canvas

import cv2 as cv
from cv2.typing import MatLike

from puck.programs.ppitw.slide import Slide

logger = logging.getLogger(__name__)


@dataclass
class PresView:
    canvas: Canvas

    def set_slide(self, slide: Slide) -> int:
        logger.info("called set_slide")
        assert slide.slide_content is not None
        logger.info(f"{slide.slide_content_tk = }")
        img_id = self.canvas.create_image(
            0,
            0,
            anchor=tk.NW,
            image=slide.slide_content_tk,
        )

        logger.info(f"{img_id = }")
        return img_id

    def update_slide(self, slide: Slide, id: int):
        self.canvas.itemconfig(id, image=slide.slide_content_tk)
