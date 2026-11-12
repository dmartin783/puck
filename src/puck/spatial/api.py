import tkinter as tk
from tkinter import Label, PhotoImage

import cv2
from chorus import Proxy, me
from PIL import Image, ImageTk

from puck.programs.ppitw.slide import Slide

from ..geometry import Polygon


def corners(spatial: Proxy):
    return spatial.ask.corners(me())


def create_polygon(canvas: Proxy, polygon: Polygon) -> int:
    points = polygon.flat_unwrap()
    return canvas.ask.create_polygon(points, fill="white")


def update_polygon(canvas: Proxy, graphical_id: int, polygon: Polygon):
    points = polygon.flat_unwrap()
    canvas.tell.coords(graphical_id, points)


def create_image(canvas: Proxy, slide: Slide) -> int:
    # image_id: int = canvas.ask.create_image(
    #     0, 0, image=slide.slide_content_tk, anchor=tk.NW
    # )
    image_id = canvas.ask.set_slide(slide)
    return image_id


def update_image(main_thread: Proxy, slide: Slide, id: int):
    main_thread.tell.update_slide(slide, id)
