import logging
from ..spatial.api import corners, update_polygon, create_polygon


def run(spatial, canvas):
    polygon_ids = [create_polygon(canvas, c) for c in corners(spatial)]
    while True:
        for i, corner in enumerate(corners(spatial)):
            update_polygon(
                canvas,
                polygon_ids[i],
                corner,
            )
