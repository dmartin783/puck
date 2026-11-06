from chorus import Proxy, me

from ..geometry import Polygon


def corners(spatial: Proxy):
    return spatial.ask.corners(me())


def create_polygon(canvas: Proxy, polygon: Polygon) -> int:
    points = polygon.flat_unwrap()
    return canvas.ask.create_polygon(points, fill="white")


def update_polygon(canvas: Proxy, graphical_id: int, polygon: Polygon):
    points = polygon.flat_unwrap()
    canvas.tell.coords(graphical_id, points)
