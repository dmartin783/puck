import logging

# # Relative import of the Actor class.
# from ..actor import Actor
# from ..messages import (
#     Message,
#     AddPolygonMessage,
#     DrawingQueueReferenceMessage,
#     PaperLocationMessage,
#     KillMessage,
#     CanvasIdMessage,
#     UpdatePolygonMessage,
# )

# # Every file has a logger.
# logger = logging.getLogger(__name__)

def run():
    while True:
        print("blarp")
# def run(self: Actor[Message]) -> None:
#     """Target function for an actor that just tracks a paper on the webcam.

#     Args:
#         self: the Actor whose target function is 'run'
#     """
#     logger.info("test_run has been called sucessfully")

#     # First message should establish a reference to the drawing queue.
#     first_message = self.recieve()
#     assert type(first_message) is DrawingQueueReferenceMessage
#     drawing_queue = first_message.drawing_queue

#     # These are lists that are for this actor to hold onto whatever graphical objects and actors it spawns off.
#     # TODO: use sentinels in future Python?
#     paper_highlight_shape_id_list: list[int] = []
#     spawned_actors: list[Actor[Message]] = []

#     # Handle incoming messages
#     while True:
#         message = self.recieve()
#         logger.debug(f"run has been gotten {message}")
#         match message:
#             case KillMessage():
#                 break



            
#             case PaperLocationMessage(paper_location):
#                 # Highlight the paper in white
#                 if len(paper_highlight_shape_id_list) < 4:
#                     drawing_queue.put(AddPolygonMessage(self, paper_location))
#                 elif len(paper_highlight_shape_id_list) == 4:
#                     for i in paper_highlight_shape_id_list:
#                         drawing_queue.put(
#                             UpdatePolygonMessage(self, i, paper_location)
#                         )
#                 elif len(paper_highlight_shape_id_list) > 4:
#                     print(paper_highlight_shape_id_list)
#                     print("BAD BAD BAD BAD BAD")
#             case CanvasIdMessage(canvas_id):
#                 # This is the id Tkinter has assigned to our paper shape
#                 paper_highlight_shape_id_list.append(canvas_id)
#             case _:
#                 logger.error(f"Received unexpected message {message}")

#     logger.info("run finished")
