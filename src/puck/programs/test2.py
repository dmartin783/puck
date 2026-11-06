import logging

import numpy as np

# Relative import of the Actor class.
from ..actor import Actor
from ..geometry import Polygon

# Every file has a logger.
logger = logging.getLogger(__name__)


def run(self: Actor) -> None:
    """Target function for an actor that just tracks a paper on the webcam.

    Args:
        self (Actor): the Actor whose target function is 'run'
    """
    logger.info("test2_run has been called sucessfully")
    # The first message establishes the drawing queue as a variable that this function can access.
    # Thus when you initialise the actor you should always (current implementation, this might change) send a message with the drawing_queue reference
    # Any messages that follow this will be processed by the while loop and will likely use the drawing_queue to pass graphical commands to the main thread.
    first_message = self.recieve()
    assert first_message[0] == "drawing_queue"
    drawing_queue = first_message[1]

    # These are lists that are for this actor to hold onto whatever graphical objects and actors it spawns off.
    associated_canvas_ids: list[int] = []
    spawned_actors: list[Actor] = []

    # This is an infinite loop to constantly be reading the mailbox in its thread.
    # It uses a match-case structure to figure out what each message is telling us.
    # This likely/may change in future implementations and you can change this depending on what messages you wish to send to your programs.abs
    while True:
        message = self.recieve()
        logger.debug(f"run has been gotten {message}")
        match message:
            case "kill":
                # self.end() # This command was only really useful if .end() has a .join() in its implementation
                # The current implementation lacks .join() so this is really just used to exit the infinite loop.
                break
            case ("new_shape", ("type", type), ("coordinates", coordinates)):
                # If you have a new shape, it will need to be drawn by the canvas.
                # To draw anything on the canvas, one has to add to the drawing_queue to hand that off to the main thread.
                drawing_queue.put(
                    (
                        "action",
                        (
                            "new",
                            ("type", "rectangle"),
                            ("sender", self),
                            (
                                "coordinates",
                                Polygon.from_array(
                                    np.array([[0, 0], [0, 100], [100, 100], [100, 0]])
                                ),
                            ),
                        ),
                    )
                )
            case ("update_shape", ("id", id), ("coordinates", coordinates)):
                # Currently we are grabbing the associated_canvas_id list arbitrarily and if there's something within the list we will update that object.
                if associated_canvas_ids:
                    # As before, anything to do with graphics you must pass back a message to the drawing queue to do any update to the graphics system.
                    ...
                else:
                    # If there is nothing in the associated_canvas_ids list we must assume there are no graphical objects to update.
                    logger.error(
                        "We have no associated_canvas_ids yet so we cannot update."
                    )
            case ("information", *info):
                # Information currently is just a returned message from the main thread providing you the canvas_ids that it has created from your previous requests.
                # As such it is just added to the list of 'associated_canvas_ids' for the actor.
                match info[0]:
                    case ("add_ids", id_list):
                        associated_canvas_ids.append(id_list)
                    case _ as info:
                        # In the case that it is not the tuple ('add_ids', id_list), simply print the information so you know what it is.
                        print(info)
            case _ as undefined:
                # If it is something we haven't accounted for print that back out to the user to tell them that you don't know what to do with this message.
                print(
                    f"Youv'e given me this message: {undefined} \n"
                    "I do not know what to do with it as I do not have a match case...."
                )
    logger.info("run finished")
