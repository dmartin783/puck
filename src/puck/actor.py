"""
This module provides the Actor class for running Puck programs

For every program in Puck there is an associated actor which runs its program in its own thread.
This allows programs to be encapsulated and run concurrently alongside each other.

In other actor systems, the program that runs might be referred to as a 'process'.
In our system, we call it the target function. This is because we are inheriting from the Python thread class, which requires a target function that it will run upon starting the thread.

Each actor has a 'mailbox' which it uses to receive communication from other actors and the main thread in a controlled manner.
Our actors' mailboxes are Python Queues and therefore adheres to a FIFO reading of messages. This assumption diverges from the abstract Actor model theory, which requires an arbiter but not necessarily a FIFO ordering to decide which message to read and is noted here for this reason.

By using 'receive' it can read messages from the mailbox (blocking) and using 'send' can address messages to other actors' mailboxes (non blocking).
It can use 'end' to trigger joining the spawned thread to the main thread.


Example 1:
    from actor import Actor
    import importlib

    ...
    module_name = program_lookup[program_encoding]
    module = importlib.import_module(module_name)

    actor = Actor(target=module.run)
    actor.start()

In Example 1, by using importlib we can dynamically import in the `module` which the `run` function originates from from a given `module_name`.
This allows us to create any number of programs with a `run` function and interchange them based on their apriltag encoding using the program_lookup list.

Example 2:
TODO: Make more examples
"""

from collections.abc import Callable
from queue import Queue
from threading import Thread
from typing import Generic, Self, TypeVar

# Type of object sent and received by actors via mailboxes.
# In Puck, this will generally be the Message class.
M = TypeVar("M")


class Actor(Generic[M], Thread):
    def __init__(self, target: Callable[[Self], None]):
        """Initialises an actor.

        Args:
            target (a function): The function that will be triggered when you start the actor.

        Attributes:
            mailbox: a queue which other actors and the main thread can send messages to.
        """
        super().__init__(target=target, args=(self,))
        self.mailbox: Queue[M] = Queue()

    def send(self, message: M) -> None:
        self.mailbox.put(message)

    def recieve(self) -> M:
        """Get the first message from the mailbox.

        If the mailbox is empty, this hangs until a message exists.
        """
        return self.mailbox.get()
