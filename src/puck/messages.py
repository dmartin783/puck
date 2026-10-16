class Message():
    def __init__(self, contents):
        self.contents = contents

class Monologue(Message):
    def __init__(self, contents):
        self.contents= contents

class Dialogue(Message):
    __match_args__ = ("contents", "sender")
    def __init__(self, contents, sender):
        self.contents = contents
        self.sender = sender



