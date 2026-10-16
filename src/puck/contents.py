class Contents():
    def __init__(self, content):
        self.content = content

class FunctionContents(Contents):
    __match_args__ = ("function_call", "args", "kwargs")
    def __init__(self, function_call, args, kwargs):
        self.function_call = function_call
        self.args = args
        self.kwargs = kwargs
