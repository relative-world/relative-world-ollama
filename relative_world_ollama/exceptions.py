class UnparsableResponseError(Exception):
    """
    Raised when the agent's response could not be parsed.

    Probably because of really wrong json, but who knows right?
    """


    def __init__(self, response):
        self.response = response
        super().__init__(f"Could not parse response: {response}")
