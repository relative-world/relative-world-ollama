class TooledLocation(Location, ):
    def __init__(self, location: str):
        self.location = location

    def __str__(self):
        return self.location

    def __repr__(self):
        return f"TooledLocation({self.location})"

    def __eq__(self, other):
        if isinstance(other, TooledLocation):
            return self.location == other.location
        return False

    def __hash__(self):
        return hash(self.location)
