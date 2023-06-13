from typing import Any


class VerifEval:
    def __init__(self, distance_function) -> None:
        self.distance_function = distance_function

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        pass
