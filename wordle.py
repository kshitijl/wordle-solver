from collections import defaultdict
from enum import Enum
from dataclasses import dataclass
from functools import cache
from math import log


class Answer(str):
    pass


class Move(str):
    pass


class ResponseElem(Enum):
    NotPresent = 1
    WrongPlace = 2
    RightPlace = 3


class ResponseType(Enum):
    NotAWord = 1
    IsAWord = 2


@dataclass(eq=True, frozen=True)
class Response:
    type_: ResponseType
    elems: tuple[ResponseElem]


def create_response(s: str) -> Response:
    elems = []
    for c in s:
        assert c in "nwr"
        if c == "n":
            elems.append(ResponseElem.NotPresent)
        elif c == "w":
            elems.append(ResponseElem.WrongPlace)
        elif c == "r":
            elems.append(ResponseElem.RightPlace)
        else:
            assert False
    return Response(type_=ResponseType.IsAWord, elems=tuple(elems))


@cache
def predict(move: Move, answer: Answer) -> Response:
    assert len(move) == len(answer)
    response_elems = []
    for (move_elem, answer_elem) in zip(move, answer):
        if move_elem == answer_elem:
            response_elems.append(ResponseElem.RightPlace)
        elif move_elem in answer:
            response_elems.append(ResponseElem.WrongPlace)
        else:
            response_elems.append(ResponseElem.NotPresent)
    return Response(type_=ResponseType.IsAWord, elems=tuple(response_elems))


@cache
def is_consistent(answer: Answer, move: Move, response: Response) -> bool:
    return predict(move, answer) == response


def compute_entropy(dist: dict[Response, int]) -> float:
    total = float(sum(dist.values()))
    probabilities = [float(n) / total for n in dist.values()]
    return -sum([p * log(p) for p in probabilities])


class GameState(object):
    def __init__(self, dictionary: set[Answer]):
        self.possible_answers = dictionary
        self.dictionary = dictionary

    def _eliminate(self, move: Move, response: Response):
        self.possible_answers = {
            x for x in self.possible_answers if is_consistent(x, move, response)
        }

    def eliminate(self, move, response):
        print(f"Before: {len(self.possible_answers)} possibilities")
        self._eliminate(Move(move), create_response(response))
        print(f"After: {len(self.possible_answers)} possibilities")
        if len(self.possible_answers) < 20:
            print(self.possible_answers)

    def best_move(self) -> Move:
        response_distribution: dict[Move, dict[Response, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        for move in self.dictionary:
            move = Move(move)
            for answer in self.possible_answers:
                prediction = predict(move, answer)
                response_distribution[move][prediction] += 1

        entropy: dict[Move, float] = {}
        for move, distribution in response_distribution.items():
            entropy[move] = compute_entropy(distribution)

        entropy_list = list(reversed(sorted(entropy.items(), key=lambda y: y[1])))
        print(entropy_list[:10])
        return entropy_list[0][0]
