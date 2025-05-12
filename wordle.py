from collections import defaultdict
from enum import Enum
from math import log
import time
from contextlib import contextmanager
from typing import List, Any
import numpy as np


@contextmanager
def timing_context(description="Execution"):
    start_time = time.perf_counter()
    try:
        yield
    finally:
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        print(f"{description} took: {execution_time:.4f} seconds")


class Answer(int):
    pass


class Move(int):
    pass


class ResponseElem(Enum):
    NotPresent = 0
    WrongPlace = 1
    RightPlace = 2


class Response(int):
    pass


def create_response(s: str) -> Response:
    """After a move is played, we get a response from the game. It's easier to
    type those responses as short strings like "nwr" to mean that one letter
    was not present, the next was present but in the wrong place, and the next
    was in the right place. This function turns these strings into a typed
    [Response].

    """
    # elems: List[ResponseElem] = []
    response_packed = 0
    for c in s:
        assert c in "nwr"
        if c == "n":
            elem = ResponseElem.NotPresent
        elif c == "w":
            elem = ResponseElem.WrongPlace
        elif c == "r":
            elem = ResponseElem.RightPlace
        else:
            assert False
        response_packed = response_packed * 3 + elem.value

    assert response_packed < 256
    return Response(response_packed)


def predict_(move: str, answer: str) -> Response:
    """If the true answer to the puzzle is the word [answer], and we play the
    word [move], then what would be the game's [Response], i.e., the clues that
    it would give back?

    """

    response_packed = 0

    for move_elem, answer_elem in zip(move, answer):
        if move_elem == answer_elem:
            response_item = ResponseElem.RightPlace
        elif move_elem in answer:
            response_item = ResponseElem.WrongPlace
        else:
            response_item = ResponseElem.NotPresent

        response_packed = response_packed * 3 + response_item.value

    assert response_packed < 256
    return Response(response_packed)


def predict(dictionary: List[str], move: Move, answer: Answer) -> Response:
    return predict_(dictionary[move], dictionary[answer])


def is_consistent(answer: Answer, move: Move, response: Response) -> bool:
    """If the true answer to the puzzle is the word [answer], and we play the
    word [move], then would the game respond with [response]?

    """
    # return predict(move, answer) == response
    return True


def compute_entropy(dist: np.ndarray) -> float:
    """Given a dictionary of Response -> how often that response arises,
    compute the entropy of the probability distribution.

    """
    total = float(np.sum(dist))
    probabilities = [float(n) / total for n in dist if n > 0]
    return -sum([p * log(p) for p in probabilities])


class GameState(object):
    def __init__(self, dictionary: List[str]):
        self.dictionary: List[str] = dictionary
        self.possible_answers: List[Answer] = [
            Answer(x) for x in range(len(self.dictionary))
        ]

    # def _eliminate(self, move: Move, response: Response):
    #     self.possible_answers = [
    #         x for x in self.possible_answers if is_consistent(x, move, response)
    #     ]

    # def eliminate(self, move, response):
    #     """We have played the word [move] and the game responded with
    #     [response]. Eliminate all possibilities that are inconsistent with that
    #     response.

    #     """
    #     print(f"Before: {len(self.possible_answers)} possibilities")
    #     self._eliminate(Move(move), create_response(response))
    #     print(f"After: {len(self.possible_answers)} possibilities")
    #     if len(self.possible_answers) < 20:
    #         print(self.possible_answers)

    def compute_all_predictions(self):
        num_words = len(self.dictionary)
        self.predictions = np.zeros(num_words * num_words, dtype=np.uint8)

        total_moves = len(self.dictionary)
        print_every = int(total_moves / 10)
        print(
            f"There are {total_moves} moves and {len(self.possible_answers)} answers to evaluate. Will print progress every {print_every} moves"
        )

        for possible_move_idx in range(len(self.dictionary)):
            if possible_move_idx % print_every == 0:
                print(f"Evaluated {possible_move_idx} out of {total_moves} moves")

            move = Move(possible_move_idx)
            for possible_answer_idx in range(len(self.dictionary)):
                prediction: Response = predict(
                    self.dictionary, move, Answer(possible_answer_idx)
                )
                self.predictions[
                    possible_move_idx * num_words + possible_answer_idx
                ] = prediction

    def best_move(self) -> Move:
        """Given the current state of the game, what's the best move?

        The best move is the one that gives us "most information".

        To develop some intuition, imagine playing a move such that, for all
        possible true answers, the game would produce the same response. It is
        useless to play this move. Playing it gives us no information. The
        distribution of responses over answers looks like a big spike. The
        precise way of saying that is that this distribution has low entropy.

        Imagine playing a move that elicits a different response from the game
        for each different possible answer. That would be amazing! We would
        play this move. The game would give us its response. We would eliminate
        all answers that are inconsistent with that response, leaving the one
        and only true answer. We win the game next move. The precise way of
        saying that is that this distribution has high entropy.

        This function computes the entropy of the response distribution for
        each possible move, by iterating over each possible answer for each
        move. Then, it returns the move with highest response entropy.

        A key insight is that the history of past moves doesn't matter except
        in that it tells us which words are still possible contenders for being
        the answer.

        """
        if len(self.possible_answers) == 1:
            return Move(list(self.possible_answers)[0])

        # TODO: either allcate this in one big fell swoop or go back to defaultdict of defaultdict
        response_distribution: dict[Move, np.ndarray[Any, Any]] = defaultdict(
            lambda: np.zeros(256, np.uint16)
        )

        total_moves = len(self.dictionary)
        print_every = int(total_moves / 10)
        print(
            f"There are {total_moves} moves and {len(self.possible_answers)} answers to evaluate. Will print progress every {print_every} moves"
        )

        for possible_move_idx in range(len(self.dictionary)):
            if possible_move_idx % print_every == 0:
                print(f"Evaluated {possible_move_idx} out of {total_moves} moves")

            move = Move(possible_move_idx)
            for answer in self.possible_answers:
                prediction: Response = self.predictions[
                    possible_move_idx * total_moves + answer
                ]
                response_distribution[move][prediction] += 1

        entropy: dict[Move, float] = {}
        for move, distribution in response_distribution.items():
            entropy[move] = compute_entropy(distribution)

        entropy_list = list(reversed(sorted(entropy.items(), key=lambda y: y[1])))
        for top10_item in entropy_list[:10]:
            print((self.dictionary[top10_item[0]], top10_item[1]))
        return entropy_list[0][0]


if __name__ == "__main__":
    dictionary = [x.strip() for x in open("dictionary.txt").readlines()]
    dictionary_ = list(sorted([x.lower() for x in dictionary if len(x) == 5]))
    game = GameState(dictionary_)
    with timing_context("precompute"):
        game.compute_all_predictions()
    with timing_context("first time"):
        game.best_move()
    with timing_context("second time"):
        game.best_move()
