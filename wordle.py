from collections import defaultdict
from enum import Enum
from dataclasses import dataclass
from functools import cache
from math import log


class Answer(int):
    pass


class Move(int):
    pass


class ResponseElem(Enum):
    NotPresent = 1
    WrongPlace = 2
    RightPlace = 3


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
        response_packed = response_packed * 10 + elem.value

    return Response(response_packed)


@cache
def pack_word(word: str) -> int:
    assert len(word) == 5
    packed: int = 0
    ord_a = ord("a")
    for c in word:
        packed = packed * 26 + ord(c.lower()) - ord_a
    return packed


@cache
def unpack_word(word: int) -> str:
    unpacked = []
    ord_a = ord("a")
    for i in range(5):
        c = word % 26
        word = word // 26
        unpacked.append(chr(c + ord_a))
    return "".join(reversed(unpacked))


@cache
def predict(move: Move, answer: Answer) -> Response:
    """If the true answer to the puzzle is the word [answer], and we play the
    word [move], then what would be the game's [Response], i.e., the clues that
    it would give back?

    """

    response_packed = 0
    unpacked_move, unpacked_answer = unpack_word(move), unpack_word(answer)
    for move_elem, answer_elem in zip(unpacked_move, unpacked_answer):
        if move_elem == answer_elem:
            response_item = ResponseElem.RightPlace
        elif move_elem in unpacked_answer:
            response_item = ResponseElem.WrongPlace
        else:
            response_item = ResponseElem.NotPresent

        response_packed = response_packed * 10 + response_item.value

    return Response(response_packed)


@cache
def is_consistent(answer: Answer, move: Move, response: Response) -> bool:
    """If the true answer to the puzzle is the word [answer], and we play the
    word [move], then would the game respond with [response]?

    """
    return predict(move, answer) == response


def compute_entropy(dist: dict[Response, int]) -> float:
    """Given a dictionary of Response -> how often that response arises,
    compute the entropy of the probability distribution.

    """
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
        """We have played the word [move] and the game responded with
        [response]. Eliminate all possibilities that are inconsistent with that
        response.

        """
        print(f"Before: {len(self.possible_answers)} possibilities")
        self._eliminate(Move(move), create_response(response))
        print(f"After: {len(self.possible_answers)} possibilities")
        if len(self.possible_answers) < 20:
            print(self.possible_answers)

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

        response_distribution: dict[Move, dict[Response, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        total_moves = len(self.dictionary)
        print_every = int(total_moves / 10)
        print(
            f"There are {total_moves} moves and {len(self.possible_answers)} answers to evaluate. Will print progress every {print_every} moves"
        )

        for i, possible_answer in enumerate(self.dictionary):
            i += 1
            if i % print_every == 0:
                print(f"Evaluated {i} out of {total_moves} moves")

            move = Move(possible_answer)
            for answer in self.possible_answers:
                prediction: Response = predict(move, answer)
                response_distribution[move][prediction] += 1

        entropy: dict[Move, float] = {}
        for move, distribution in response_distribution.items():
            entropy[move] = compute_entropy(distribution)

        entropy_list = list(reversed(sorted(entropy.items(), key=lambda y: y[1])))
        for top10_item in entropy_list[:10]:
            print((unpack_word(top10_item[0]), top10_item[1]))
        return entropy_list[0][0]


if __name__ == "__main__":
    for test_word in ["tales", "stare", "zzzzz", "aaaaa"]:
        assert unpack_word(pack_word(test_word)) == test_word
    dictionary = [x.strip() for x in open("common.txt").readlines()]
    dictionary_ = [Answer(pack_word(x.lower())) for x in dictionary if len(x) == 5]
    game = GameState(set(dictionary_))
    game.best_move()
