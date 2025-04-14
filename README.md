# Wordle solver

This program solves wordle puzzles. You can use the provided ipython notebook interactively to play a wordle puzzle. 

You make a move, then tell this program the move you made and the response you got. Then you ask it, "What's the best
move now?"

# How it works

Here's an explanation copy-pasted from a comment in the code:

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
