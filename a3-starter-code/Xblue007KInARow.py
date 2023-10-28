"""
Xblue007KInARow.py
Author: Xander Bishop
An agent for playing "K-in-a-Row with Forbidden Squares"
CSE 415, University of Washington

THIS IS A TEMPLATE WITH STUBS FOR THE REQUIRED FUNCTIONS.
YOU CAN ADD WHATEVER ADDITIONAL FUNCTIONS YOU NEED IN ORDER
TO PROVIDE A GOOD STRUCTURE FOR YOUR IMPLEMENTATION.

"""
import copy
import time

# Global variables to hold information about the opponent and game version:
INITIAL_STATE = None
OPPONENT_NICKNAME = 'Not yet known'
OPPONENT_PLAYS = 'X'  # Update this after the call to prepare.

# Information about this agent:
MY_LONG_NAME = 'Doyle Abernathy'
MY_NICKNAME = 'D-Meister'
I_PLAY = 'O'  # Gets updated by call to prepare.

# GAME VERSION INFO
M = 0
N = 0
K = 0
TIME_LIMIT = 0
plays = {}


############################################################
# INTRODUCTION
def introduce():
    intro = '\nHello, my name is Doyle Abernathy.\n' + \
            'Xander Bishop (2042499) made me.\n' + \
            'I\'m a devious agent who will do\n' + \
            'anything to take the win.'
    return intro


def nickname():
    return MY_NICKNAME


############################################################

# Receive and acknowledge information about the game from
# the game master:
def prepare(initial_state, k, what_side_I_play, opponent_nickname):
    # Write code to save the relevant information in either
    # global variables.
    global INITIAL_STATE, M, N, K, OPPONENT_PLAYS, OPPONENT_NICKNAME
    INITIAL_STATE = initial_state
    M = len(INITIAL_STATE[0])
    N = len(INITIAL_STATE[0][0])
    K = k
    I_PLAY = what_side_I_play
    if I_PLAY == 'X':
        OPPONENT_PLAYS = 'O'
    else:
        OPPONENT_PLAYS = 'X'
    OPPONENT_NICKNAME = opponent_nickname
    return "OK"


############################################################

def makeMove(currentState, currentRemark, timeLimit=10000):
    print("makeMove has been called")

    print("code to compute a good move should go here.")
    # Here's a placeholder:
    a_default_move = [0, 0]  # This might be legal ONCE in a game,
    # if the square is not forbidden or already occupied.
    global TIME_LIMIT
    start = time.time()
    TIME_LIMIT = timeLimit
    bMove = float("-inf")
    # looks for a valid move which could be made
    newState = currentState.copy()  # This is not allowed, and even if
    # it were allowed, the newState should be a deep COPY of the old.
    newRemark = "Let's Do this.\n" + \
                "Watch out for my secret move."

    for s in range(M):
        for t in range(N):
            if currentState[0][s][t] == ' ':
                a_default_move = [s, t]

    for s in range(M):
        for t in range(N):
            if currentState[0][s][t] == ' ':
                tempState = copy.deepcopy(currentState)
                tempState[0][s][t] = I_PLAY
                better = minimax(tempState, 2, startTime=start)
                if better[0] > bMove:
                    a_default_move = [s, t]
                    bMove = better[0]
                    newState = tempState
    if newState[1] == 'X':
        newState[1] = 'O'
    else:
        newState[1] = 'X'
    print("Returning from makeMove")
    return [[a_default_move, newState], newRemark]


##########################################################################

# The main adversarial search function:
def minimax(state, depthRemaining, startTime, pruning=False, alpha=None, beta=None, zHashing=None):
    print("running minimax")
    if depthRemaining == 0:
        return staticEval(state[0])

    if state[1] == I_PLAY:
        default_score = float("-inf")
    else:
        default_score = float("inf")

    if TIME_LIMIT - (time.time() - startTime) < 0.5:
        return [default_score, "my own optional stuff", "more of my stuff"]

    for s in range(M):
        for t in range(N):
            if state[0][s][t] == ' ':
                altState = copy.deepcopy(state)
                altState[0][s][t] = altState[1]
                if altState[1] == 'X':
                    altState[1] = 'O'
                else:
                    altState[1] = 'X'
                newVal = minimax(altState, depthRemaining - 1, startTime)
                if isinstance(newVal, list):
                    newVal = newVal[0]
                if (state[1] == I_PLAY and newVal > default_score) \
                        or (state[1] != I_PLAY and newVal < default_score):
                    default_score = newVal

    return [default_score, "my own optional stuff", "more of my stuff"]
    # Only the score is required here but other stuff can be returned
    # in the list, after the score.


##########################################################################

def staticEval(state):
    print('calling staticEval. Its value needs to be computed!')
    # Values should be higher when the states are better for X,
    # lower when better for O.
    total = 0
    for s in range(M):
        for t in range(N):
            if state[s][t] == 'X' or state[s][t] == 'O':
                board = state[s][t]
                # horizontal
                if t + K <= N:
                    total = total + check(state, s, t, board, 0, 1)
                # down vertical
                if s + K <= M:
                    total = total + check(state, s, t, board, 1, 0)
                # diagonal
                if t + K <= N and s + K <= M:
                    total = total + check(state, s, t, board, 1, 1)
                # anti-diagonal
                if t + 1 >= K and s + K <= M:
                    total = total + check(state, s, t, board, 1, -1)

    return total


def check(state, s, t, board, x, y):
    p = -1
    for i in range(K):
        if state[s + x * i][t + y * i] == board:
            p = p + 1
        elif state[s + x * i][t + y * i] != ' ':
            p = -1
            break
    if p > -1:
        if board == I_PLAY:
            return 10 ** p
        else:
            return 0 - 10 ** p
    return 0
##########################################################################
