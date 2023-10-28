'''
yc324KInARow.py
Author: Emma Cao
An agent for playing "K-in-a-Row with Forbidden Squares"
CSE 415, University of Washington

'''

import time
import copy

# Global variables to hold information about the opponent and game version:
INITIAL_STATE = None
OPPONENT_NICKNAME = 'Not yet known'
OPPONENT_PLAYS = 'O' # Update this after the call to prepare.

# Information about this agent:
MY_LONG_NAME = 'Foggy'
MY_NICKNAME = 'ycemma'
I_PLAY = None # Gets updated by call to prepare.
 
# GAME VERSION INFO
M = 0
N = 0
K = 0
TIME_LIMIT = 0

############################################################
# INTRODUCTION
def introduce():
    intro = '\nMy name is Foggy.\n'+\
            'Emma Cao (yc324) made me.\n'+\
            'I am the game K-in-a-Row!\n' 
    return intro
 
def nickname():
    return MY_NICKNAME
 
############################################################

# Receive and acknowledge information about the game from
# the game master:
def prepare(initial_state, k, what_side_I_play, opponent_nickname):
    # Write code to save the relevant information in either
    # global variables.
    global M, N, K, I_PLAY, INITIAL_STATE, OPPONENT_PLAYS, OPPONENT_NICKNAME, START_TIME
    INITIAL_STATE = initial_state
    I_PLAY = what_side_I_play
    K = k
    M = len(initial_state[0])   # M rows
    N = len(initial_state[0][0])   # N columns
    OPPONENT_NICKNAME = opponent_nickname
    OPPONENT_PLAYS = 'O' if what_side_I_play == 'X' else 'X'
    START_TIME = time.time()
    return "OK"
 
############################################################

def makeMove(currentState, currentRemark, timeLimit=10000):
    global TIME_LIMIT, START_TIME
    START_TIME = time.time()
    TIME_LIMIT = timeLimit

    newRemark = ""
    move = minimax(currentState, 3)[1][2]
    if move is None:
        newRemark = "OMG, no way to move :("
        return [None, newRemark]

    newState = copy.deepcopy(currentState)
    newState[0][move[0]][move[1]] = I_PLAY
    newState[1] = OPPONENT_PLAYS

    if "good" in currentRemark:
        newRemark = "Thanks! Here we go!"
    elif "bad" in currentRemark:
        newRemark = "Oooooops!!!"

    evaluation = minimax(currentState, 3)[0]
    if evaluation >= 10 ** (K + 1):
        newRemark = "Absolutely win!"
    elif evaluation >= 10 ** K:
        newRemark = "HHhhhhhhh! I believe I will win!"
    elif evaluation >= 10 ** (K - 2):
        newRemark = "How it always comes up short!"
    elif evaluation >= 10:
        newRemark = "I think I'm a little lost..."
    elif evaluation < 0:
        newRemark = "What step did I take wrong?"
    elif evaluation <= -10 ** (K - 2):
        newRemark = "Wow...You took a good step..."

    return [[move, newState], newRemark]

##########################################################################

def minimax(state, depthRemaining, pruning=False, alpha=-float('inf'), beta=float('inf'), zHashing=None):
    global TIME_LIMIT, START_TIME
    TIME_LIMIT = TIME_LIMIT - (time.time() - START_TIME)
    START_TIME = time.time()

    if depthRemaining == 0 or TIME_LIMIT <= 0.01:
        return [staticEval(state)]

    best_move = None
    if state[1] == I_PLAY:
        provisional = -float('inf')
        for s in successors(state, I_PLAY):
            newVal = minimax(s, depthRemaining - 1)
            if newVal[0] > provisional:
                provisional = newVal[0]
                best_move = s
            if pruning:
                alpha = max(alpha, provisional)
                if beta <= alpha:
                    break
    else:
        provisional = float('inf')
        for s in successors(state, OPPONENT_PLAYS):
            newVal = minimax(s, depthRemaining - 1)
            if newVal[0] < provisional:
                provisional = newVal[0]
                best_move = s
            if pruning:
                beta = min(beta, provisional)
                if beta <= alpha:
                    break
    return [provisional, best_move]

def successors(state, whoseMove):
    successor = []
    next_move = OPPONENT_PLAYS if whoseMove == I_PLAY else I_PLAY
    for r in range(M):
        for c in range(N):
            if state[0][r][c] == ' ':
                currState = copy.deepcopy(state)
                newState = [currState[0], whoseMove, [0, 0]]
                newState[0][r][c] = whoseMove
                newState[1] = next_move
                newState[2] = [r, c]
                successor.append(newState)
    return successor

##########################################################################
 
def staticEval(state):
    total_value = 0

    for row in range(M):
        total_value += checkRow(state, row)
    for col in range(N):
        total_value += checkCol(state, col)

    for row in range(M):
        total_value += checkPositiveDiagonal(state, row, 0)
    for col in range(1, N):
        total_value += checkPositiveDiagonal(state, 0, col)

    for row in range(M):
        total_value += checkNegativeDiagonal(state, row, 0)
    for col in range(1, N):
        total_value += checkNegativeDiagonal(state, M - 1, col)

    return total_value

##########################################################################

def checkRow(state, row):
    state = state[0]
    count_X = 0
    count_O = 0
    last_bash = 0

    for col in range(N):
        if state[row][col] == '-':
            remaining = N - col - 1
            if remaining < K:
                if col - last_bash > K:
                    if count_X > 0 and count_O == 0: return 10 ** (count_X - 1) * count_X
                    if count_X == 0 and count_O > 0: return -10 ** (count_O - 1) * count_O
                    return 0
                else: return 0
            else:
                count_X = 0
                count_O = 0
                last_bash = col

        if state[row][col] == I_PLAY:
            if state[row][col - 1] == I_PLAY: count_X += 2
            else: count_X += 1
        elif state[row][col] == OPPONENT_PLAYS:
            if state[row][col - 1] == OPPONENT_PLAYS: count_O += 2
            else: count_O += 1

    if count_X > 0 and count_O == 0: return 10 ** (count_X - 1) * count_X
    if count_X == 0 and count_O > 0: return -(10 ** (count_O - 1) * count_O)
    return 0

def checkCol(state, col):
    state = state[0]
    count_X = 0
    count_O = 0
    last_bash = 0

    for row in range(M):
        if state[row][col] == '-':
            remaining = len(state) - row - 1
            if remaining < K:
                if row - last_bash > K:
                    if count_X > 0 and count_O == 0: return 10 ** (count_X - 1) * count_X
                    if count_X == 0 and count_O > 0: return -10 ** (count_O - 1) * count_O
                    return 0
                else: return 0
            else:
                count_X = 0
                count_O = 0
                last_bash = row

        if state[row][col] == I_PLAY:
            if state[row - 1][col] == I_PLAY: count_X += 2
            else: count_X += 1
        elif state[row][col] == OPPONENT_PLAYS:
            if state[row - 1][col] == OPPONENT_PLAYS: count_O += 2
            else: count_O += 1

    if count_X > 0 and count_O == 0: return 10 ** (count_X - 1) * count_X
    if count_X == 0 and count_O > 0: return -(10 ** (count_O - 1) * count_O)
    return 0

def checkPositiveDiagonal(state, start_row, start_col):
    state = state[0]
    count_X = 0
    count_O = 0
    last_bash = -1

    row = start_row
    col = start_col

    while row < len(state) and col < len(state[row]):
        if state[row][col] == '-':
            remaining = min(len(state) - row, len(state[row]) - col)
            if remaining < K:
                if row - start_row - last_bash > K or col - start_col - last_bash > K:
                    if count_X > 0 and count_O == 0: return 10 ** (count_X - 1) * count_X
                    if count_X == 0 and count_O > 0: return -10 ** (count_O - 1) * count_O
                    return 0
                else: return 0
            else:
                count_X = 0
                count_O = 0
                last_bash = row - start_row

        if state[row][col] == I_PLAY:
            if state[row - 1][col - 1] == I_PLAY: count_X += 2
            else: count_X += 1
        elif state[row][col] == OPPONENT_PLAYS:
            if state[row - 1][col - 1] == OPPONENT_PLAYS: count_O += 2
            else: count_O += 1

        row += 1
        col += 1

    if count_X > 0 and count_O == 0: return 10 ** (count_X - 1) * count_X
    if count_X == 0 and count_O > 0: return -(10 ** (count_O - 1) * count_O)
    return 0

def checkNegativeDiagonal(state, start_row, start_col):
    state = state[0]
    count_X = 0
    count_O = 0
    last_bash = -1

    row = start_row
    col = start_col

    while row >= 0 and col < len(state[row]):
        if state[row][col] == '-':
            remaining = min(row + 1, len(state[row]) - col)
            if remaining < K:
                if start_row - row - last_bash > K or col - start_col - last_bash > K:
                    if count_X > 0 and count_O == 0: return 10 ** (count_X - 1) * count_X
                    if count_X == 0 and count_O > 0: return -10 ** (count_O - 1) * count_O
                    return 0
                else: return 0
            else:
                count_X = 0
                count_O = 0
                last_bash = start_row - row

        if state[row][col] == I_PLAY:
            count_X += 1
        elif state[row][col] == OPPONENT_PLAYS:
            count_O += 1

        row -= 1
        col += 1

    if count_X > 0 and count_O == 0: return 10 ** (count_X - 1) * count_X
    if count_X == 0 and count_O > 0: return -(10 ** (count_O - 1) * count_O)
    return 0