

import time

# Global variables to hold information about the opponent and game version:
INITIAL_STATE = None
OPPONENT_NICKNAME = 'Not yet known'
OPPONENT_PLAYS = 'O' # Update this after the call to prepare.

# Information about this agent:
MY_LONG_NAME = 'Templatus Skeletus'
MY_NICKNAME = 'Tea-ess'
I_PLAY = 'X' # Gets updated by call to prepare.

 
# GAME VERSION INFO
M = 0 # rows
N = 0 # columns
K = 0
TIME_LIMIT = 0
 
 
############################################################
# INTRODUCTION
def introduce():
    # TODO
    intro = '\nMy name is Templatus Skeletus.\n'+\
            '"An instructor" made me.\n'+\
            'Somebody please turn me into a real game-playing agent!\n' 
    return intro
 
def nickname():
    # TODO
    return MY_NICKNAME
 
############################################################

# Receive and acknowledge information about the game from
# the game master:
def prepare(initial_state, k, what_side_I_play, opponent_nickname):
    # Write code to save the relevant information in either
    # global variables.
    INITIAL_STATE = initial_state
    K = k
    I_PLAY = what_side_I_play
    # TODO what if agent in game with itself?
    if I_PLAY == 'X': OPPONENT_PLAYS = 'O'
    else: OPPONENT_PLAYS = 'X'
    OPPONENT_NICKNAME = opponent_nickname
    # TODO preprocessing:
    print("Change this to return 'OK' when ready to test the method.")
    return "OK"
 
############################################################
 
def makeMove(currentState, currentRemark, timeLimit=10000):
    print("makeMove has been called")

    print("code to compute a good move should go here.")
    
    newRemark = "I need to think of something appropriate.\n" +\
    "Well, I guess I can say that this move is probably illegal."

    # take time into consideration
    if timeLimit > 5000:
        newRemark = "I have plenty of time. Well, I guess I will make the best move then."
    elif timeLimit < 1000:
        newRemark = "This move might not be the best. I made it because there wasn't enough time!"

    n = 1 # iterative depth
    items = None
    while timeLimit != 0:
        items = minimax(currentState, n)
        n += 1
    new_state = items[1]
    new_move = items[2]
    return [[new_move, new_state], newRemark]
 
 
##########################################################################
 
# The main adversarial search function:
def minimax(state, depthRemaining, pruning=False, alpha=None, beta=None, zHashing=None):
    print("Calling minimax. We need to implement its body.")
    provisional = 0

    if depthRemaining == 0: staticEval(state[0])
    if state[1] == I_PLAY: provisional = -100000
    else: provisional = 100000
    best_s = None # save the state that's the best move for min/max
    best_m = None # save the best move for min/max
    for s, m in zip(successors(state)):
        newVal = minimax(state, depthRemaining-1)
        if (state[1] == I_PLAY and newVal > provisional) or (state[1] == OPPONENT_PLAYS and newVal < provisional):
            provisional = newVal
            best_s = s
            best_m = m
    return [provisional, s, m]
    # return [default_score, "my own optional stuff", "more of my stuff"]
    # Only the score is required here but other stuff can be returned
    # in the list, after the score.
 

# computes all possible moves the current player can make
def successors(state):
    possible_states = []
    possible_moves = []
    for i in range(M):
        for j in range(N):
            if state[0][i][j] == ' ':
                state[0][i][j] == state[1]
                possible_moves.append([i, j])
                possible_states.append(state[0])
    return [possible_states, possible_moves]

##########################################################################
 
def staticEval(state):
    print('calling staticEval. Its value needs to be computed!')
    # Values should be higher when the states are better for X,
    # lower when better for O.
    # TODO
    f_list = [0 for i in range(3)]
    # # of Xs and Os per row
    for i in range(3):
        x_count, o_count = 0, 0
        for j in range(3):
            if state[i][j] == 'X': x_count += 1
            elif state[i][j] == 'O': o_count += 1
        if o_count == 0:
            for i in range(1, 3):
                if x_count == i: f_list[i-1] += 1
        elif x_count == 0:
            for i in range(1, 3):
                if o_count == i: f_list[i-1] -= 1
        print(f_list)
    for i in range(3):
        x_count, o_count = 0, 0
        for j in range(3):
            if state[j][i] == 'X': x_count += 1
            elif state[j][i] == 'O': o_count += 1
        if o_count == 0:
            for i in range(1, 3):
                if x_count == i: f_list[i-1] += 1
        elif x_count == 0:
            for i in range(1, 3):
                if o_count == i: f_list[i-1] -= 1
        print(f_list)
    return f_list[0] + f_list[1] * 10 + f_list[2] * 100


def test_statEval(state):
    row = 0
    num_consec = {'X': 0, 'O': 0}
    num_moves = {'X': 0, 'O': 0}
    i = 0
    while i < len(state[row]):
        while i < len(state[row]) and state[row][i] == 'X':
            if not i == 0 or not i == len(state[row]) - 1:
                num_moves['X'] += 1
            num_consec['X'] += 1
            i += 1
        i+=1
    print(num_consec, num_moves)