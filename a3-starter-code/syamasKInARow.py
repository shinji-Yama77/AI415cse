'''
syamasKInARow.py
Author: <Shinji Yamashita>
An agent for playing "K-in-a-Row with Forbidden Squares"
CSE 415, University of Washington

THIS IS A TEMPLATE WITH STUBS FOR THE REQUIRED FUNCTIONS.
YOU CAN ADD WHATEVER ADDITIONAL FUNCTIONS YOU NEED IN ORDER
TO PROVIDE A GOOD STRUCTURE FOR YOUR IMPLEMENTATION.

'''

import time
import copy

# Global variables to hold information about the opponent and game version:
INITIAL_STATE = None
OPPONENT_NICKNAME = 'Christmas'
OPPONENT_PLAYS = 'O' # Update this after the call to prepare.

# Information about this agent:
MY_LONG_NAME = 'Shinji Yama'
MY_NICKNAME = 'Big-boy'
I_PLAY = 'X' # Gets updated by call to prepare.

 
# GAME VERSION INFO
M = 0
N = 0
K = 0
TIME_LIMIT = 0
 
 
############################################################
# INTRODUCTION
def introduce():
    intro = f'''\nMy name is {MY_LONG_NAME}.\n'+\
            '"Shinji Yama" made me.\n'+\
            'I am a Japanese artificial Intelligence agent\n'''
    return intro
 
def nickname():
    return f'''{MY_NICKNAME}'''
 
############################################################

# Receive and acknowledge information about the game from
# the game master:
def prepare(initial_state, k, what_side_I_play, opponent_nickname):
    # Write code to save the relevant information in either
    # global variables.
    global M, N, K, INITIAL_STATE, I_PLAY, OPPONENT_PLAYS, OPPONENT_NICKNAME
    INITIAL_STATE = initial_state[0]
    #player_turn = initial_state[1]
    K = k
    M = len(initial_state[0]) # number of rows
    N = len(initial_state[0][0]) # number of columns
    I_PLAY = what_side_I_play
    if I_PLAY == 'O':
        OPPONENT_PLAYS = 'X'
    else:
        OPPONENT_PLAYS = 'O'
    OPPONENT_NICKNAME = opponent_nickname
    #print(player_turn + 'playing') # we can keep track of who is playing
    return "OK"
 
############################################################
 
def makeMove(currentState, currentRemark, timeLimit=10000):
    global TIME_LIMIT
    print("makeMove has been called")
    # call the minimax here 

    start_time = time.time()
    move = None
    newState = None
    score = 0
    TIME_LIMIT = timeLimit
    
    if (generate_successors(currentState[0]) == 0):
        return [None, "utterance"]
    # get all possible moves from the current state
    new_score, new_State = minimax(currentState, depthRemaining=3, start_time=start_time, pruning=True, alpha=float('-inf'), beta=float('inf'))
    newState = new_State
    #newRemark = make_remark(newState, prevState)
    move = get_move(currentState, new_State)

    newRemark = make_remark(currentState, move, new_State, currentRemark)

    

    return [[move, newState], newRemark]
    

##########################################################################


# The main adversarial search function:
def minimax(state, depthRemaining, start_time, pruning=False, alpha=None, beta=None, zHashing=None):
    player_turn = state[1]
    if (TIME_LIMIT - (time.time() - start_time) < 0.3):
        return [staticEval(state[0]), state]

    if (depthRemaining == 0 or generate_successors(state[0]) == 0): # see if depth=0 or no more possible moves
        return [staticEval(state[0]), state]
    if (player_turn == 'X'): # agent's turn to play
        maxEval = float('-inf')
        best_state = None
        for i in range(M):
            for j in range(N):
                if (state[0][i][j] == ' '): # check if empty
                    # create deepy copy
                    new_state = copy.deepcopy(state)
                    new_state[0][i][j] = 'X'
                    new_state[1] = 'O'
                    curr_val, now_state = minimax(new_state, depthRemaining=depthRemaining-1, start_time=start_time, pruning=pruning, alpha=alpha, beta=beta)
                    alpha = max(alpha, curr_val)
                    if (pruning):
                        if alpha >= beta:
                            break
                    if (curr_val > maxEval):
                        best_state = copy.deepcopy(new_state)
                        maxEval = max(maxEval, curr_val)
        return maxEval, best_state
    else:           
        minEval = float('inf')
        best_state = None
        for i in range(M):
            for j in range(N):
                if (state[0][i][j] == ' '): # check if empty
                    new_state = copy.deepcopy(state)
                    new_state[0][i][j] = 'O'
                    new_state[1] = 'X'
                    curr_val, now_state = minimax(new_state, depthRemaining=depthRemaining-1, start_time=start_time, pruning=pruning, alpha=alpha, beta=beta)
                    beta = min(beta, curr_val)
                    if (pruning):
                        if alpha >= beta:
                            break
                    if (curr_val < minEval):
                        best_state = copy.deepcopy(new_state)
                        minEval = min(minEval, curr_val)
                        beta = minEval
        return minEval, best_state

##########################################################################

def staticEval(state):
    total_points = 0

    for i in range(1, K+1):
        diff = all_lines(state, i, I_PLAY) - all_lines(state, i, OPPONENT_PLAYS) 
        if (i == K):
            if(diff > 0):
                diff += 1000
            else:
                diff -= 1000
        total_points += diff*10**i

    return total_points

##########################################################################

# calculate all horizontal and vertical number of X's or O's in a row
def all_lines(state, limit, player): 
    total = 0
    transposed_state = [list(col) for col in zip(*state)]   
    horizontal_lines = X_O_lines(state, limit, player, M, N)
    vertical_lines = X_O_lines(transposed_state, limit, player, N, M)
    diags = X_O_diags_lines(test_diagonals(state), limit, player)
    total += horizontal_lines + vertical_lines + diags
    return total

   
# works for 2 and above to K, doesn't count for single lines
def X_O_lines(state, limit, player, rows, cols):
    unblocked_instances = 0
    for i in range(rows):
        for j in range(cols):
            max_num_consecutive = 0 # check max number consecutive 
            in_player_consecutive = False
            num_x = 0
            num_consecutive = 0
            num_spaces = 0
            if (j+K <= N):
                for k in range(K):
                    if(state[i][j+k] == player):
                        num_x += 1
                        if (in_player_consecutive):
                            num_consecutive += 1
                            if (num_consecutive > max_num_consecutive):
                                max_num_consecutive = num_consecutive
                            continue
                        else:
                            num_consecutive = 1
                            if (num_consecutive >= max_num_consecutive):
                                max_num_consecutive = num_consecutive # counting for one case
                            in_player_consecutive = True
                    elif (state[i][j+k] != ' '):
                        break
                    else:
                        num_spaces += 1
                        in_player_consecutive = False
                if (max_num_consecutive == limit and num_spaces >= 1): # least number of spaces
                    unblocked_instances += 1
    return unblocked_instances




# see if there are any possible moves from current state
def generate_successors(state): 
    count = 0
    for i in range(M):
        for j in range(N):
            if (state[i][j] == ' '): # see if there any possible states from current state
                count += 1

    return count


# choose move from given new state and previous state. expects change in only one row
def get_move(prevState, newState):
    #import pdb; pdb.set_trace()
    for i in range(M):
        for j in range(N):
            if newState[0][i][j] != prevState[0][i][j]:
                return [i, j]


# return remarks
def make_remark(current_state, move, newState, currentRemark):
    state = current_state[0]
    state1 = newState[0]
    comment = ""

    if (generate_successors(state) <= 1):
        return "Looks like the game is over! Who dares places a move!"
    
    score = staticEval(state)
    newScore = staticEval(state1)
    # respond to game state
    if (newScore - score) > 100:
        comment += "You boutta get beaten my boy!"
    elif newScore - score < 0:
        comment += "Gotcha! Blocked you didn't I"
    else:
        comment += "Game is still on!"

    return comment




# how many diagonal possibilities
def test_diagonals(current_state):
    result = [row for row in current_state]
    diagonals = []

    for i in range(M - K + 1):
        for j in range(N - K + 1):
            diagonal = []
            for z in range(K):
                diagonal.append(result[i + z][j + z])
            diagonals.append(diagonal)

    for i in range(M - K + 1):
        for j in range(K - 1, N):
            diagonal = []
            for z in range(K):
                diagonal.append(result[i + z][j - z])
            diagonals.append(diagonal)
    return diagonals


def X_O_diags_lines(state, limit, player):
    unblocked_instances = 0
    num_rows = len(state)
    num_cols = len(state[0])
    for i in range(num_rows):
        max_num_consecutive = 0 # check max number consecutive 
        in_player_consecutive = False
        num_x = 0
        num_consecutive = 0
        num_spaces = 0
        for j in range(num_cols):
            if(state[i][j] == player):
                num_x += 1
                if (in_player_consecutive):
                    num_consecutive += 1
                    if (num_consecutive > max_num_consecutive):
                        max_num_consecutive = num_consecutive
                    continue
                else:
                    num_consecutive = 1
                    if (num_consecutive >= max_num_consecutive):
                        max_num_consecutive = num_consecutive # counting for one case
                    in_player_consecutive = True
            elif (state[i][j] != ' '):
                break
            else:
                num_spaces += 1
                in_player_consecutive = False
        if (max_num_consecutive == limit and num_spaces >= 1): # least number of spaces
            unblocked_instances += 1

    return unblocked_instances

