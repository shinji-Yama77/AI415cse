'''
yongqz2KInARow.py
Author: Yongqin Zhao
An agent for playing "K-in-a-Row with Forbidden Squares"
CSE 415, University of Washington

THIS IS A TEMPLATE WITH STUBS FOR THE REQUIRED FUNCTIONS.
YOU CAN ADD WHATEVER ADDITIONAL FUNCTIONS YOU NEED IN ORDER
TO PROVIDE A GOOD STRUCTURE FOR YOUR IMPLEMENTATION.
'''
import copy
import time
import random
# Global variables to hold information about the opponent and game version:
INITIAL_STATE = None
OPPONENT_NICKNAME = 'Eazzy Pezzy'
OPPONENT_PLAYS = 'O' # Update this after the call to prepare.

# Information about this agent:
MY_LONG_NAME = 'Jornsen Chao'#'Templatus Skeletus'
MY_NICKNAME = 'Jornsen the 415 survivor'
I_PLAY = 'X' # Gets updated by call to prepare.


# GAME VERSION INFO
M = 0
N = 0
K = 0
TIME_LIMIT = 0

############################################################

############################################################
# INTRODUCTION
def introduce():
    intro = '\nMy name is'+ MY_LONG_NAME +' \n'+ '.' +\
            '"An instructor" made me.\n'+\
            'Somebody please turn me into a real game-playing agent!\n'
    return intro

def nickname():
    return MY_NICKNAME

############################################################


# player1.prepare(INITIAL_STATE, K, 'X', player2.nickname())
def prepare(initial_state, k, what_side_I_play, opponent_nickname):
    # Write code to save the relevant information in either
    # global variables.
    global INITIAL_STATE, K, I_PLAY, OPPONENT_NICKNAME
    INITIAL_STATE = initial_state
    K = k
    I_PLAY = what_side_I_play
    OPPONENT_NICKNAME = opponent_nickname
    return "OK"

############################################################
remarks = [
    "Hmm, let's try this!",
    "You are pushing me so hard!",
    "This is a good move.",
    "Hope this will surprise you.",
    "Eat this.",
]
def evaluateState(state):
    return state.count('X'), state.count('O')
def generateRemark(currentState, currentRemark):
    num_X, num_O = evaluateState(currentState[0])

    # Commenting on the game state
    if num_X > num_O:
        remarks.append("You're leading.")
    elif num_O > num_X:
        remarks.append("I'm leading.")

    # Interactivity: Respond to the player's previous remark
    if "good move" in currentRemark.lower():
        remarks.append("Thank you! Let's see how this turns out.")
    if "surprise" in currentRemark.lower():
        remarks.append("No big deal.")
    if "pushing" in currentRemark.lower():
        remarks.append("I am not going to apologize for that.")
    # Choose a random remark from the updated list
    newRemark = random.choice(remarks)

    return newRemark
############################################################

def makeMove(currentState, currentRemark, timeLimit=10000):
    # initialize player side, best_move, bestEval
    depth = 1
    print("makeMove has been called")
    player = currentState[1]
    print("code to compute a good move should go here.")
    best_move = None
    best_eval = - 10**(K+5) if player == 'X' else 10**(K+5)
    best_eval = float(best_eval)

    for child_state in getDirectChildren(currentState):
        child_eval = minimax(child_state, depth)
        if player == 'X' and child_eval[0] > best_eval:
            best_eval = child_eval[0]
            best_move = child_state
        elif player == 'O' and child_eval[0] < best_eval:
            best_eval = child_eval[0]
            best_move = child_state

    # compute distance between current and best_move
    move = [0, 0] # This might be legal ONCE in a game, if the square is not forbidden or already occupied.
    for i in range(len(currentState[0])):
        for j in range(len(currentState[0][0])):
            if currentState[0][i][j] == ' ' and best_move[0][i][j] == player:
                move = [i, j]
                break

    newState = best_move
    newRemark = generateRemark(currentState, currentRemark)

    print("Returning from makeMove")
    return [[move, newState], newRemark]


##########################################################################

def minimax(state, depthRemaining, alpha=float('-inf'), beta=float('inf'), zHashing=None):
    # base case, return [staticEval of state, None]
    is_end = isEnd(state)
    if depthRemaining == 0 or is_end:
        return [staticEval(state), None]


    if state[1] == 'X':  # EXPECT A LARGE NUMBER
        maxEval = -10**(K+5)
        best_move = None
        for s in getDirectChildren(state):
            eval_result = minimax(s, depthRemaining-1, alpha, beta, zHashing)
            if eval_result[0] > maxEval:
                maxEval = eval_result[0]
                best_move = s  # or other information you want
            alpha = max(alpha, maxEval)
            if beta <= alpha:
                break
        return [maxEval, best_move]

    else:  # state[1] == 'O'
        minEval = 10**(K+5)
        best_move = None
        for s in getDirectChildren(state):
            eval_result = minimax(s, depthRemaining-1, alpha, beta, zHashing)
            if eval_result[0] < minEval:
                minEval = eval_result[0]
                best_move = s  # or other information you want
            beta = min(beta, minEval)
            if beta <= alpha:
                break
        return [minEval, best_move]



def getDirectChildren (state):
    children = []
    # tempState = copy.deepcopy(state)
    for i in range(len(state[0])):
        for j in range(len(state[0][0])):
            if state[0][i][j] == ' ':
                tempState = copy.deepcopy(state)
                tempState[0][i][j]=tempState[1]
                tempState[1] = 'X' if tempState[1] == 'O' else 'O'
                children.append(tempState)
    return children

def isEnd(state):
    for row in state[0]:
        if ' ' in row:
            return False
    return True
##########################################################################
# calculate score of given state
def staticEval(state):

    my_side = state[1]
    oppo_side = 'O' if my_side == 'X' else 'X'
    board = state[0]
    k = K
    return helperAllMatrix(board, my_side, k) - helperAllMatrix(board, oppo_side, k)

def helperAllMatrix(board, my_side, k):
    # horizontal
    horizontal_val=sum(calculate_value_in_a_row(row,my_side, k) for row in board)
    # vertical
    transposed_board = [list(row) for row in zip(*board)]
    vertical_val=sum(calculate_value_in_a_row(row,my_side, k) for row in transposed_board)
    # diagonal
    diagonals = []
    for i in range(len(board) * 2 - 1):
        main_diag = []
        anti_diag = []
        for j in range(max(0, i - len(board) + 1), min(i + 1, len(board))):
            main_diag.append(board[j][i - j])
            anti_diag.append(board[len(board) - 1 - j][i - j])
        diagonals.append(main_diag)
        diagonals.append(anti_diag)
    diagonal_val = sum(calculate_value_in_a_row(diagonal,my_side, k) for diagonal in diagonals)


    return horizontal_val + vertical_val + diagonal_val

def calculate_value_in_a_row(lst, my_side,k):
    potential_ct =0
    space_ct =0
    # check from left to right
    for char in lst:
        # calculate potential length and space length within which
        if char in [' ', my_side]:
            potential_ct += 1
            if char == ' ':
                space_ct += 1
        else:
            if potential_ct >= k:
                return 10**(k - space_ct)
            potential_ct, space_ct = 0, 0

    return 10 ** (k - space_ct) if potential_ct >= k else 0



