INITIAL_STATE = \
              [[['X',' ','X','X'],
                ['X','-','X','X'],
                ['X','-',' ','X']], "X"]


#K = 3 # from 1 to k


import copy

#board = INITIAL_STATE[0]
num_rows = len(INITIAL_STATE[0])
print(num_rows)
#num_cols = len(INITIAL_STATE[0][0])



# static evaluation for horionztal lines from 1 to K
def X_O_hori(state, limit, player):
    unblocked_instances = 0
    for i in range(3):
        for j in range(3):
            max_num_consecutive = 0 # check max number consecutive 
            in_player_consecutive = False
            num_x = 0
            num_consecutive = 0
            num_spaces = 0
            if (j+K <= 3):
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



def X_O_vert(state, limit, player):
    #list1 = list(zip(INITIAL_STATE[0][0], INITIAL_STATE[0][1], INITIAL_STATE[0][2]))
    #board = list(zip(INITIAL_STATE[0][0], INITIAL_STATE[0][1], INITIAL_STATE[0][2]))

    transposed_board = [list(col) for col in zip(*state)]
    unblocked_instances = 0
    for i in range(3):
        for j in range(3):
            max_num_consecutive = 0 # check max number consecutive 
            in_player_consecutive = False
            num_x = 0
            num_consecutive = 0
            num_spaces = 0
            if (j+K <= 3):
                for k in range(K):
                    if(transposed_board[i][j+k] == player):
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
                    elif (transposed_board[i][j+k] != ' '):
                        break
                    else:
                        num_spaces += 1
                        in_player_consecutive = False

                if (max_num_consecutive == limit and num_spaces >= 1): # least number of spaces
                    unblocked_instances += 1
    return unblocked_instances







#print(calc_score())

                            
#print(X_O_hori(1, 'O'))                    
#print(X_O_hori(2, 'X'))                  


#print(X_O_vert(2, 'X'))    


def minimax1(state, depthRemaining, pruning=False, alpha=None, beta=None, zHashing=None):
    player_turn = state[1]
    #curr_state = state[0]
    #best_move = [0, 0]
    if (depthRemaining == 0 or generate_successors(state[0]) == 0):
        return [staticEval(state), state]
    if (player_turn == 'X'): # agent's turn to play
        maxEval = float('-inf')
        best_state = None
        best_move = None
        for i in range(3):
            for j in range(3):
                if (state[0][i][j] == ' '): # check if empty
                    # create deepy copy
                    new_state = copy.deepcopy(state)
                    new_state[0][i][j] = 'X'
                    new_state[1] = 'O'
                    #move = [i, j]
                    curr_val, now_state = minimax1(new_state, depthRemaining=depthRemaining-1)
                    print(curr_val)
                    if (curr_val > maxEval):
                        best_state = copy.deepcopy(new_state)
                        maxEval = max(maxEval, curr_val)
        return maxEval, best_state
    else:           
        minEval = float('inf')
        best_state = None
        best_move = None
        for i in range(3):
            for j in range(3):
                if (state[0][i][j] == ' '): # check if empty
                    new_state = copy.deepcopy(state)
                    new_state[0][i][j] = 'O'
                    new_state[1] = 'X'
                    #move = [i, j]
                    curr_val, now_state = minimax1(new_state, depthRemaining=depthRemaining-1)
                    print(curr_val)
                    if (curr_val < minEval):
                        best_state = copy.deepcopy(new_state)
                        minEval = min(minEval, curr_val)
        return minEval, best_state


K = 3

def generate_successors(state): 
    count = 0

    for i in range(3):
        for j in range(3):
            if (state[i][j] == ' '): # see if there any possible states from current state
                count += 1
    
    return count



def staticEval(state):
    this_state = state[0]
    total_points = 0
    for i in range(1, K+1):
        diff = X_O_hori(this_state, i, 'X') - X_O_hori(this_state, i, 'O')  
        total_points += diff*2**i

    return total_points

test_board = [
    [[' ', 'X', ' '],
     ['O', ' ', 'O'],
     ['X', ' ', ' ']], 'O'
]

player = 'O'
depth = 0 # Set the search depth

test_diagonals = 





# Call minimax with the test case
#score, best_state = minimax1([test_board, player], depth)

#print(score)
#print(best_state)


#print(X_O_vert(test_board, 2, 'X'))
#print(X_O_hori(test_board, 2, 'X'))



#board = [list(zip(*row)) for row in test_board]
#print(board)



#transposed_board = [list(col) for col in zip(*test_board)]
#print(transposed_board)
    


# Verify the results


# Loop through columns


#print(X_O_hori(test_board, 2, player))


