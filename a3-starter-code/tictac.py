test_board = [
    [['X', 'X', ' '],
     ['O', 'X', 'O'],
     ['X', ' ', ' '],
     ['X', ' ', ' ']], 'O'
]

player = 'O'
depth = 0 # Set the search depth

K = 3
num_rows = len(test_board[0])
num_columns = len(test_board[0][0])
#print(num_rows)
#print(num_columns)

def test_diagonals(current_state):
    current_state = current_state[0]
    result = [row for row in current_state]
    possibe_combos = []

    #print(result)
    #for i in range(len(result)):
        #for j in range(len(result[i])):
            #print(len(result))
            #if ((len(result) - (i + j)) >= K):
                #list = []
                #for z in range(len(result) - (i + j)):
                    #print(z)
                    #list.append(result[i+z][j+z])
                #print(list)
                #possibe_combos.append(list)
            #else:
                #continue
    diagonals = []

    for i in range(num_rows - K + 1):
        for j in range(num_columns - K + 1):
            diagonal = []
            for z in range(K):
                diagonal.append(result[i + z][j + z])
            diagonals.append(diagonal)


    for i in range(num_rows - K + 1):
        for j in range(K - 1, num_columns):
            diagonal = []
            for z in range(K):
                diagonal.append(result[i + z][j - z])
            diagonals.append(diagonal)



    return diagonals

p = test_diagonals(test_board)

print(p)

def X_O_lines(state, limit, player):
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


print(X_O_lines(p, 2, 'X'))