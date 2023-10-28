from queue import Queue 


INITIAL_STATE1 = \
              [[['X','X',' ',' ','X','X','X'],
                ['X','X','X','X','X','X','X'],
                [' ',' ',' ',' ',' ',' ','X'],
                [' ','X',' ',' ','X','X',' ']], "X"]

K = 4 # from 1 to k





#for item1, item2, item3, item4 in zip(INITIAL_STATE1[0][0], INITIAL_STATE1[0][1], INITIAL_STATE1[0][2], INITIAL_STATE1[0][3]):
    #print(item1, item2, item3, item4)

list1 = list(zip(INITIAL_STATE1[0][0], INITIAL_STATE1[0][1], INITIAL_STATE1[0][2], INITIAL_STATE1[0][3]))



        


num_rows = len(INITIAL_STATE1[0])

num_cols = len(INITIAL_STATE1[0][0])
#print(num_rows) # would return to you the first dash
#print(num_cols)

board = INITIAL_STATE1[0]



# create a copy of the board

def X_O_unblocked_vert(limit, player):
    list1 = list(zip(INITIAL_STATE1[0][0], INITIAL_STATE1[0][1], INITIAL_STATE1[0][2], INITIAL_STATE1[0][3]))
    board = list(zip(INITIAL_STATE1[0][0], INITIAL_STATE1[0][1], INITIAL_STATE1[0][2], INITIAL_STATE1[0][3]))
    unblocked_instances = 0
    for i in range(len(list1)):
        num_consecutive = 0
        num_spaces = 0
        in_consecutive = False
        space_consecutive = False
        queue = []
        for j in range(len(list1[i])):
            if (board[i][j] == player):
                queue.append(board[i][j])
                if (in_consecutive):
                    num_consecutive += 1
                    if (num_consecutive > limit):
                        num_consecutive = 0
                else:
                    num_consecutive = 1
                    in_consecutive = True
                    space_consecutive = False
            elif (board[i][j] != ' '): # we are blocked by the opponent
                num_consecutive = 0
                in_consecutive = False
                num_spaces = 0
                space_consecutive = False
                continue
            elif (board[i][j] == ' '):
                queue.append(board[i][j])
                if (space_consecutive):
                    num_spaces += 1
                else:
                    num_spaces += 1
                    space_consecutive = True
                    in_consecutive = False
            if (num_consecutive == limit and num_spaces >= K - num_consecutive): # see if we have available spaces and if we have enough consecutive x
                var = queue.pop(0)
                if(var == ' '):
                    num_spaces -= 1
                elif (var == player):
                    num_consecutive -= 1
                else:
                    num_consecutive = 0
                    num_spaces = 0
                unblocked_instances += 1

    return unblocked_instances






def X_O_unblocked_hori(limit, player):
    unblocked_instances = 0
    for i in range(num_rows):
        num_consecutive = 0
        num_spaces = 0
        in_consecutive = False
        space_consecutive = False
        queue = []
        for j in range(num_cols):
            if (board[i][j] == player):
                queue.append(board[i][j])
                if (in_consecutive):
                    num_consecutive += 1
                    if (num_consecutive > limit):
                        num_consecutive = 0
                else:
                    num_consecutive = 1
                    in_consecutive = True
                    space_consecutive = False
            elif (board[i][j] != ' '): # we are blocked by the opponent
                num_consecutive = 0
                in_consecutive = False
                num_spaces = 0
                space_consecutive = False
                continue
            elif (board[i][j] == ' '):
                queue.append(board[i][j])
                if (space_consecutive):
                    num_spaces += 1
                else:
                    num_spaces += 1
                    space_consecutive = True
                    in_consecutive = False
            if (num_consecutive == limit and num_spaces >= K - num_consecutive): # see if we have available spaces and if we have enough consecutive x
                var = queue.pop(0)
                if(var == ' '):
                    num_spaces -= 1
                elif (var == player):
                    num_consecutive -= 1
                else:
                    num_consecutive = 0
                    num_spaces = 0
                unblocked_instances += 1

    return unblocked_instances










        

# only works for X alone and O alone
def X_O_alone(limit, player):
    num_horizontal_lines = 0
    for i in range(num_rows):
        num_x_o = 0
        singular = True
        for j in range(num_cols):
            if (board[i][j] == player):
                num_x_o += 1
            elif (board[i][j] != ' '):
                singular = False
                break
            if (num_x_o > limit): 
                break
        if (num_x_o == limit and singular):
            num_horizontal_lines += 1
    return num_horizontal_lines




#print(X_O_alone(1, 'X'))
#print(X_O_unblocked_hori(6, 'X'))


print(X_O_unblocked_vert(2, 'X'))








#diff = X_O_alone(1, 'X') - X_O_alone(1, 'O')

#print(diff)




# first number, defines which index in array
# 0 is the board game, and first index is player turn
# second number is which row, third is which column