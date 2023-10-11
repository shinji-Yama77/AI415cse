from EightPuzzle import *

GOAL_STATE = [[0,1,2],[3,4,5],[6,7,8]]
# for testing
# test = State([[0,8,1],[7,5,4],[3,6,5]])


def h(puzzle_state):
    ' returns the total cost of number of tiles out of place'
    cost = 0
    s = puzzle_state.b
    for i, sublist in enumerate(s):
        for j, num in enumerate(sublist):
            if num == 0:
                continue
            elif num != GOAL_STATE[i][j]:
                cost += 1

    return cost



# for testing
#print(h(test))