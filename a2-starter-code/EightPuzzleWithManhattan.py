from EightPuzzle import *

GOAL_STATE = [[0,1,2],[3,4,5],[6,7,8]]
# for testing
#test = State([[4,0,2],[1,6,3],[7,8,5]])


store_distance = {num: (i, j) for i, sublist in enumerate(GOAL_STATE) for j, num in enumerate(sublist)}


def h(puzzle_state):
    ' returns the total cost of manhattan distance calculated by x and y'
    cost = 0
    s = puzzle_state.b
    for i, sublist in enumerate(s):
        for j, num in enumerate(sublist):
            if num != 0:
                dist_x = abs(store_distance[num][1] - j)
                dist_y = abs(store_distance[num][0] - i)
                cost += dist_y + dist_x
                
    return cost



# for testing
#print(h(test))