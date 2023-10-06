'''puzzle3.py
An instance of the Eight Puzzle.
'''

from EightPuzzle import *

# We simply redefine the initial state.

init_state_list = [[6, 3, 0], 
                   [7, 4, 1], 
                   [8, 5, 2]]

CREATE_INITIAL_STATE = lambda: State(init_state_list)


