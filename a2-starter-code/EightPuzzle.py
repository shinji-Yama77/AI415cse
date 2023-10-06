'''EightPuzzle.py
'''
#<METADATA>
QUIET_VERSION = "0.2"
PROBLEM_NAME = "Eight Puzzle"
PROBLEM_VERSION = "0.2"
PROBLEM_AUTHORS = ['S. Tanimoto']
PROBLEM_CREATION_DATE = "8-JAN-2018"
PROBLEM_DESC=\
'''This formulation of the Eight Puzzle uses generic
Python 3 constructs and has been tested with Python 3.6.
It is designed to work according to the QUIET2 tools interface.
'''
#</METADATA>

#<COMMON_DATA>
#</COMMON_DATA>

#<COMMON_CODE>
class State:
  def __init__(self, b):
    if len(b)==9:
      list_of_lists = [b[:3],b[3:6],b[6:]]
    else:
      list_of_lists = b
    self.b = list_of_lists

  def __eq__(self,s2):
    for i in range(3):
      for j in range(3):
        if self.b[i][j] != s2.b[i][j]: return False
    return True

  def __str__(self):
    # Produces a textual description of a state.
    # Might not be needed in normal operation with GUIs.
    txt = "\n["
    for i in range(3):
      txt += str(self.b[i])+"\n "
    return txt[:-2]+"]"

  def __hash__(self):
    return (self.__str__()).__hash__()

  def copy(self):
    # Performs an appropriately deep copy of a state,
    # for use by operators in creating new states.
    news = State({})
    news.b = [row[:] for row in self.b]
    return news

  def find_void_location(self):
    '''Return the (vi, vj) coordinates of the void.
    vi is the row index of the void, and vj is its column index.'''
    for i in range(3):
      for j in range(3):
        if self.b[i][j]==0:
          return (i,j)
    raise Exception("No void location in state: "+str(self))

  def can_move(self,dir):
    '''Tests whether it's legal to move a tile that is next
       to the void in the direction given.'''
    (vi, vj) = self.find_void_location()
    if dir=='N': return vi<2
    if dir=='S': return vi>0
    if dir=='W': return vj<2
    if dir=='E': return vj>0
    raise Exception("Illegal direction in can_move: "+str(dir))

  def move(self,dir):
    '''Assuming it's legal to make the move, this computes
       the new state resulting from moving a tile in the
       given direction, into the void.'''
    news = self.copy() # start with a deep copy.
    (vi, vj) = self.find_void_location()
    b = news.b
    if dir=='N': 
      b[vi][vj] = b[vi+1][vj]
      b[vi+1][vj] = 0
    if dir=='S':
      b[vi][vj] = b[vi-1][vj]
      b[vi-1][vj] = 0
    if dir=='W':
      b[vi][vj] = b[vi][vj+1]
      b[vi][vj+1] = 0
    if dir=='E':
      b[vi][vj] = b[vi][vj-1]
      b[vi][vj-1] = 0
    return news # return new state

  def edge_distance(self, s2):
    return 1.0  # Warning, this is only correct when
    # self and s2 are neighboring states.
    # We assume that is the case.  This method is
    # provided so that problems having all move costs equal to
    # don't have to be handled as a special case in the algorithms.
  
def goal_test(s):
  '''If all the b values are in order, then s is a goal state.'''
  return s == State([[0,1,2],[3,4,5],[6,7,8]])

def goal_message(s):
  return "You've got all eight straight. Great!"

class Operator:
  def __init__(self, name, precond, state_transf):
    self.name = name
    self.precond = precond
    self.state_transf = state_transf

  def is_applicable(self, s):
    return self.precond(s)

  def apply(self, s):
    return self.state_transf(s)
#</COMMON_CODE>

#<INITIAL_STATE>
  # Use default, but override if new value supplied
             # by the user on the command line.
try:
  import sys
  init_state_string = sys.argv[2]
  print("Initial state as given on the command line: "+init_state_string)
  init_state_list = eval(init_state_string)
except:
  init_state_list = [[3, 1, 2], [0, 5, 8], [4, 6, 7]]
  print("Using default initial state list: "+str(init_state_list))
  print(" (To use a specific initial state, enter it on the command line, e.g.,")
  print("python3 UCS.py EightPuzzle '[[3, 1, 2], [0, 4, 5], [6, 7, 8]]'")

CREATE_INITIAL_STATE = lambda: State(init_state_list)
#</INITIAL_STATE>

#<OPERATORS>
directions = ['N','E','W','S']
OPERATORS = [Operator("Move a tile "+str(dir)+" into the void",
                      lambda s,dir1=dir: s.can_move(dir1),
                      # The default value construct is needed
                      # here to capture the value of dir
                      # in each iteration of the list comp. iteration.
                      lambda s,dir1=dir: s.move(dir1) )
             for dir in directions]
#</OPERATORS>

#<GOAL_TEST> (optional)
GOAL_TEST = lambda s: goal_test(s)
#</GOAL_TEST>

#<GOAL_MESSAGE_FUNCTION> (optional)
GOAL_MESSAGE_FUNCTION = lambda s: goal_message(s)
#</GOAL_MESSAGE_FUNCTION>

