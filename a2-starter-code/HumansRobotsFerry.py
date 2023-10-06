'''HumansRobotsFerry.py
("Humans, Robots and Ferry" problem)
'''
#<METADATA>
SOLUTION_VERSION = "2.0"
PROBLEM_NAME = "Humans, Robots, and Ferry"
PROBLEM_VERSION = "1.0"
PROBLEM_AUTHORS = ['S. Tanimoto']
PROBLEM_CREATION_DATE = "06-APR-2021"

# The following field is mainly for the human solver, via either the Text_SOLUZION_Client.
# or the SVG graphics client.
PROBLEM_DESC=\
 '''The <b>"Humans, Robots and Ferry"</b> problem is a variation of
the classic puzzle "Missionaries and Cannibals." In the Humans, Robots
and Ferry problem, the player starts off with three humans and three
robots on the left bank of a creek.  The object is to execute a
sequence of legal moves that transfers them all to the right bank of
the creek.  In this puzzle, there is a ferry that can carry at most
three agents (humans, robots), and one of them must be a human to steer
the ferry.  It is forbidden to ever have one or two humans outnumbered
by robots, either on the left bank, right bank, or on the ferry.
In the formulation presented here, the computer will not let you make a
move to such a forbidden situation, and it will only show you moves
that could be executed "safely."
'''
#</METADATA>

#<COMMON_DATA>
#</COMMON_DATA>

#<COMMON_CODE>
H=0  # array index to access human counts
R=1  # same idea for robots
LEFT=0 # same idea for left side of creek
RIGHT=1 # etc.

class State():

  def __init__(self, d=None):
    if d==None: 
      d = {'agents':[[0,0],[0,0]],
           'ferry':LEFT}
    self.d = d

  def __eq__(self,s2):
    for prop in ['agents', 'ferry']:
      if self.d[prop] != s2.d[prop]: return False
    return True

  def __str__(self):
    # Produces a textual description of a state.
    p = self.d['agents']
    txt = "\n H on left:"+str(p[H][LEFT])+"\n"
    txt += " R on left:"+str(p[R][LEFT])+"\n"
    txt += "   H on right:"+str(p[H][RIGHT])+"\n"
    txt += "   R on right:"+str(p[R][RIGHT])+"\n"
    side='left'
    if self.d['ferry']==1: side='right'
    txt += " ferry is on the "+side+".\n"
    return txt

  def __hash__(self):
    return (self.__str__()).__hash__()

  def copy(self):
    # Performs an appropriately deep copy of a state,
    # for use by operators in creating new states.
    news = State({})
    news.d['agents']=[self.d['agents'][H_or_R][:] for H_or_R in [H, R]]
    news.d['ferry'] = self.d['ferry']
    return news 

  def can_move(self,h,r):
    '''Tests whether it's legal to move the ferry and take
     h humans and r robots.'''
    side = self.d['ferry'] # Where the ferry is.
    p = self.d['agents']
    if h<1: return False # Need an H to steer boat.
    h_available = p[H][side]
    if h_available < h: return False # Can't take more h's than available
    r_available = p[R][side]
    if r_available < r: return False # Can't take more r's than available
    h_remaining = h_available - h
    r_remaining = r_available - r
    # Humans must not be outnumbered on either side:
    if h_remaining > 0 and h_remaining < r_remaining: return False
    h_at_arrival = p[H][1-side]+h
    r_at_arrival = p[R][1-side]+r
    if h_at_arrival > 0 and h_at_arrival < r_at_arrival: return False
    return True


  def move(self,h,r):
    '''Assuming it's legal to make the move, this computes
     the new state resulting from moving the ferry carrying
     h humans and r robots.'''
    news = self.copy()      # start with a deep copy.
    side = self.d['ferry']        # where is the ferry?
    p = news.d['agents']          # get the array of arrays of agents.
    p[H][side] = p[H][side]-h     # Remove agents from the current side.
    p[R][side] = p[R][side]-r
    p[H][1-side] = p[H][1-side]+h # Add them at the other side.
    p[R][1-side] = p[R][1-side]+r
    news.d['ferry'] = 1-side      # Move the ferry itself.
    return news

def goal_test(s):
  '''If all Ms and Cs are on the right, then s is a goal state.'''
  p = s.d['agents']
  return (p[H][RIGHT]==3 and p[R][RIGHT]==3)

def goal_message(s):
  return "Congratulations on successfully guiding the humans and robots across the creek!"

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
CREATE_INITIAL_STATE = lambda : State(d={'agents':[[3, 0], [3, 0]], 'ferry':LEFT })
#</INITIAL_STATE>

#<OPERATORS>
HR_combinations = [(1,0),(2,0),(3,0),(1,1),(2,1)]

OPERATORS = [Operator(
  "Cross the creek with "+str(h)+" humans and "+str(r)+" robots",
  lambda s, h1=h, r1=r: s.can_move(h1,r1),
  lambda s, h1=h, r1=r: s.move(h1,r1) ) 
  for (h,r) in HR_combinations]
#</OPERATORS>

#<GOAL_TEST> (optional)
GOAL_TEST = lambda s: goal_test(s)
#</GOAL_TEST>

#<GOAL_MESSAGE_FUNCTION> (optional)
GOAL_MESSAGE_FUNCTION = lambda s: goal_message(s)
#</GOAL_MESSAGE_FUNCTION>
