'''Farmer_Fox.py
by Shinji Yamashita
UWNetID: syamas
Student number: 2022314

Assignment 2, in CSE 415, Autumn 2023.
 
This file contains my problem formulation for the problem of
the Farmer, Fox, Chicken, and Grain.
'''

#<METADATA>
SOLUTION_VERSION = "2.0"
PROBLEM_NAME = "Farmer, Fox, and Chicken"
PROBLEM_VERSION = "1.0"
PROBLEM_AUTHORS = ['Y. Shinji']
PROBLEM_CREATION_DATE = "06-OCT-2023"
#</METADATA>





# Put your formulation of the Farmer-Fox-Chicken-and-Grain problem here.
# Be sure your name, uwnetid, and 7-digit student number are given above in 
# the format shown.



#<COMMON_CODE>

H=0  # array index to access farmer
F=1  # array index to access fox
C=2  # array index to access chicken
G=3  # array index to access grain
LEFT=0 # for left side of river
RIGHT=1 # for right side or river

class State():

  def __init__(self, d=None):
    if d==None: 
      d = {'agents':[[0,0],[0,0], [0, 0], [0, 0]],
           'boat':LEFT}
    self.d = d

  def __eq__(self,s2):
    for prop in ['agents', 'boat']:
      if self.d[prop] != s2.d[prop]: return False
    return True

  def __str__(self):
    # Produces a textual description of a state.
    p = self.d['agents']
    txt = "\n H on left:"+str(p[H][LEFT])+"\n"
    txt += " F on left:"+str(p[F][LEFT])+"\n"
    txt += " C on left:"+str(p[C][LEFT])+"\n"
    txt += " G on left:"+str(p[G][LEFT])+"\n"
    txt += "   H on right:"+str(p[H][RIGHT])+"\n"
    txt += "   F on right:"+str(p[F][RIGHT])+"\n"
    txt += "   C on right:"+str(p[C][RIGHT])+"\n"
    txt += "   G on right:"+str(p[G][RIGHT])+"\n"
    side='left'
    if self.d['boat']==1: side='right'
    txt += " boat is on the "+side+".\n"
    return txt
  
  def __hash__(self):
    return (self.__str__()).__hash__()
  
  def copy(self):
    # Performs an appropriately deep copy of a state,
    # for use by operators in creating new states.
    news = State({})
    news.d['agents']=[self.d['agents'][H_F_C_G][:] for H_F_C_G in [H, F, C, G]] 
    # for each single row, get all the columns with slicing notation
    news.d['boat'] = self.d['boat']
    return news 

  def can_move(self,h,f,c,g):
    '''Tests whether it's legal to move the ferry with the farmer and
    take a chicken, grain, or a fox.'''
    side = self.d['boat'] # Where the ferry is.
    p = self.d['agents']
    if h<1: return False # Need an H(farmer) to steer boat.
    h_available = p[H][side]
    if f+c+g > 1: return False # can only take one item at a time
    # can't take more farmers then available
    if h_available < h: return False # Can't take more h's than available
    F_available = p[F][side] # how many foxes on this side
    if F_available < f: return False # can't take more foxes than available
    C_available = p[C][side] # how many chickens on this side
    if C_available < c: return False # can't take more chickens than available
    G_available = p[G][side] # how much grain is available
    if G_available < g: return False
    # can't have fox and chicken on current side
    # can't have chicken and grain on current side
    f_remaining = F_available - f
    c_remaining = C_available - c
    g_remaining = G_available - g
    if f_remaining and c_remaining: return False
    if c_remaining and g_remaining: return False 
    # can't have fox and chicken on arrival side
    # can't have chicken and grain on arrival side
    #h_at_arrival = p[H][1-side]+h # how many farmers
    #f_at_arrival = p[F][1-side]+f # how many foxes on new side
    #c_at_arrival = p[C][1-side]+c # how many chickens on new side
    #g_at_arrival = p[G][1-side]+g # how many grains on new side
    return True
  
  def move(self,h,f,c,g):
    '''Assuming it's legal to make the move, this computes
     the new state resulting from moving the ferry carrying
     one farmer and one item'''
    news = self.copy()      # start with a deep copy.
    side = self.d['boat']        # where is the ferry?
    p = news.d['agents']          # get the array of arrays of agents.
    p[H][side] = p[H][side]-h     # Remove agents from the current side.
    p[F][side] = p[F][side]-f
    p[C][side] = p[C][side]-c
    p[G][side] = p[G][side]-g
    p[H][1-side] = p[H][1-side]+h # Add farmer at the other side.
    p[F][1-side] = p[F][1-side]+f # add fox to other side
    p[C][1-side] = p[C][1-side]+c # add chicken to other side
    p[G][1-side] = p[G][1-side]+g # add grain to other side
    news.d['boat'] = 1-side      # Move the ferry itself.
    return news   
  


def goal_test(s):
  '''If all Ms and Cs are on the right, then s is a goal state.'''
  p = s.d['agents']
  return (p[H][RIGHT]==1 and p[F][RIGHT]==1 and p[C][RIGHT] == 1 and p[G][RIGHT] == 1)

def goal_message(s):
  return "Congratulations on successfully getting the fox, chicken, grain across the river!"


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
CREATE_INITIAL_STATE = lambda : State(d={'agents':[[1, 0], [1, 0], [1, 0], [1, 0]], 'boat':LEFT })
#</INITIAL_STATE>

#<OPERATORS>
HR_combinations = [(1, 0, 0, 0),(1, 1, 0 ,0),(1, 0, 1, 0),(1, 0, 0, 1)]

OPERATORS = [Operator(
  "Cross the creek with "+str(h)+" human and "+str(f)+" fox and "+str(c)+" chicken and "+str(g)+" grain" ,
  lambda s, h1=h, f1=f, c1=c, g1=g : s.can_move(h1,f1,c1,g1),
  lambda s, h1=h, f1=f, c1=c, g1=g: s.move(h1,f1,c1,g1) ) 
  for (h,f,c,g) in HR_combinations]
#</OPERATORS>


#<GOAL_TEST> (optional)
GOAL_TEST = lambda s: goal_test(s)
#</GOAL_TEST>

#<GOAL_MESSAGE_FUNCTION> (optional)
GOAL_MESSAGE_FUNCTION = lambda s: goal_message(s)
#</GOAL_MESSAGE_FUNCTION>



b = State() 
print(b.copy().__str__())









