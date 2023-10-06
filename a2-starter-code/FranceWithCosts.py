'''FranceWithCosts.py
("Route Planning in France" problem)
'''
#<METADATA>
SOLUTION_VERSION = "2.0"
PROBLEM_NAME = "France-Trip Planning: Driving from Rennes to Avignon"
PROBLEM_VERSION = "1.0"
PROBLEM_AUTHORS = ['S. Tanimoto']
PROBLEM_CREATION_DATE = "23-JAN-2019"

# The following field is mainly for the human solver, via either the Text_SOLUTION_Client.
# or the SVG graphics client.
PROBLEM_DESC=\
 '''The <b>"
France-Trip Planning"</b> problem is to find a shortest driving route from the
city of Rennes to the city of Avignon, using the map data provided.
'''
#</METADATA>

#<COMMON_DATA>
STARTING_CITY = "Rennes"
DESTINATION_CITY = "Avignon"
STATES = {}

ADJ = {}
ADJ['Brest'] = ['Rennes']
ADJ['Rennes'] = ['Caen','Paris','Brest','Nantes']
ADJ['Caen'] = ['Calais','Paris','Rennes']
ADJ['Calais'] = ['Nancy','Paris','Caen']
ADJ['Nancy'] = ['Strasbourg','Dijon','Paris','Calais']
ADJ['Strasbourg'] = ['Dijon','Nancy']
ADJ['Dijon'] = ['Strasbourg','Lyon','Paris','Nancy']
ADJ['Lyon'] = ['Grenoble','Avignon','Limoges','Dijon']
ADJ['Grenoble'] = ['Avignon','Lyon']
ADJ['Avignon'] = ['Grenoble','Marseille','Montpellier','Lyon']
ADJ['Marseille'] = ['Nice','Avignon']
ADJ['Nice'] = ['Marseille']
ADJ['Montpellier'] = ['Avignon','Toulouse']
ADJ['Toulouse'] = ['Montpellier','Bordeaux','Limoges']
ADJ['Bordeaux'] = ['Limoges','Toulouse','Nantes']
ADJ['Limoges'] = ['Lyon','Toulouse','Bordeaux','Nantes','Paris']
ADJ['Nantes'] = ['Limoges','Bordeaux','Rennes']
ADJ['Paris'] = ['Calais','Nancy','Dijon','Limoges','Rennes','Caen']

DISTANCE = {}
DISTANCE['Brest'] = {'Rennes':244}
DISTANCE['Rennes'] = {'Caen':176,'Paris':348,'Brest':244,'Nantes':107}
DISTANCE['Caen'] = {'Calais':120,'Paris':241,'Rennes':176}
DISTANCE['Calais'] = {'Nancy':534,'Paris':297,'Caen':120}
DISTANCE['Nancy'] = {'Strasbourg':145,'Dijon':201,'Paris':372,'Calais':534}
DISTANCE['Strasbourg'] = {'Dijon':335,'Nancy':145}
DISTANCE['Dijon'] = {'Strasbourg':335,'Lyon':192,'Paris':313,'Nancy':201}
DISTANCE['Lyon'] = {'Grenoble':104,'Avignon':216,'Limoges':389,'Dijon':192}
DISTANCE['Grenoble'] = {'Avignon':227,'Lyon':104}
DISTANCE['Avignon'] = {'Grenoble':227,'Marseille':99,'Montpellier':212,'Lyon':216}
DISTANCE['Marseille'] = {'Nice':188,'Avignon':99}
DISTANCE['Nice'] = {'Marseille':188}
DISTANCE['Montpellier'] = {'Avignon':212,'Toulouse':240}
DISTANCE['Toulouse'] = {'Montpellier':240,'Bordeaux':253,'Limoges':313}
DISTANCE['Bordeaux'] = {'Limoges':220,'Toulouse':253,'Nantes':329}
DISTANCE['Limoges'] = {'Lyon':389,'Toulouse':313,'Bordeaux':220,'Nantes':329,'Paris':396}
DISTANCE['Nantes'] = {'Limoges':329,'Bordeaux':329,'Rennes':107}
DISTANCE['Paris'] = {'Calais':297,'Nancy':372,'Dijon':313,'Limoges':396,'Rennes':348,'Caen':241}
#</COMMON_DATA>

#<COMMON_CODE>

class State():

  def __init__(self, name="no name yet"):

    self.name = name

  def __eq__(self,s2):
    #print("In State.__eq__: s2 is ", str(s2))
    return self.name==s2.name

  def __str__(self):
    return self.name

  def __hash__(self):
    return (self.__str__()).__hash__()

  def copy(self):
    # Performs an appropriately deep copy of a state,
    # for use by operators in creating new states.
    news = State()
    news.name = self.name
    return news 

  def ith_neighbor_exists(self,i):
    '''Tests whether there are enough adjacent cities
    to go to the ith.'''
    return len(ADJ[self.name])>i

  def move(self,i):
    '''Assuming it's legal to transition to the ith neighbor,
    this does it.'''
    neighbor = STATES[ADJ[self.name][i]]
    return neighbor

  def edge_distance(self, s2):
    return DISTANCE[self.name][s2.name]

def goal_test(s):
  return s.name==DESTINATION_CITY

def goal_message(s):
  return "Congratulations on finding a route to Avignon!"

class Operator:
  def __init__(self, name, precond, state_transf):
    self.name = name
    self.precond = precond
    self.state_transf = state_transf

  def is_applicable(self, s):
    return self.precond(s)

  def apply(self, s):
    return self.state_transf(s)

def create_all_states():
  for name in ADJ.keys():
    STATES[name]=State(name)

create_all_states()
#</COMMON_CODE>

#<INITIAL_STATE>
CREATE_INITIAL_STATE = lambda : STATES[STARTING_CITY]
#</INITIAL_STATE>

#<OPERATORS>

OPERATORS = [Operator(
  "Go to neighboring city number "+str(i),
  lambda s, i1=i: s.ith_neighbor_exists(i1),
  lambda s, i1=i: s.move(i1))
             for i in range(6)] # Paris has the most neighbors (6)
#</OPERATORS>

#<GOAL_TEST> (optional)
GOAL_TEST = lambda s: goal_test(s)
#</GOAL_TEST>

#<GOAL_MESSAGE_FUNCTION> (optional)
GOAL_MESSAGE_FUNCTION = lambda s: goal_message(s)
#</GOAL_MESSAGE_FUNCTION>
