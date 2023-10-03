"""grammar.py

STARTER CODE VERSION <- replace with "Version by " [your name]

Defines classes for the various parts of speech
and rules for constructing phrases.

A rules has the form
CONSTRUCT ::= CONSTRUCT CONSTRUCT ... CONSTRUCT
where a construct is either a nonterminal phrase type
or a word part of speech.

A rule can have a "weight".  This is optional and
if not included, the weight defaults to 1.
To specify a weight other than 1, use the form

CONSTRUCT ::= CONSTRUCT CONSTRUCT ... CONSTRUCT @ 5

with your preferred weight where the 5 is.
These weights are used in the random.choices call
to give higher probability to more highly weighted
rules.
"""

import random

""" A production rule for a grammar normally has a
 left-hand side (a.k.a. its "head") and a right-hand side.
 For example:
   S ::= NP VP
 But here we also allow a weight, e.g.,
   S ::= NP VP @ 7.5
 so that the random selection of the rule can be
 intentionally  biased to prefer or avoid the rule
 coming up often.
"""
class Rule:
    def __init__(self, lhs, rhs, weight):
        # Begin by constructing this rule.            
        self.lhs = lhs
        self.rhs = rhs
        self.weight = weight
        # if no rule group yet exists for this lhs, create one.
        if not (lhs in RuleGroup.heads):
            RuleGroup.heads.append(lhs)
            grp = RuleGroup(lhs)
            RuleGroup.groups[lhs]=grp
        else:
            grp = RuleGroup.groups[lhs]
        # Put this rule in its group
        grp.add_rule(self)        

    def __str__(self):
        rhs_desc = ' '.join(self.rhs)
        return "(Rule:) " + self.lhs + " ::= " + rhs_desc + "; weight="+str(self.weight)

class RuleGroup:
    # All rules for one head (lhs) will go together in a group.
    heads = []
    groups = {}
    
    def __init__(self, head):
        self.head = head
        self.rules = []
        self.weights = []

    def add_rule(self, rule):
        self.rules.append(rule)
        self.weights.append(rule.weight)
        
    def choose(self, choice_mode):
        if choice_mode=="first":
            return self.choose_first()
        elif choice_mode=="last":
            return self.choose_last()
        elif choice_mode=="random":
            return self.choose_random()

    def choose_first(self):
        return self.rules[0]

    def choose_last(self):
        return self.rules[-1]

    def choose_random(self):
        # Here we use the weights when selecting a rule.
        return random.choices(self.rules, weights=self.weights)[0]

    def __str__(self):
        desc = "RuleGroup for head: "+self.head+"\n"
        for rule in self.rules:
            desc += "  "+str(rule)+"\n"
        return desc


def read_rule_reps(rule_reps):
   lines = rule_reps.split("\n")
   for line in lines:
       if len(line)<4: continue
       sides = line.split(" ::= ")
       # print("Left side: ", sides[0])
       # print("  Right side: ", sides[1])
       right = sides[1]
       right_parts = right.split(" @ ")
       if len(right_parts)==2:
           weight = eval(right_parts[1])
       else:
           weight = 1
       # print("weight is ", weight)
       rhs = right_parts[0].split(" ")
       rule = Rule(sides[0], rhs, weight)

# The VERBI and VERBT categories are intended for
# intransitive (no direct object) and transitive verbs.
# The higher weighted rules are those that tend
# to generate shorter expansions in the grammar.
# Without such weighting, the expansions can become
# VERY large too often.

ALL_RULES = """
MESSAGE ::= EVENT REACTION

EVENT ::= VERB_PHRASE
EVENT ::= NOUN_PHRASE VERB_PHRASE
EVENT ::= ANIMAL_SOUND

NOUN_PHRASE ::= NOUN @ 5
NOUN_PHRASE ::= ADJECTIVE NOUN_PHRASE @ 2
NOUN_PHRASE ::= NOUN_PHRASE PREPOSITIONAL_PHRASE

PREPOSITIONAL_PHRASE ::= PREPOSITION NOUN_PHRASE

VERB_PHRASE ::= VERBI @ 4
VERB_PHRASE ::= VERBT NOUN_PHRASE @ 2
VERB_PHRASE ::= ADVERB VERB_PHRASE
"""

# Initialize the rules.  If the rules are not
# syntactically valid, this will probably raise an
# error right here.
read_rule_reps(ALL_RULES)



