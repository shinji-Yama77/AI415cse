"""vocabulary.py

STARTER CODE VERSION <- replace with "Version by " [your name]

Establishes the word categories and the words.
The categories are used in the rules in grammar.py.

"""
import random

# An instance of the class Words will contain one of
# the lists of words, e.g., the NOUN list as members.
# The class adds methods for selecting from the list.
class Words:

    def __init__(self, members):
        self.members = members

    def choose(self, choice_mode):
        if choice_mode=="first":
            return self.choose_first()
        elif choice_mode=="last":
            return self.choose_last()
        elif choice_mode=="random":
            return self.choose_random()

    def choose_first(self):
        return self.members[0]

    def choose_last(self):
        return self.members[-1]

    def choose_random(self):
        # Assume items are equally likely.
        return random.choice(self.members)
        
NOUN = ['guy', 'girl', 'boss', 'meeting', 'bike', 'turkey']

VERBI = ['went away', 'ranted', 'quit', 'fainted']

VERBT = ['avoided', 'scolded', 'devoured']

ADVERB = ['totally', 'barely', 'quickly', 'slowly']

PREPOSITION = ['in', 'after', 'over', 'under', 'beyond', 'with']

ADJECTIVE = ['humongous', 'mini', 'crazy', 'best']

REACTION = ['lol', '+1', '--gr8', 'nice', '- sad', 'omg', '... bfd', 'idc', '(too bad)']

POS_KEYS = ['NOUN', 'VERBI', 'VERBT', 'ADJECTIVE',
            'ADVERB', 'PREPOSITION', 'REACTION']

