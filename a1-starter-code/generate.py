"""generate.py

Version by Shinji Yamashita

This will generate messages randomly, using the given grammar and vocabulary.
Orignally created, Sept. 12, 2023.
for CSE 415, Paul G. Allen School of Computer Science and Engineering,
Univ. of Washington.

The purpose of this program is to demonstrate how Python can be used
in a generative artificial intelligence context. Here, we'll use a
"small language model" to generate text messages.
The probabilistic model here is simply made up, rather than learned
from any data.
"""

import random

# Get grammar and vocabulary read in.

from grammar import RuleGroup
import vocabulary as vocab

# Set up a dictionary to map parts of speech to lists of words.
WORD_GROUPS = {k: vocab.Words(eval("vocab."+k)) for k in vocab.POS_KEYS}

# Set the mode for choosing among rules and words:
CHOICE_MODE = "random"  # Normal "creative" mode.
# CHOICE_MODE = "first"  # Needed by autograders.
# CHOICE_MODE = "last"   #   "     "      "
# Note that "first" or "last" mode can cause runaway recursion,
# if the first or last grammar production in any group is recursive.
# Guideline: make sure the first production in any group is
# not recursive. Don't worry about the last production.

# For debugging only:
def show_word_groups():
    for k in WORD_GROUPS.keys():
        print(k + ": ")
        print(WORD_GROUPS[k])

RULE_GROUPS = RuleGroup.groups
show_word_groups()
#print(RULE_GROUPS['NOUN_PHRASE'])
#print(RULE_GROUPS['NOUN_PHRASE'].choose_first())

# Function to generate some text starting with a particular construct,
# such as MESSAGE, VERB_PHRASE, or ADJECTIVE.
def gen_text(construct):

    text = "a default value, which your code below should replace"

    # Test whether the construct is a part of speech by seeing if
    # if is in vocab.POS_KEYS.
    # If so, return a chosen word from the appropriate word group.
    #  (Get the appropriate word group, and then call its 'choose' method.)

    # (STUDENT CODE GOES HERE)
    if construct in vocab.POS_KEYS:
        chose_word = WORD_GROUPS.get(construct)
        text = chose_word.choose(CHOICE_MODE)
        return text
    else:
        # get the appropriate word group by getting it's left hand side, rule group
        word_group = RULE_GROUPS[construct].choose(CHOICE_MODE)
        return " ".join([gen_text(word) for word in word_group.rhs])



    # Otherwise, get the rule group for this construct, and
    # then choose one of its rules.
    
    # (STUDENT CODE GOES HERE)



    # Next get the right-hand side (rhs) of this rule.
    # There will be one or more constructs in this rhs.
    # Recursively generate text for each of them.
    # Suggestion: make use of a list comprehension here.
    # Then join the strings together with spaces between them.
    # Consider using the 'join' method of strings for this.

    # (STUDENT CODE GOES HERE)

    # Then return the resulting string.

def gen_overall_message():
    # The overall text will be a concatenation of several messages.
    if CHOICE_MODE=='random':
        n = 7  # Students: Replace this with code that
        # sets n to a random integer between 1 and 8, inclusive.

    else:
      n = 2
    text = ""
    for i in range(n):
        text += gen_text('MESSAGE')+".  "
    return text

if __name__=='__main__':
    print(gen_overall_message())

