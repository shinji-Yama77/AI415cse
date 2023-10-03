import math
import re
def is_a_triple(n):
    """Return True if n is a multiple of 3; False otherwise."""
    if (n % 3 == 0):
        return True
    return False
def last_prime(m):
    """Return the largest prime number p that is less than or equal to m.
    You might wish to define a helper function for this.
    You may assume m is a positive integer."""
    for i in range(m, 0, -1):
        if (check_prime(i)):
            return i


def check_prime(n):
    for i in range(n-1, 1, -1):
        if (n % i == 0):
            return False
    return True



def quadratic_roots(a, b, c):
    """Return the roots of a quadratic equation (real cases only).
    Return results in tuple-of-floats form, e.g., (-7.0, 3.0)
    Return "complex" if real roots do not exist."""
    list1 = []
    determinant = pow(b, 2) - (4 * a * c)
    if (determinant < 0):
        return "complex"
    first = (-b + math.sqrt(determinant)) / (2 * a)
    second = (-b - math.sqrt(determinant)) / (2 * a)
    list1.append(first)
    list1.append(second)
    return tuple(list1)


def new_quadratic_function(a, b, c):
    """Create and return a new, anonymous function (for example
    using a lambda expression) that takes one argument x and 
    returns the value of ax^2 + bx + c."""
    return lambda x: a * pow(x, 2) + b*x + c


def perfect_shuffle(even_list):
    """Assume even_list is a list of an even number of elements.
    Return a new list that is the perfect-shuffle of the input.
    Perfect shuffle means splitting a list into two halves and then interleaving
    them. For example, the perfect shuffle of [0, 1, 2, 3, 4, 5, 6, 7] is
    [0, 4, 1, 5, 2, 6, 3, 7]."""
    new = []
    to_split_num = len(even_list) // 2
    list1 = even_list[:to_split_num]
    list2 = even_list[to_split_num:]
    for i in range(to_split_num):
        new.append(list1[i])
        new.append(list2[i])
    return new




def list_of_3_times_elts_plus_1(input_list):
    """Assume a list of numbers is input. Using a list comprehension,
    return a new list in which each input element has been multiplied
    by 3 and had 1 added to it."""
    list_3_plus_1 = [(x*3) + 1 for x in input_list]
    return list_3_plus_1

def triple_vowels(text):
    """Return a new version of text, with all the vowels tripled.
    For example:  "The *BIG BAD* wolf!" => "Theee "BIIIG BAAAD* wooolf!".
    For this exercise assume the vowels are
    the characters A,E,I,O, and U (and a,e,i,o, and u).
    Maintain the case of the characters."""
    dict1 = {
        "a": "aaa",
        "e": "eee",
        "i": "iii",
        "o": "ooo",
        "u": "uuu",
        "A": "AAA",
        "E": "EEE",
        "I": "III",
        "O": "OOO",
        "U": "UUU"
    }
    new = ""
    for char in text:
        result = dict1.get(char, None)
        if(result):
            new += result
        else:
            new += char
    return new

def count_words(text):
    """Return a dictionary having the words in the text as keys,
    and the numbers of occurrences of the words as values.
    Assume a word is a substring of letters and digits and the characters
    '-', '+', *', '/', '@', '#', '%', and "'" separated by whitespace,
    newlines, and/or punctuation (characters like . , ; ! ? & ( ) [ ] { } | : ).
    Convert all the letters to lower-case before the counting."""
    # convert letters to lowercase
    lower_case_text = text.lower()
    # should match on anything else except these what counts as word 
    # using negation in regex
    
    delimiters = r'[^a-z0-9\-\+\*\'/@#%]+' 
    # only preserve the words if there it isn't empty in list
    result = [s for s in re.split(delimiters, lower_case_text) if s]
    dict1 = {}
    for word in result:
        if word in dict1:
            dict1[word] += 1
        else:
            dict1[word] = 1
    return dict1



    

