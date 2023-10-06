"""
PriorityQueue.py

Contains the implementation for the custom class My_Priority_Queue that
implements a special kind of priority queue.
 Steve Tanimoto, Univ. of Washington.
 Paul G. Allen School of Computer Science and Engineering
 April 6, 2021.
"""


class My_Priority_Queue:
    def __init__(self):
        self.q = []  # Actual data goes in a list.

    def __contains__(self, elt):
        """If there is a (state, priority) pair on the list
        where state==elt, then return True."""
        # print("In My_Priority_Queue.__contains__: elt= ", str(elt))
        for pair in self.q:
            if pair[0] == elt: return True
        return False

    def delete_min(self):
        """ Standard priority-queue dequeuing method."""
        if self.q == []: return []  # Simpler than raising an exception.
        temp_min_pair = self.q[0]
        temp_min_value = temp_min_pair[1]
        temp_min_position = 0
        for j in range(1, len(self.q)):
            if self.q[j][1] < temp_min_value:
                temp_min_pair = self.q[j]
                temp_min_value = temp_min_pair[1]
                temp_min_position = j
        del self.q[temp_min_position]
        return temp_min_pair

    def insert(self, state, priority):
        """We do not keep the list sorted, in this implementation."""
        # print("calling insert with state, priority: ", state, priority)

        if self[state] != -1:
            print("Error: You're trying to insert an element into a My_Priority_Queue instance,")
            print(" but there is already such an element in the queue.")
            return
        self.q.append((state, priority))

    def __len__(self):
        """We define length of the priority queue to be the
        length of its list."""
        return len(self.q)

    def __getitem__(self, state):
        """This method enables Pythons right-bracket syntax.
        Here, something like  priority_val = my_queue[state]
        becomes possible. Note that the syntax is actually used
        in the insert method above:  self[state] != -1  """
        for (S, P) in self.q:
            if S == state: return P
        return -1  # This value means not found.

    def __delitem__(self, state):
        """This method enables Python's del operator to delete
        items from the queue."""
        # print("In MyPriorityQueue.__delitem__: state is: ", str(state))
        for count, (S, P) in enumerate(self.q):
            if S == state:
                del self.q[count]
                return

    def __str__(self):
        """Code to create a string representation of the PQ."""
        txt = "My_Priority_Queue: ["
        for (s, p) in self.q: txt += '(' + str(s) + ',' + str(p) + ') '
        txt += ']'
        return txt
