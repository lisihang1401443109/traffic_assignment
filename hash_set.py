def error(msg):
    print(f'error: {msg}')

class Deleted:

    def __str__(self) -> str:
        return 'deleted'
    
    def __repr__(self) -> str:
        return 'deleted'
    
class Empty:

    def __str__(self) -> str:
        return 'empty'
    
    def __repr__(self) -> str:
        return 'empty'

DELETED = Deleted()
EMPTY = Empty()

class HashSet:

    array : list
    size : int
    flag : list

    def __init__(self, size=7) -> None:
        self.size = size
        self.array = [EMPTY] * self.size
    

    def _hash(self, elt) -> int:
        """
        Calculate the hash value for the given element.

        Args:
            elt (int): The element to calculate the hash value for.

        Returns:
            int: The calculated hash value.
        """
        return elt % self.size
    
    def add(self, elt) -> None:
        """
        Adds an element to the data structure.

        Parameters:
            elt: The element to be added.

        Returns:
            None
        """
        self._add(elt, elt)

    def _add(self, elt, _elt) -> None:
        """
        Adds an element to the set.

        Parameters:
            elt (Any): The element to be added to the set.
            _elt (int): used to calculate the position of the element in cases of collision and probing

        Returns:
            None

        Raises:
            Error: If the set is full or if the element already exists in the set.
        """
        if _elt == elt + self.size:
            error('set is full')
        if self.array[self._hash(_elt)] in [EMPTY, DELETED]:
            self.array[self._hash(_elt)] = elt
            return
        if self.array[self._hash(_elt)] == elt:
            error('duplicate')
        else:
            self._add(elt, _elt+1)

    def find(self, elt) -> bool:
        """
        Find the given element in the data structure.

        Args:
            elt: The element to find.

        Returns:
            bool: True if the element is found, False otherwise.
        """
        return self._find(elt, elt)
    

    def _find(self, elt, _elt) -> bool:
        """
        Recursive function to find an element in the array.

        Parameters:
            elt (Any): The element to find in the array.
            _elt (int): The current index in the array to be checked.

        Returns:
            bool: True if the element is found, False otherwise.
        """
        if _elt == elt + self.size:
            return False
        if self.array[self._hash(_elt)] == EMPTY:
            return False
        if self.array[self._hash(_elt)] == elt:
            return True
        else:
            return self._find(elt, _elt+1)
        
    def delete(self, elt) -> None:
        """
        Deletes an element from the data structure.

        Args:
            elt: The element to be deleted.

        Returns:
            None
        """
        self._delete(elt, elt)

    def _delete(self, elt, _elt):
        """
        Deletes an element from the set.

        Parameters:
            elt (int): The element to be deleted from the set.
            _elt (int): The position of the index to be checked in cases of collision and probing

        Returns:
            None

        Raises:
            ValueError: If the element is not found in the set.

        Notes:
            This function recursively searches for the element in the set and deletes it if found. If the element is not found, a ValueError is raised.
        """
        if _elt == elt + self.size:
            error('element not in set')
        if self.array[self._hash(_elt)] == EMPTY:
            return
        if self.array[self._hash(_elt)] == elt:
            self.array[self._hash(_elt)] = DELETED
            return
        else:
            self._delete(elt, _elt+1)

    def display(self) -> None:
        """
        Display the contents of the array.

        This function prints the contents of the array to the console.

        Parameters:
            self (ClassName): The instance of the class.
        
        Returns:
            None: This function does not return anything.
        """
        print(self.array)


s = HashSet()

'''
    TODO: generate testcate using s that demostrates hash collision and linear probing
'''

print('adding 3 and 5 to a hashset of size 7')
s.add(3)
s.add(5)
s.display()

print('adding 10, 4 to the hashset (collision and linear probing)')

s.add(10)
s.add(4)
s.display()

print('deleting 3, 5 from the hashset')
s.delete(3)
s.delete(5)
s.display()

print(f'{s.find(10)=}, {s.find(4)=}, {s.find(17)=}, {s.find(5)=}, {s.find(3)=}')

print('adding 17 to the hashset')
s.add(17)
s.display()