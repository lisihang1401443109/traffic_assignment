from collections import deque

LENGTH = 15

class stack:
    def __init__(self) -> None:
        self.deque = deque()

    def pop(self):
        return self.deque.pop()
    
    def push(self, value):
        self.deque.append(value)

    def display(self):
        print(list(self.deque))


class queue:
    def __init__(self) -> None:
        self.deque = deque()

    def enqueue(self, value):
        self.deque.append(value)
    
    def dequeue(self):
        return self.deque.popleft()
    
    def display(self):
        print(list(self.deque))
    
class node:
    def __init__(self,  val = 0, left = None, right = None) -> None:
        self.left = left
        self.right = right
        self.val = val

    def left(self):
        return self.left
    
    def right(self):
        return self.right
    
    def val(self):
        return self.val
    
    def nodeprint(self):
        if self.left:
            self.left.nodeprint()
        print(self.val)
        if self.right:
            self.right.nodeprint()

class bst:
    def __init__(self) -> None:
        self.head = None

    def insert(self, value):
        if self.head is None:
            self.head = node(value)
        else:
            curr = self.head
            while curr is not None:
                if curr.val < value:
                    if curr.right is None:
                        curr.right = node(value)
                        return
                    else:
                        curr = curr.right
                if curr.val >= value:
                    if curr.left is None:
                        curr.left = node(value)
                        return
                    else:
                        curr = curr.left
        
    def find(self, value) -> node:
        if self.head is None:
            return False
        else:
            curr = self.head
            while curr is not None:
                if curr.val < value:
                    if curr.right is None:
                        return False
                    else:
                        curr = curr.right
                if curr.val > value:
                    if curr.left is None:
                        return False
                    else:
                        curr = curr.left
                else:
                    return True
    
    def delete(self, value):
        if not self.find(value):
            return False
        
        curr = self.head
        parent = None
        while curr is not None:
            if curr.val < value:
                if curr.right is None:
                    return False
                else:
                    parent = curr
                    curr = curr.right
            if curr.val > value:
                if curr.left is None:
                    return False
                else:
                    parent = curr
                    curr = curr.left
            else:
                # found it, delete
                if curr.left is None:
                    if curr.right is None:
                        if parent.val < curr.val:
                            # parent.right == curr
                            parent.right = None
                        else:
                            parent.left = None
                        return
                    else:
                        parent.right = curr.right
                elif curr.right is None:
                    # only have right child
                    parent.left = curr.left
                else:
                    # have both left and right child
                    minpar = curr.right
                    min = curr.right
                    if min.left is None:
                        curr.val = min.val
                        curr.right = None
                        return
                    while min.left is not None:
                        minpar = min
                        min = min.left
                    minpar.left = None
                    curr.val = min.val
                return
                    
                    

    
    def display(self) -> None:
        self._display(self.head, LENGTH)
    
    def _display(self, head, offset):
        if head is None:
            print('nil')
            return
        self._display(head.left, offset-4)
        print((' '*offset) + '\\\n' + (' '*offset) + str(head.val) + '\n' + (' '*offset) + '/')
        self._display(head.right, offset-4)

    def treeprint(self) -> None:
        self.head.nodeprint()        

    

print('intializing stack with [1, 2, 3, 4, 5]')
s = stack()
s.push(1)
s.push(2)
s.push(3)
s.push(4)
s.push(5)
s.display()

print('popping stack five times until empty')
s.pop()
s.pop()
s.pop()
s.pop()
s.pop()
print('displaying stack after poping')
s.display()

print('initializing queue with [1, 2, 3, 4, 5]')
s = queue()
s.enqueue(1)
s.enqueue(2)
s.enqueue(3)
s.enqueue(4)
s.enqueue(5)

s.display()
print('dequeueing queue five times until empty')
s.dequeue()
s.dequeue()
s.dequeue()
s.dequeue()
s.dequeue()

print('displaying quee after five dequeues')
s.display()


print('initializing bst with 4, 2, 5, 3, 7 (note that order matters)')
tree = bst()
tree.insert(4)
tree.insert(2)
tree.insert(5)
tree.insert(3)
tree.insert(7)
tree.display()

# print('deleting 3 (deleting a leaf)')
# tree.delete(3)
# tree.display()

# print('deleting 5 (deleting a non-leaf with one child)')
# tree.delete(5)
# tree.display()

# print('deleting 4 (deleting a non-leaf with both child)')
# tree.delete(4)
# tree.display()

tree.treeprint()