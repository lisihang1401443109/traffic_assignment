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