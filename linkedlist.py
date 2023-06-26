# class definitions

class listNode:
    # self.next points to the next node in the list
    # self.val returns the value of the node
    def __init__(self, val=0, next=None) -> None:
        self.val = val
        self.next = next
        # if u want to modify the class to make it bidirectional
        # self.prev = prev  


# helper functions

def toarray(head):
    # given a head node, prints the linked list
    curr = head
    arr = []
    while curr is not None:
        arr.append(curr.val)
        curr=curr.next
    return arr

def findtail(head):
    # given a head node, find the tail node of the linked list
    tail = head
    while tail.next is not None:
        tail = tail.next
    return tail

def findnth(head, n):
    temp = head
    for i in range(n):
        temp = temp.next
    return temp


print('define a linkedlist 0->1->2')
head = listNode(0)
head.next = listNode(1)
head.next.next = listNode(2)

print(toarray(head))

# ----------------------------------------------------------------
print('find the 2nd (0-index) node of the linkedlist')
print(findnth(head, 2).val)

# ----------------------------------------------------------------

print('adding node 3 at tail')

tail = findtail(head)
tail.next = listNode(3)

print(toarray(head))


# ----------------------------------------------------------------

print('adding node -1 at head')

newhead = listNode(-1)
newhead.next = head
head = newhead

print(toarray(head))
# ----------------------------------------------------------------

print('adding node 2.5 between 2 and 3')

node2 = findnth(head, 2)

newNode = listNode(2.5)
newNode.next = node2.next
node2.next = newNode

print(toarray(head))


# ----------------------------------------------------------------

print('removing node 2.5')

node2 = findnth(head, 2)
node2.next = node2.next.next

print(toarray(head))
