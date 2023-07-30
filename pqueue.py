import math

'''
 we are implementing the heap using an array:
    the tree: 
            3
           / \
          5   7
        /  \ /  \
       6  9  10 11

    is represented as:
    [3, 5, 7, 6, 9, 10, 11]
    and we can access the parents and children of any node using the functions below
 
'''
def parent(index) -> int:
    return int((index-1)/2)

def left(index) -> int:
    return int(index*2+1)

def right(index) -> int:
    return int(index*2+2)

class heap:

    data: list

    def __init__(self) -> None:
        self.data = []

    def __len__(self) -> int:
        return len(self.data)
    
    def __iter__(self) -> iter:
        return self.data.__iter__()
    
    # insertion
    def insert(self, value) -> None:
        # adding the value to the end of the array (last leaf) and heapify up
        self.data.append(value)
        if self.__len__() > 1:
            self.heapify_up()

    def _swap(self, i, j) -> None:
        temp = self.data[i]
        self.data[i] = self.data[j]
        self.data[j] = temp


    def heapify_up(self):
        i = self.__len__()-1
        # while we have not reached the root
        while i:
            if self.data[i] > self.data[parent(i)]:
                # if the child is greater than that of the parent
                # end loop
                break
            # if the child is less than its parent, swap i and the parent of i
            self._swap(i, parent(i))
            i = parent(i)
    
    # poping: remove the root, and move the last leaf to the root
    #         and begin heapify down
    def pop(self):
        ret = self.data[0]
        self.data[0] = self.data[-1]
        self.data = self.data[:-1]
        self.heapify_down()
        return ret

    def heapify_down(self) -> None:
        i = 0
        # three scenarios: having no children, having one children, and having both children
        while i < self.__len__():
            if right(i) < self.__len__():
                # having both children
                left_d = self.data[left(i)]
                right_d = self.data[right(i)]
                # we seek to swap the parent with the smaller child
                smaller = left(i) if left_d <= right_d else right(i)
            elif left(i) < self.__len__():
                # having one children
                smaller = left(i)
            else:
                # having no children, no need to swap, end loop
                break
            if self.data[i] < self.data[smaller]:
                # if the parent is already smaller than the smaller children, do nothing, end loop
                break
            # otherwise swap the parent with the smaller child
            self._swap(i, smaller)
            i = smaller

    def __str__(self) -> str:
        total_length = len(self) * 2
        ret = ''
        it = self.data.__iter__()
        for i in range(0, int(math.log2(self.__len__()))+1):
            for j in range(int(math.pow(2, i))):
                ret += ' ' * int((total_length/(i+1)))
                try:   
                    ret += (str(next(it)))
                except:
                    break
            ret += ' ' * int((total_length/(i+1)))
            ret += ('\n')
        return ret



h = heap()

print('inserting 1, 3, 2, 25, 13, 16 =================================================================')

h.insert(1)
h.insert(3)
h.insert(2)
h.insert(25)
h.insert(13)
h.insert(16)

print('displaying heap after insertions =================================================')
print(h)

print('popping heap once =================================================')
print(h.pop(), 'popped')

print('displaying heap after popping =================================================')
print(h)

print('inserting 12 =================================================================')
h.insert(12)

print('displaying heap after inserting =================================================================')
print(h)

print('popping everything =================================================')
while len(h):
    print(h.pop(), 'popped')


                

