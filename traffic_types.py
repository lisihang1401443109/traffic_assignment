class Node(int):

    def __new__(cls, value, xfc=False):
        obj = super().__new__(cls, value)
        obj.xfc = xfc
        return obj

    def set_XFC(self):
        self.xfc = True
    
    def unset_XFC(self):
        self.xfc = False

    def is_XFC(self):
        return self.xfc
    
    def __str__(self):
        return f'Node({super().__str__()}, {self.xfc})'