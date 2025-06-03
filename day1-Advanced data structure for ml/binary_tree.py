class BinaryTree:
    def __init__(self, val=0, left=0, right=0):
        self.val = val
        self.left = left
        self.right = right
        
    def insert(self, val):
        if self.val is None:
            self.val = val
            return
        
        if val < self.val:
            if self.left is None:
                self.left = BinaryTree(val)
            else:
                self.left.insert(val)
        else:
            if self.right is None:
                self.right = BinaryTree(val)
            else:
                self.right.insert(val)
    def in_order_traversal(self):
        elements = []
        if self.left:
            elements += self.left.in_order_traversal()
        elements.append(self.val)
        if self.right:
            elements += self.right.in_order_traversal()
        return elements