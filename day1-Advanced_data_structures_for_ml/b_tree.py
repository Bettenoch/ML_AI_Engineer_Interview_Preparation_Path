class BTreeNode:
    def __init__(self, t, leaf=False):
        self.t = t #defines the minimum degree -number of keys in a node
        self.keys = [] 
        self.children = [] #list of child nodes
        self.leaf = leaf #if true, node is a leaf node
        
    def __str__(self):
        return f"Keys: {self.keys}, Leaf: {self.leaf}"
    

class BTree:
    def __init__(self, t=2):
        self.root = BTreeNode(t, leaf=True)
        self.t = t #minimum degree of the B-tree
        
    def insert(self, k):
        root = self.root
        if len(root.keys) == 2 * self.t - 1:
            #split the root if it is full
            new_root = BTreeNode(self.t, False)
            new_root.children.append(self.root)
            self._split_child(new_root,0)
            self.root = new_root
            
        self._insert_non_full(self.root, k)
        
    def _insert_non_full(self, node, k):
        i = len(node.keys) - 1

        if node.leaf:
            # Insert in sorted order
            node.keys.append(None)
            while i >= 0 and k < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                i -= 1
            node.keys[i + 1] = k
        else:
            # Find child to insert into
            while i >= 0 and k < node.keys[i]:
                i -= 1
            i += 1

            # If the child is full, split it
            if len(node.children[i].keys) == (2 * self.t - 1):
                self._split_child(node, i)
                # Decide which of the two children to descend into
                if k > node.keys[i]:
                    i += 1

            self._insert_non_full(node.children[i], k)
            
    def _split_child(self, parent, i):
        t = self.t
        full_child = parent.children[i]
        new_child = BTreeNode(t, leaf=full_child.leaf)

        # New child gets the last t-1 keys
        new_child.keys = full_child.keys[t:]
        full_child.keys = full_child.keys[:t - 1]

        # If not a leaf, split the children as well
        if not full_child.leaf:
            new_child.children = full_child.children[t:]
            full_child.children = full_child.children[:t]

        # Insert middle key into parent
        parent.keys.insert(i, full_child.keys.pop())
        parent.children.insert(i + 1, new_child)
    
    def search(self, k, node=None):
        if node is None:
            node = self.root

        # Find the first key >= k
        i = 0
        while i < len(node.keys) and k > node.keys[i]:
            i += 1

        # If the key is found
        if i < len(node.keys) and node.keys[i] == k:
            return True

        # If it's a leaf, and not found
        if node.leaf:
            return False

        # Recurse into the child
        return self.search(k, node.children[i])
    
    def print_tree(self, node=None, level=0):
        if node is None:
            node = self.root
        print("  " * level + str(node))
        for child in node.children:
            self.print_tree(child, level + 1)


if __name__ == "__main__":
    btree = BTree(t=2)

    values = [10, 20, 5, 6, 12, 30, 7, 17]
    for v in values:
        btree.insert(v)

    print("\nTree structure:")
    btree.print_tree()

    print("\nSearch results:")
    for test in [6, 15, 17, 25]:
        print(f"Search {test}: {btree.search(test)}")
