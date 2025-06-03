class SimpleHarshTable:
    def __init__(self, bucket):
        self.bucket = bucket
        self.table = [[] for _ in range(bucket)]
    def _harsh(self, key):
        # Use bitwise AND if bucket is a power of 2
        if (self.bucket & (self.bucket - 1)) == 0:
            return hash(key) & (self.bucket - 1)
        return hash(key) % self.bucket

    
    def put(self, key, value):
        index = self._harsh(key)
        
        for i, (k, v) in enumerate(self.table[index]):
            if k == key:
                self.table[index][i] = (key, value)
                return
        self.table[index].append((key, value))
        
    def get(self, key):
        index = self._harsh(key)
        for (k, v) in self.table(index):
            if k == key:
                return v
        raise KeyError(f"Key '{key}' not found")
    
    def delete(self, key):
        
        index = self._harsh(key)
        bucket = self.table(index)
        for i, (k, v) in bucket:
            if k == key:
                del bucket[i]
        raise KeyError(f"Key '{key}' not found")       
    #enumerate is a built in python function used when you want to loop through a list of iterables and keep track of the index of the current item
    # chaining : storong multiple key_vale pairs in the same bucket
    
    
    