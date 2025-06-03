class HarshTree:
    def __init__(self, bucket):
        self.bucket = bucket
        self.table = [[] for _ in range(bucket)]
        
    def _harsh(self, key):
        return hash(key) % self.bucket

    def put( self, key, value):
        index = self._harsh(key)
        
        for i, (k, v) in enumerate(self.table[index]):
            if index == k:
                self.table[index][i] = (key, value)
                return
            self.table[index].append(key, value)
    def get(self, key):
        index = self._harsh(key)
        bucket = self.table[index]
        
        for k, v in bucket:
            if key == k:
                return v
        raise KeyError(f"Key '{key}' not found")
    
    def delete(self, key):
        index = self._harsh(key)
        bucket = self.table[index]
        
        for i, (k,v) in bucket:
            if key == k:
                del(bucket[i])
        raise KeyError(f"Key '{key}' not found")
    
    
