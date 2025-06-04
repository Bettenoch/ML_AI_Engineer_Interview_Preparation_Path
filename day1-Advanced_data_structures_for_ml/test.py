from collections import defaultdict
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
    
dy = defaultdict(list)

pairs = [("a", 1), ("b", 1), ("c", 1), ("a", 2), ("c", 4),]

for k, v in pairs:
    dy[k].append(v)
    
# print (dy)
# Python example using feature hashing on text data

# Python example of a simple feature hashing function
def feature_hashing(feature, vector_size):
    index = hash(feature) % vector_size
    vector = [0] * vector_size
    vector[index] = 1
    return vector

# Test the function
print(feature_hashing("apple", 5))

sentence = "The quick brown fox"
vector_size = 10
hashed_sentence = [feature_hashing(word, vector_size) for word in sentence.split()]
print(hashed_sentence)