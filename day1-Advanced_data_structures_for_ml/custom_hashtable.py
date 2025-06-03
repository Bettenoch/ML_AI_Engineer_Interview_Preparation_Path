import hashlib
import math
from typing import List, Tuple, Any, Dict
import numpy as np
from collections import defaultdict

class CustomHashTable:
    """
    Custom Hash Table implementation for efficient feature storage and retrieval.
    Supports multiple collision resolution strategies and dynamic resizing.
    """
    
    def __init__(self, initial_size: int = 16, collision_strategy: str = 'chaining', 
                 load_factor_threshold: float = 0.75):
        """
        Initialize the hash table.
        
        Args:
            initial_size: Initial size of the hash table (should be power of 2)
            collision_strategy: 'chaining' or 'open_addressing'
            load_factor_threshold: When to resize the table
        """
        self.size = self._next_power_of_2(initial_size)
        self.collision_strategy = collision_strategy
        self.load_factor_threshold = load_factor_threshold
        self.count = 0
        
        # Initialize based on collision strategy
        if collision_strategy == 'chaining':
            self.table = [[] for _ in range(self.size)]
        else:  # open_addressing
            self.table = [None for _ in range(self.size)]
            self.deleted = [False for _ in range(self.size)]
        
        # Statistics tracking
        self.collisions = 0
        self.lookups = 0
        self.hash_calls = 0
    
    def _next_power_of_2(self, n: int) -> int:
        """Find the next power of 2 greater than or equal to n."""
        return 2 ** math.ceil(math.log2(max(n, 1)))
    
    def _hash_function(self, key: Any) -> int:
        """
        Primary hash function using built-in hash() with bit manipulation.
        
        Args:
            key: The key to hash
            
        Returns:
            Hash value as integer
        """
        self.hash_calls += 1
        # Use built-in hash and apply bit masking for better distribution
        hash_value = hash(key)
        # Apply bit manipulation to improve distribution
        hash_value ^= (hash_value >> 16)
        return hash_value & (self.size - 1)  # Equivalent to % size for powers of 2
    
    def _secondary_hash(self, key: Any) -> int:
        """
        Secondary hash function for double hashing in open addressing.
        
        Args:
            key: The key to hash
            
        Returns:
            Secondary hash value (must be odd to ensure we visit all slots)
        """
        # Use a different hash algorithm for secondary hashing
        hash_str = str(key).encode('utf-8')
        hash_value = int(hashlib.md5(hash_str).hexdigest()[:8], 16)
        # Ensure the result is odd and non-zero
        return (hash_value % (self.size - 1)) | 1
    
    def _should_resize(self) -> bool:
        """Check if the hash table should be resized based on load factor."""
        load_factor = self.count / self.size
        return load_factor > self.load_factor_threshold
    
    def _resize(self):
        """
        Resize the hash table when load factor exceeds threshold.
        This is crucial for maintaining O(1) average performance.
        """
        print(f"Resizing hash table from {self.size} to {self.size * 2}")
        
        # Store old table
        old_table = self.table
        old_size = self.size
        old_deleted = getattr(self, 'deleted', None)
        
        # Create new larger table
        self.size *= 2
        self.count = 0
        
        if self.collision_strategy == 'chaining':
            self.table = [[] for _ in range(self.size)]
            
            # Rehash all existing entries
            for bucket in old_table:
                for key, value in bucket:
                    self.put(key, value)
        else:  # open_addressing
            self.table = [None for _ in range(self.size)]
            self.deleted = [False for _ in range(self.size)]
            
            # Rehash all existing entries
            for i in range(old_size):
                if old_table[i] is not None and not old_deleted[i]:
                    key, value = old_table[i]
                    self.put(key, value)
    
    def put(self, key: Any, value: Any) -> bool:
        """
        Insert or update a key-value pair in the hash table.
        
        Args:
            key: The key to insert/update
            value: The value to associate with the key
            
        Returns:
            True if insertion was successful
        """
        # Check if resize is needed
        if self._should_resize():
            self._resize()
        
        index = self._hash_function(key)
        
        if self.collision_strategy == 'chaining':
            return self._put_chaining(key, value, index)
        else:
            return self._put_open_addressing(key, value, index)
    
    def _put_chaining(self, key: Any, value: Any, index: int) -> bool:
        """Insert using chaining collision resolution."""
        bucket = self.table[index]
        
        # Check if key already exists
        for i, (existing_key, existing_value) in enumerate(bucket):
            if existing_key == key:
                bucket[i] = (key, value)  # Update existing
                return True
        
        # Key doesn't exist, add new entry
        if len(bucket) > 0:
            self.collisions += 1
        
        bucket.append((key, value))
        self.count += 1
        return True
    
    def _put_open_addressing(self, key: Any, value: Any, index: int) -> bool:
        """Insert using open addressing with double hashing."""
        original_index = index
        step = self._secondary_hash(key)
        attempts = 0
        
        while attempts < self.size:
            if self.table[index] is None or self.deleted[index]:
                # Found empty slot or deleted slot
                if self.table[index] is not None:  # It was deleted
                    self.collisions += 1
                
                self.table[index] = (key, value)
                self.deleted[index] = False
                self.count += 1
                return True
            
            # Check if key already exists
            existing_key, existing_value = self.table[index]
            if existing_key == key:
                self.table[index] = (key, value)  # Update
                return True
            
            # Collision occurred, try next position
            if attempts == 0:  # First collision
                self.collisions += 1
            
            index = (index + step) % self.size
            attempts += 1
        
        # Table is full (shouldn't happen with proper load factor management)
        return False
    
    def get(self, key: Any) -> Any:
        """
        Retrieve a value by key.
        
        Args:
            key: The key to look up
            
        Returns:
            The value associated with the key
            
        Raises:
            KeyError: If key is not found
        """
        self.lookups += 1
        index = self._hash_function(key)
        
        if self.collision_strategy == 'chaining':
            return self._get_chaining(key, index)
        else:
            return self._get_open_addressing(key, index)
    
    def _get_chaining(self, key: Any, index: int) -> Any:
        """Retrieve using chaining collision resolution."""
        bucket = self.table[index]
        
        for existing_key, value in bucket:
            if existing_key == key:
                return value
        
        raise KeyError(f"Key '{key}' not found")
    
    def _get_open_addressing(self, key: Any, index: int) -> Any:
        """Retrieve using open addressing with double hashing."""
        original_index = index
        step = self._secondary_hash(key)
        attempts = 0
        
        while attempts < self.size:
            if self.table[index] is None:
                # Empty slot means key was never inserted
                if not self.deleted[index]:
                    break
            elif not self.deleted[index]:
                # Non-deleted slot, check the key
                existing_key, value = self.table[index]
                if existing_key == key:
                    return value
            
            index = (index + step) % self.size
            attempts += 1
        
        raise KeyError(f"Key '{key}' not found")
    
    def delete(self, key: Any) -> bool:
        """
        Delete a key-value pair from the hash table.
        
        Args:
            key: The key to delete
            
        Returns:
            True if deletion was successful, False if key not found
        """
        index = self._hash_function(key)
        
        if self.collision_strategy == 'chaining':
            return self._delete_chaining(key, index)
        else:
            return self._delete_open_addressing(key, index)
    
    def _delete_chaining(self, key: Any, index: int) -> bool:
        """Delete using chaining collision resolution."""
        bucket = self.table[index]
        
        for i, (existing_key, value) in enumerate(bucket):
            if existing_key == key:
                del bucket[i]
                self.count -= 1
                return True
        
        return False
    
    def _delete_open_addressing(self, key: Any, index: int) -> bool:
        """Delete using open addressing (lazy deletion)."""
        original_index = index
        step = self._secondary_hash(key)
        attempts = 0
        
        while attempts < self.size:
            if self.table[index] is None:
                if not self.deleted[index]:
                    break
            elif not self.deleted[index]:
                existing_key, value = self.table[index]
                if existing_key == key:
                    self.deleted[index] = True
                    self.count -= 1
                    return True
            
            index = (index + step) % self.size
            attempts += 1
        
        return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get performance statistics of the hash table.
        
        Returns:
            Dictionary containing various performance metrics
        """
        load_factor = self.count / self.size
        
        if self.collision_strategy == 'chaining':
            # Calculate average chain length
            chain_lengths = [len(bucket) for bucket in self.table]
            avg_chain_length = sum(chain_lengths) / len(chain_lengths)
            max_chain_length = max(chain_lengths)
            empty_buckets = sum(1 for length in chain_lengths if length == 0)
        else:
            # Calculate clustering for open addressing
            empty_slots = sum(1 for slot in self.table if slot is None)
            deleted_slots = sum(1 for deleted in self.deleted if deleted)
            avg_chain_length = "N/A (Open Addressing)"
            max_chain_length = "N/A (Open Addressing)"
            empty_buckets = empty_slots
        
        return {
            'size': self.size,
            'count': self.count,
            'load_factor': load_factor,
            'collision_strategy': self.collision_strategy,
            'total_collisions': self.collisions,
            'total_lookups': self.lookups,
            'total_hash_calls': self.hash_calls,
            'avg_chain_length': avg_chain_length,
            'max_chain_length': max_chain_length,
            'empty_buckets': empty_buckets,
            'collision_rate': self.collisions / max(self.hash_calls, 1)
        }
    
    def __len__(self) -> int:
        """Return the number of key-value pairs in the hash table."""
        return self.count
    
    def __contains__(self, key: Any) -> bool:
        """Check if a key exists in the hash table."""
        try:
            self.get(key)
            return True
        except KeyError:
            return False
    
    def keys(self) -> List[Any]:
        """Return a list of all keys in the hash table."""
        keys = []
        
        if self.collision_strategy == 'chaining':
            for bucket in self.table:
                for key, value in bucket:
                    keys.append(key)
        else:
            for i, slot in enumerate(self.table):
                if slot is not None and not self.deleted[i]:
                    key, value = slot
                    keys.append(key)
        
        return keys
    
    def values(self) -> List[Any]:
        """Return a list of all values in the hash table."""
        values = []
        
        if self.collision_strategy == 'chaining':
            for bucket in self.table:
                for key, value in bucket:
                    values.append(value)
        else:
            for i, slot in enumerate(self.table):
                if slot is not None and not self.deleted[i]:
                    key, value = slot
                    values.append(value)
        
        return values
    
    def items(self) -> List[Tuple[Any, Any]]:
        """Return a list of all key-value pairs in the hash table."""
        items = []
        
        if self.collision_strategy == 'chaining':
            for bucket in self.table:
                for key, value in bucket:
                    items.append((key, value))
        else:
            for i, slot in enumerate(self.table):
                if slot is not None and not self.deleted[i]:
                    items.append(slot)
        
        return items


# Feature Storage Extension
class FeatureHashTable(CustomHashTable):
    """
    Specialized hash table for machine learning feature storage.
    Optimized for handling high-dimensional sparse features.
    """
    
    def __init__(self, initial_size: int = 64, collision_strategy: str = 'chaining'):
        super().__init__(initial_size, collision_strategy)
        self.feature_stats = defaultdict(lambda: {'count': 0, 'sum': 0.0, 'sum_squares': 0.0})
    
    def add_feature(self, feature_name: str, value: float, sample_id: str = None):
        """
        Add a feature value and update statistics.
        
        Args:
            feature_name: Name of the feature
            value: Feature value
            sample_id: Optional sample identifier
        """
        # Store the feature value
        key = f"{feature_name}_{sample_id}" if sample_id else feature_name
        self.put(key, value)
        
        # Update feature statistics
        stats = self.feature_stats[feature_name]
        stats['count'] += 1
        stats['sum'] += value
        stats['sum_squares'] += value * value
    
    def get_feature_stats(self, feature_name: str) -> Dict[str, float]:
        """Get statistical summary for a feature."""
        stats = self.feature_stats[feature_name]
        if stats['count'] == 0:
            return {'mean': 0.0, 'variance': 0.0, 'std': 0.0}
        
        mean = stats['sum'] / stats['count']
        variance = (stats['sum_squares'] / stats['count']) - (mean * mean)
        
        return {
            'count': stats['count'],
            'mean': mean,
            'variance': max(0, variance),  # Ensure non-negative due to floating point errors
            'std': math.sqrt(max(0, variance))
        }
    
    def get_features_by_sample(self, sample_id: str) -> Dict[str, float]:
        """Get all features for a specific sample."""
        features = {}
        suffix = f"_{sample_id}"
        
        for key in self.keys():
            if key.endswith(suffix):
                feature_name = key[:-len(suffix)]
                features[feature_name] = self.get(key)
        
        return features


# Demonstration and Testing
def demonstrate_hash_table():
    """Comprehensive demonstration of the hash table functionality."""
    
    print("=" * 60)
    print("CUSTOM HASH TABLE DEMONSTRATION")
    print("=" * 60)
    
    # Test both collision strategies
    for strategy in ['chaining', 'open_addressing']:
        print(f"\n--- Testing {strategy.upper()} Strategy ---")
        
        # Create hash table
        ht = CustomHashTable(initial_size=8, collision_strategy=strategy)
        
        # Insert test data
        test_data = [
            ("apple", 1.5),
            ("banana", 2.3),
            ("cherry", 0.8),
            ("date", 3.2),
            ("elderberry", 1.1),
            ("fig", 2.7),
            ("grape", 4.1),
            ("honeydew", 1.9)
        ]
        
        print(f"Inserting {len(test_data)} items...")
        for key, value in test_data:
            ht.put(key, value)
            print(f"  {key}: {value}")
        
        # Test retrieval
        print("\nTesting retrieval:")
        for key, expected_value in test_data[:3]:
            retrieved_value = ht.get(key)
            print(f"  {key}: {retrieved_value} (expected: {expected_value})")
        
        # Test updates
        print("\nTesting updates:")
        ht.put("apple", 999.9)
        print(f"  Updated apple: {ht.get('apple')}")
        
        # Test deletion
        print("\nTesting deletion:")
        deleted = ht.delete("banana")
        print(f"  Deleted banana: {deleted}")
        try:
            ht.get("banana")
        except KeyError:
            print("  Banana successfully deleted (KeyError raised)")
        
        # Show statistics
        stats = ht.get_statistics()
        print(f"\nHash Table Statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
    
    # Demonstrate Feature Hash Table
    print("\n" + "=" * 60)
    print("FEATURE HASH TABLE DEMONSTRATION")
    print("=" * 60)
    
    fht = FeatureHashTable()
    
    # Add sample features
    samples = ['sample_1', 'sample_2', 'sample_3']
    features = ['height', 'weight', 'age', 'income']
    
    # Generate some sample data
    np.random.seed(42)
    for sample in samples:
        for feature in features:
            value = np.random.normal(100, 20)  # Random values
            fht.add_feature(feature, value, sample)
    
    print("Added features for 3 samples with 4 features each")
    
    # Show feature statistics
    print("\nFeature Statistics:")
    for feature in features:
        stats = fht.get_feature_stats(feature)
        print(f"  {feature}: mean={stats['mean']:.2f}, std={stats['std']:.2f}")
    
    # Show features for a specific sample
    print(f"\nFeatures for sample_1:")
    sample_features = fht.get_features_by_sample('sample_1')
    for feature, value in sample_features.items():
        print(f"  {feature}: {value:.2f}")
    
    print(f"\nTotal items in feature hash table: {len(fht)}")

if __name__ == "__main__":
    demonstrate_hash_table()