class AdjacencyMatrix:
    def __init__(self, vertices):
        self.ve = vertices
        self.graph = [[0] * vertices for _ in range(vertices)]
        
    def addVertex(self, i, j, weight=1):
        self.graph[i][j] = weight
        self.graph[j][i] = weight
        
    def displayGraph(self):
        for row in self.graph:
            print(' '.join(map(str, row)))

# Test the implementation with detailed examples
print("=" * 50)
print("TESTING ADJACENCY MATRIX IMPLEMENTATION")
print("=" * 50)

# Example 1: Simple unweighted graph
print("\n1. SIMPLE UNWEIGHTED GRAPH (5 vertices)")
print("-" * 40)

# Create a graph with 5 vertices (0, 1, 2, 3, 4)
graph1 = AdjacencyMatrix(5)

print("Initial empty matrix:")
graph1.displayGraph()

# Add edges step by step
print("\nAdding edges:")
edges_to_add = [(0, 1), (0, 2), (1, 3), (2, 3), (3, 4)]

for i, (u, v) in enumerate(edges_to_add):
    print(f"\nStep {i+1}: Adding edge {u} -- {v}")
    graph1.addVertex(u, v, 1)  # Using weight=1 for unweighted edges
    print("Current matrix:")
    graph1.displayGraph()

print("\nFinal graph representation:")
print("Vertices: 0, 1, 2, 3, 4")
print("Edges: 0-1, 0-2, 1-3, 2-3, 3-4")
graph1.displayGraph()

# Example 2: Weighted graph
print("\n\n2. WEIGHTED GRAPH (4 vertices)")
print("-" * 40)

graph2 = AdjacencyMatrix(4)
print("Creating weighted graph with vertices: A(0), B(1), C(2), D(3)")

# Add weighted edges
weighted_edges = [
    (0, 1, 5),   # A to B with weight 5
    (0, 2, 3),   # A to C with weight 3
    (1, 2, 2),   # B to C with weight 2
    (1, 3, 6),   # B to D with weight 6
    (2, 3, 4)    # C to D with weight 4
]

print("\nAdding weighted edges:")
for u, v, weight in weighted_edges:
    vertex_names = ['A', 'B', 'C', 'D']
    print(f"Adding edge {vertex_names[u]} -- {vertex_names[v]} with weight {weight}")
    graph2.addVertex(u, v, weight)

print("\nFinal weighted graph matrix:")
print("     A  B  C  D")
for i, row in enumerate(graph2.graph):
    vertex_names = ['A', 'B', 'C', 'D']
    print(f"{vertex_names[i]}:   {' '.join(map(str, row))}")

# Example 3: Testing edge cases and demonstrating properties
print("\n\n3. GRAPH PROPERTIES DEMONSTRATION")
print("-" * 40)

graph3 = AdjacencyMatrix(3)

# Add some edges
graph3.addVertex(0, 1, 7)
graph3.addVertex(1, 2, 3)

print("Graph with edges 0-1 (weight 7) and 1-2 (weight 3):")
graph3.displayGraph()

print("\nDemonstrating symmetry (undirected graph):")
print(f"graph[0][1] = {graph3.graph[0][1]}")
print(f"graph[1][0] = {graph3.graph[1][0]}")
print("Both values are equal - confirms undirected graph representation")

print("\nChecking connectivity:")
print("Vertex 0 is connected to:", end=" ")
for j in range(3):
    if graph3.graph[0][j] != 0:
        print(f"{j}(weight: {graph3.graph[0][j]})", end=" ")
print()

print("Vertex 1 is connected to:", end=" ")
for j in range(3):
    if graph3.graph[1][j] != 0:
        print(f"{j}(weight: {graph3.graph[1][j]})", end=" ")
print()

print("Vertex 2 is connected to:", end=" ")
for j in range(3):
    if graph3.graph[2][j] != 0:
        print(f"{j}(weight: {graph3.graph[2][j]})", end=" ")
print()

# Example 4: Real-world scenario - City connections
print("\n\n4. REAL-WORLD EXAMPLE: CITY CONNECTIONS")
print("-" * 40)

print("Representing distances between 4 cities:")
print("0: New York, 1: Boston, 2: Philadelphia, 3: Washington DC")

city_graph = AdjacencyMatrix(4)

# Add distances (in hundreds of miles)
city_connections = [
    (0, 1, 2),   # NY to Boston: 200 miles
    (0, 2, 1),   # NY to Philadelphia: 100 miles  
    (0, 3, 2),   # NY to Washington: 200 miles
    (1, 2, 3),   # Boston to Philadelphia: 300 miles
    (2, 3, 1)    # Philadelphia to Washington: 100 miles
]

cities = ["New York", "Boston", "Philadelphia", "Washington DC"]

print("\nAdding city connections:")
for i, j, distance in city_connections:
    print(f"{cities[i]} <-> {cities[j]}: {distance}00 miles")
    city_graph.addVertex(i, j, distance)

print("\nDistance matrix (in hundreds of miles):")
print("           NY  Bos Phil Wash")
for i in range(4):
    city_abbrev = ["NY  ", "Bos ", "Phil", "Wash"]
    print(f"{city_abbrev[i]}:       {' '.join(f'{x:2d}' for x in city_graph.graph[i])}")

# Example 5: Testing with self-loops and multiple edges
print("\n\n5. EDGE CASES AND LIMITATIONS")
print("-" * 40)

edge_case_graph = AdjacencyMatrix(3)

print("Testing self-loop (vertex connected to itself):")
edge_case_graph.addVertex(0, 0, 5)  # Self-loop
print("After adding self-loop at vertex 0:")
edge_case_graph.displayGraph()

print("\nTesting overwriting edges:")
edge_case_graph.addVertex(0, 1, 3)
print("Added edge 0-1 with weight 3:")
edge_case_graph.displayGraph()

edge_case_graph.addVertex(0, 1, 8)  # Overwrite previous edge
print("Overwrote edge 0-1 with weight 8:")
edge_case_graph.displayGraph()
print("Note: Previous weight is lost - adjacency matrix can't store multiple edges")

# Analysis and recommendations
print("\n\n6. ANALYSIS OF YOUR IMPLEMENTATION")
print("-" * 40)

print("✅ STRENGTHS:")
print("- Simple and clean implementation")
print("- Correctly handles undirected graphs (symmetric matrix)")
print("- O(1) edge lookup and insertion")
print("- Good for dense graphs")

print("\n⚠️  OBSERVATIONS:")
print("- Method name 'addVertex' is misleading - it actually adds edges")
print("- Default weight=0 might be confusing (usually 1 for unweighted)")
print("- No error checking for invalid vertex indices")
print("- No way to remove edges")

print("\n🔧 SUGGESTED IMPROVEMENTS:")

class ImprovedAdjacencyMatrix:
    def __init__(self, vertices):
        self.vertices = vertices
        self.graph = [[0] * vertices for _ in range(vertices)]
        
    def add_edge(self, u, v, weight=1):  # Better method name
        if 0 <= u < self.vertices and 0 <= v < self.vertices:
            self.graph[u][v] = weight
            self.graph[v][u] = weight
        else:
            raise ValueError(f"Vertex indices must be between 0 and {self.vertices-1}")
    
    def remove_edge(self, u, v):
        if 0 <= u < self.vertices and 0 <= v < self.vertices:
            self.graph[u][v] = 0
            self.graph[v][u] = 0
    
    def has_edge(self, u, v):
        if 0 <= u < self.vertices and 0 <= v < self.vertices:
            return self.graph[u][v] != 0
        return False
    
    def get_weight(self, u, v):
        if 0 <= u < self.vertices and 0 <= v < self.vertices:
            return self.graph[u][v]
        return None
    
    def display_graph(self):
        print("Adjacency Matrix:")
        for row in self.graph:
            print(' '.join(f'{x:2d}' for x in row))

print("\nTesting improved version:")
improved = ImprovedAdjacencyMatrix(3)
improved.add_edge(0, 1, 5)
improved.add_edge(1, 2, 3)

print(f"Has edge 0-1: {improved.has_edge(0, 1)}")
print(f"Weight of edge 0-1: {improved.get_weight(0, 1)}")
print(f"Has edge 0-2: {improved.has_edge(0, 2)}")

improved.display_graph()

print("\n" + "=" * 50)
print("TEST COMPLETED SUCCESSFULLY!")
print("=" * 50)