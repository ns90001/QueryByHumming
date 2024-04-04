import numpy as np

class MLSH:
    def __init__(self, d, b, r, data):
        self.d = d
        self.b = b
        self.r = r
        self.data = data
        self.hash_functions = self.generate_hash_functions()
        self.buckets = {}
        self.load_data()

    def generate_hash_functions(self):
        h = [None] * (self.b * self.r)
        for i in range(len(h)):
            a = np.random.normal(size=self.d)
            bias = np.random.uniform(low=0, high=self.b)
            h[i] = lambda x: np.floor((np.dot(a, x[0]) + bias) / self.b)
        return h
    
    def load_data(self):
        for i in range(len(self.data)):
            id = tuple(h(self.data[i]) for h in self.hash_functions)
            if id not in self.buckets:
                self.buckets[id] = [i]
            else:
                self.buckets[id].append(i)
    
    def get_candidates(self, query_vector):
        id = tuple(h(query_vector) for h in self.hash_functions)
        if id not in self.buckets:
            return []
        indices = self.buckets[id]
        candidates = []
        for idx in indices:
            candidates.append(self.data[idx])
        return np.array(candidates)
    
    def query(self, query_vector, k):
        candidates = self.get_candidates(query_vector)
        spread = max(query_vector[0]) - min(query_vector[0])
        distances = []
        for c in candidates:
            d = np.linalg.norm(c[0] - query_vector[0]) / spread
            distances.append((c, d))

        distances.sort(key=lambda x: x[1])

        k_closest = []
        for i in range(min(k, len(distances))):
            c = distances[i][0]
            d = distances[i][1]
            k_closest.append([c[0], c[1], c[2], d])
        
        return k_closest
    
def test():

    data = [[1, 1], [2, 2], [10, 5], [-10, -20]]
    print("initializing lsh...")
    mlsh = MLSH(2, 4, 4, data)

    query = [-10, -19]
    print("running query...")
    closest = mlsh.query(query, 2)

    print("Closest via E2LSH:" + str(closest))

# uncomment to run tests
# test()