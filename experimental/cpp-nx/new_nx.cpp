#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <cmath>
#include <random>
#include <queue>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>

class Graph {
 public:
    // Default constructor
    Graph() {}

    // Add a node to the graph
    void add_node(int node) {
        nodes_.insert(node);
    }

    // Add an edge to the graph
    void add_edge(int u, int v) {
        adjacency_list_[u].insert(v);
        adjacency_list_[v].insert(u);
    }

    // Remove a node from the graph
    void remove_node(int node) {
        nodes_.erase(node);
        adjacency_list_.erase(node);
        for (auto& pair : adjacency_list_) {
            pair.second.erase(node);
        }
    }

    // Remove an edge from the graph
    void remove_edge(int u, int v) {
        adjacency_list_[u].erase(v);
        adjacency_list_[v].erase(u);
    }

    // Return the number of nodes in the graph. Use unsigned int to avoid compiler warning
    unsigned int num_nodes() const {
        return nodes_.size();
    }

    // Return the number of edges in the graph
    unsigned int num_edges() const {
        int num_edges = 0;
        for (const auto& pair : adjacency_list_) {
            num_edges += pair.second.size();
        }
        return num_edges / 2;
    }

    // Return the degree of a node
    int degree(int node) const {
        return adjacency_list_.at(node).size();
    }

    // Return the neighbors of a node
    std::vector<int> neighbors(int node) const {
        return std::vector<int>(adjacency_list_.at(node).begin(),
                                adjacency_list_.at(node).end());
    }

    // Check if a node is in the graph
    bool has_node(int node) const {
        return nodes_.count(node) > 0;
    }

    // Check if an edge exists in the graph
    bool has_edge(int u, int v) const {
        if (!has_node(u) || !has_node(v)) {
            return false;
        }
        return adjacency_list_.at(u).count(v) > 0;
    }

    // Check if the graph is connected
    bool is_connected() const {
        if (num_nodes() == 0) {
            return true;
        }
        std::unordered_set<int> visited;
        std::queue<int> queue;
        queue.push(*nodes_.begin());
        while (!queue.empty()) {
            int node = queue.front();
            queue.pop();
            if (visited.count(node) == 0) {
                visited.insert(node);
                for (int neighbor : neighbors(node)) {
                    queue.push(neighbor);
                }
            }
        }
        // convert num_nodes() to unsigned int to avoid compiler warning

          return visited.size() == num_nodes();
    }

    // Return the set of nodes in the graph
    const std::unordered_set<int>& nodes() const {
        return nodes_;
    }

    // Return the set of edges in the graph
    std::vector<std::pair<int, int>> edges() const {
        std::vector<std::pair<int, int>> edges;
        for (const auto& pair : adjacency_list_) {
            int u = pair.first;
            for (int v : pair.second) {
                if (u < v) {
                    edges.push_back({u, v});
                }
            }
        }
        return edges;
    }

    // Return the adjacency list representation of the graph
    const std::unordered_map<int, std::unordered_set<int>>& adjacency_list() const {
        return adjacency_list_;
    }

    // Info function that prints number of nodes and edges
    void info() const {
        std::cout << "Graph with " << num_nodes() << " nodes and " << num_edges() << " edges" << std::endl;
    }

    // Density function that returns the density of the graph
    double density() const {
        return 2.0 * num_edges() / (num_nodes() * (num_nodes() - 1));
    }

 private:
    // Set of nodes in the graph
    std::unordered_set<int> nodes_;
    // Adjacency list representation of the graph
    std::unordered_map<int, std::unordered_set<int>> adjacency_list_;
};


// Read graph from edge list file
Graph read_graph(const std::string& filename) {
    Graph G;
    std::ifstream file(filename);
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        int u, v;
        ss >> u >> v;
        G.add_node(u);
        G.add_node(v);
        G.add_edge(u, v);
    }
    G.info();
    return G;
}


// Return a bool if there is a path between two nodes
bool has_path(const Graph& G, int u, int v) {
    if (!G.has_node(u) || !G.has_node(v)) {
        return false;
    }
    std::unordered_set<int> visited;
    std::queue<int> queue;
    queue.push(u);
    while (!queue.empty()) {
        int node = queue.front();
        queue.pop();
        if (visited.count(node) == 0) {
            visited.insert(node);
            for (int neighbor : G.neighbors(node)) {
                if (neighbor == v) {
                    return true;
                }
                queue.push(neighbor);
            }
        }
    }
    return false;
}


// Check if the graph is connected. If not, return a list of connected components
std::vector<std::vector<int>> connected_components(const Graph& G) {
  std::vector<std::vector<int>> components;
  std::unordered_set<int> visited;
  for (int u : G.nodes()) {
    if (visited.count(u) == 0) {
      std::vector<int> component;
      std::queue<int> q;
      q.push(u);
      visited.insert(u);
      while (!q.empty()) {
        int v = q.front();
        q.pop();
        component.push_back(v);
        for (int w : G.neighbors(v)) {
          if (visited.count(w) == 0) {
            visited.insert(w);
            q.push(w);
          }
        }
      }
      components.push_back(component);
    }
  }
  return components;
}


// Return the largest connected component of a graph, use the connected_components function
Graph largest_component(const Graph& G) {
  std::vector<std::vector<int>> components = connected_components(G);
  std::vector<int> largest_component;
  unsigned int largest_size = 0;
  for (const std::vector<int>& component : components) {
    if (component.size() > largest_size) {
      largest_size = component.size();
      largest_component = component;
    }
  }
  Graph H;
  for (int u : largest_component) {
    for (int v : G.neighbors(u)) {
      if (std::find(largest_component.begin(), largest_component.end(), v) != largest_component.end()) {
        H.add_edge(u, v);
      }
    }
  }
  return H;
}


// Randomly rewire an edge with probability `p`
void rewire(Graph& G, int u, int v, double p) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<> dis(0, 1);

  if (dis(gen) < p) {
    // Choose a new node randomly
    std::uniform_int_distribution<> node_dis(0, G.nodes().size() - 1);
    int w = *std::next(G.nodes().begin(), node_dis(gen));

    // Remove the edge (u, v) and add the edge (u, w)
    G.add_edge(u, w);
    G.add_edge(v, w);
  }
}


// Compute a random graph by swapping edges of a given graph.
Graph random_reference(const Graph& G, int k) {
  Graph H;

  // Add the nodes to the new graph
  for (int u : G.nodes()) {
    H.add_node(u);
  }

  // Choose the probability of rewiring an edge
  double p = k / (G.nodes().size() - 1.0);

  // Add edges to the new graph
  for (int u : G.nodes()) {
    for (int v : G.neighbors(u)) {
      if (u < v) {
        H.add_edge(u, v);
        rewire(H, u, v, p);
      }
    }
  }

  return H;
}


// Latticize the given graph by swapping edges.
Graph lattice_reference(const Graph& G, int n) {
  Graph H;

  // Add the nodes to the new graph
  for (int u : G.nodes()) {
    H.add_node(u);
  }

  // Add edges to the new graph
  for (int u : G.nodes()) {
    for (int v : G.neighbors(u)) {
      if (u < v && has_path(G, u, v)) {
        // Check if u and v are neighbors in the lattice
        int u_x = u / n;
        int u_y = u % n;
        int v_x = v / n;
        int v_y = v % n;
        if ((u_x == v_x && std::abs(u_y - v_y) == 1) ||
            (u_y == v_y && std::abs(u_x - v_x) == 1)) {
          H.add_edge(u, v);
        }
      }
    }
  }

  return H;
}


std::vector<int> cumulative_distribution(const Graph& G) {
  std::vector<int> cdf;
  int sum = 0;
  for (int u : G.nodes()) {
    sum += G.degree(u);
    cdf.push_back(sum);
  }
  return cdf;
}


// Calculate the D matrix for the given number of nodes.
std::vector<std::vector<int>> D_matrix(int n) {
  std::vector<std::vector<int>> D(n, std::vector<int>(n));
  std::vector<int> un(n - 1), um(n - 1);
  std::iota(un.begin(), un.end(), 1);
  std::iota(um.rbegin(), um.rend(), 1);
  std::vector<int> u(n);
  u[0] = 0;
  for (int i = 1; i < n; i++) {
    u[i] = (un[i - 1] < um[i - 1]) ? un[i - 1] : um[i - 1];
  }
  for (int v = 0; v < std::ceil(n / 2.0); v++) {
    std::vector<int> d(u.begin() + v + 1, u.end());
    d.insert(d.end(), u.begin(), u.begin() + v + 1);
    D[n - v - 1] = d;
    D[v] = std::vector<int>(d.rbegin(), d.rend());
  }
  return D;
}


// Choose a value from the given cumulative distribution.
int discrete_sequence(int n, const std::vector<int>& cdf, std::mt19937& rng) {
  std::uniform_int_distribution<int> dist(0, cdf.back());
  int value = dist(rng);
  return std::lower_bound(cdf.begin(), cdf.end(), value) - cdf.begin();
}


int random_choice(const std::vector<int>& cdf, std::mt19937& rng) {
  return discrete_sequence(1, cdf, rng);
}

// takes as input a graph, an int, an array of distance to the diagonal matrix (default None), a bool for connected (default True)
Graph lattice_reference2(const Graph& G, int niter, std::vector<std::vector<int>> distance_to_diagonal = {}, bool connected = true) {
  Graph H;

  // Add the nodes to the new graph
  for (int u : G.nodes()) {
    H.add_node(u);
  }
  
  // if there are less then 4 nodes, return an error
  if (G.nodes().size() < 4) {
    std::cout << "Error: Graph must have at least 4 nodes" << std::endl;
    return H;
  }

    // Calculate the cumulative distribution of node degree
  std::vector<int> cdf = cumulative_distribution(G);

  // Calculate the D matrix
  std::vector<std::vector<int>> D = D_matrix(G.num_nodes());

  // niter = niter * nedges
  niter = niter * G.num_edges();

  //  # maximal number of rewiring attempts per 'niter'
  int max_attempts = G.num_nodes() * G.num_edges() / (G.num_nodes() * (G.num_nodes() - 1) / 2);

  // For loop in range niter
  for (int i = 0; i < niter; i++) {
    int n = 0;
    while (n < max_attempts) {
      // pick two random edges without creating edge list 
      // choose source node indices from discrete distribution
      std::mt19937 rng;
      rng.seed(std::random_device{}());
      int u = random_choice(cdf, rng);
      int v = random_choice(cdf, rng);

      // choose target node indices from discrete distribution
      int w = random_choice(cdf, rng);
      int x = random_choice(cdf, rng);
           

      // check if the edges are distinct
      if (u == v || u == w || u == x || v == w || v == x || w == x) {
        n++;
        continue;
      }

      // check if the edges are already present
      if (H.has_edge(u, v) || H.has_edge(u, w) || H.has_edge(u, x) || H.has_edge(v, w) || H.has_edge(v, x) || H.has_edge(w, x)) {
        n++;
        continue;
      }

      // check if the edges are neighbors
      if (G.has_edge(u, v) || G.has_edge(u, w) || G.has_edge(u, x) || G.has_edge(v, w) || G.has_edge(v, x) || G.has_edge(w, x)) {
        n++;
        continue;
      }

      // check if the edges are in the same distance to the diagonal. Do not create parallel edges
      if (distance_to_diagonal.size() > 0) {
        int u_x = u / n;
        int u_y = u % n;
        int v_x = v / n;
        int v_y = v % n;
        int w_x = w / n;
        int w_y = w % n;
        int x_x = x / n;
        int x_y = x % n;
        int d_uv = distance_to_diagonal[u_x][u_y] + distance_to_diagonal[v_x][v_y];
        int d_ux = distance_to_diagonal[u_x][u_y] + distance_to_diagonal[x_x][x_y];
        int d_uw = distance_to_diagonal[u_x][u_y] + distance_to_diagonal[w_x][w_y];
        int d_vx = distance_to_diagonal[v_x][v_y] + distance_to_diagonal[x_x][x_y];
        int d_vw = distance_to_diagonal[v_x][v_y] + distance_to_diagonal[w_x][w_y];
        int d_wx = distance_to_diagonal[w_x][w_y] + distance_to_diagonal[x_x][x_y];
        if (d_uv == d_ux || d_uv == d_uw || d_uv == d_vx || d_uv == d_vw || d_uv == d_wx) {
          n++;
          continue;
        }
      }

      // check if the edges are connected
      if (connected) {
        if (has_path(H, u, v) || has_path(H, u, w) || has_path(H, u, x) || has_path(H, v, w) || has_path(H, v, x) || has_path(H, w, x)) {
          n++;
          continue;
        }
      }

      // add the edges to the new graph
      H.add_edge(u, v);
      H.add_edge(u, w);
      H.add_edge(u, x);
      H.add_edge(v, w);
      H.add_edge(v, x);
      H.add_edge(w, x);
      break;

    }
  }

  return H;
}
 

// Latticize the given graph by swapping edges and return the modified graph.
Graph latticize(const Graph& G, int niter) {
  Graph H = G;

  // Create a random number generator
  std::mt19937 rng;
  rng.seed(std::random_device{}());

  std::vector<int> degrees;
  for (int u : H.nodes()) {
    degrees.push_back(H.degree(u));
  }

  // Create a cumulative distribution of the node degrees
  std::vector<int> keys;
  for (int u : H.nodes()) {
    keys.push_back(u);
    degrees.push_back(H.degree(u));
  }
  std::vector<int> cdf = cumulative_distribution(H);

  for (int i = 0; i < niter; i++) {
    // Choose two random nodes
    int u = keys[random_choice(cdf, rng)];
    int v = keys[random_choice(cdf, rng)];

    // Choose two random neighbors of the nodes
    std::vector<int> neighbors_u = H.neighbors(u);
    int w = neighbors_u[rng() % neighbors_u.size()];
    std::vector<int> neighbors_v = H.neighbors(v);
    int x = neighbors_v[rng() % neighbors_v.size()];

    // Swap the edges
    H.add_edge(u, x);
    H.add_edge(v, w);
    H.remove_edge(u, w);
    H.remove_edge(v, x);
  }

  // Return the modified graph
  return H;
}


// Calculate the clustering coefficient of a node
double clustering_coefficient(const Graph& G, int u) {
  std::unordered_set<int> neighbors;
  for (int v : G.neighbors(u)) {
    neighbors.insert(v);
  }
  int num_neighbors = neighbors.size();
  if (num_neighbors < 2) {
    return 0.0;
  }
  int num_edges = 0;
  for (int v : neighbors) {
    for (int w : neighbors) {
      if (v < w && G.has_edge(v, w)) {
        num_edges++;
      }
    }
  }
  return static_cast<double>(num_edges) / (num_neighbors * (num_neighbors - 1) / 2.0);
}


// Calculate the average clustering coefficient of a graph
double average_clustering(const Graph& G) {
  int num_nodes = G.nodes().size();
  double sum = 0.0;
  for (int u : G.nodes()) {
    sum += clustering_coefficient(G, u);
  }
  return sum / num_nodes;
}


// Calculate the shortest path between two nodes using breadth-first search (Dijkstra's algorithm)
std::vector<int> shortest_path(const Graph& G, int source, int target) {
  std::queue<int> q;
  std::unordered_map<int, int> predecessor;
  std::unordered_set<int> visited;

  q.push(source);
  visited.insert(source);

  while (!q.empty()) {
    int u = q.front();
    q.pop();
    for (int v : G.neighbors(u)) {
      if (visited.count(v) == 0) {
        visited.insert(v);
        predecessor[v] = u;
        q.push(v);
      }
    }
  }

  // Construct the shortest path
  std::vector<int> path;
  if (predecessor.count(target) > 0) {
    int u = target;
    while (u != source) {
      path.push_back(u);
      u = predecessor[u];
    }
    path.push_back(source);
  }
  std::reverse(path.begin(), path.end());
  return path;
}


// Calculate the average shortest path length of a graph 
double average_shortest_path_length(const Graph& G) {
  double sum = 0.0;
  int num_paths = 0;
  for (int u : G.nodes()) {
    for (int v : G.nodes()) {
      if (u < v && has_path(G, u, v)) {
        sum += shortest_path(G, u, v).size();
        num_paths++;
      }
    }
  }
  return sum / num_paths;
}


// Calculate the average degree of a graph
double average_degree(const Graph& G) {
  int num_nodes = G.nodes().size();
  int sum = 0;
  for (int u : G.nodes()) {
    sum += G.neighbors(u).size();
  }
  return static_cast<double>(sum) / num_nodes;
}


// Calculate the omega index of a graph
double omega(const Graph& G, int niter=2, int nrand=2) {
  double C = average_clustering(G);
  std::cout << "Clustering Coefficient of the original graph  = " << C << std::endl;
  double L = average_shortest_path_length(G);
  std::cout << "L = " << L << std::endl;

  double Lr_sum = 0.0;
  std::cout << "Starting random reference" << std::endl;
  for (int i = 0; i < nrand; i++) {
    std::cout << "\tIteration " << i+1 << std::endl;
    Graph H = random_reference(G, niter);
    std::cout << "\tCreated random reference" << std::endl;
    Lr_sum += average_shortest_path_length(H);
    std::cout << "\tCalculated average shortest path length of the random reference" << std::endl;
  }
  double Lr = Lr_sum / nrand;

  double Cl_sum = 0.0;
  std::cout << "Starting lattice reference" << std::endl;
  for (int i = 0; i < nrand; i++) {
    std::cout << "\tIteration " << i+1 << std::endl;
    Graph H = latticize(G, niter);
    std::cout << "\tCreated lattice reference" << std::endl;
    Cl_sum += average_clustering(H);
    std::cout << "\tCalculated average clustering of the lattice reference" << std::endl;
  }
  double Cl = Cl_sum / nrand;

  return Lr / L - C / Cl;
}


// Calculate the sigma index of a graph
double sigma(const Graph& G, int niter=2, int nrand=2) {
  double C = average_clustering(G);
  std::cout << "Clustering Coefficient of the original graph  = " << C << std::endl;
  double L = average_shortest_path_length(G);
  std::cout << "Average shortest path of the original graph = " << L << std::endl;

  double Lr_sum = 0.0;
  double Cl_sum = 0.0;
  for (int i = 0; i < nrand; i++) {
    std::cout << "\tIteration " << i+1 << std::endl;
    Graph H = random_reference(G, niter);
    std::cout << "\tCreated random reference" << std::endl;
    Lr_sum += average_shortest_path_length(H);
    std::cout << "\tCalculated average shortest path length of the random reference" << std::endl;
    Cl_sum += average_clustering(H);
    std::cout << "\tCalculated average clustering of the random reference" << std::endl;
  }
  double Lr = Lr_sum / nrand;
  double Cl = Cl_sum / nrand;

  return (Lr / L) / (C / Cl);
}


int main() {

  std::cout << "\nStarting the computation for the Foursquare network" << std::endl;
  Graph foursquare = read_graph("data/foursquare/foursquare_friends_edges_filtered.tsv");
  // std::cout << "Omega: " << omega(foursquare) << std::endl;
  std::cout << "Sigma: " << sigma(foursquare) << std::endl;

  std::cout << "\nStarting the computation for the Brightkite network" << std::endl;
  Graph brightkite = read_graph("data/brightkite/brightkite_friends_edges_filtered.tsv");
  // std::cout << "Omega: " << omega(brightkite) << std::endl;
  std::cout << "Sigma: " << sigma(brightkite) << std::endl;

  std::cout << "\nStarting the computation for the Gowalla network" << std::endl;
  Graph gowalla = read_graph("data/gowalla/gowalla_friends_edges_filtered.tsv");
  // std::cout << "Omega: " << omega(gowalla) << std::endl;
  std::cout << "Sigma: " << sigma(gowalla) << std::endl;


 
  return 0;

}
