#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <utility>
#include <boost/graph/adjacency_list.hpp>
#include <boost/graph/small_world_generator.hpp>
#include <boost/random/linear_congruential.hpp>

using namespace std;
using namespace boost;

typedef adjacency_list<vecS, vecS, undirectedS, no_property, no_property> Graph;
typedef small_world_iterator<minstd_rand, Graph> SWGen;

vector<pair<int, int>> lattice_reference(const string& edge_list_file, int niter, bool connectivity) {
    vector<pair<int, int>> edges;
    int num_nodes = 0;

    // Read in the edge list from the input file
    ifstream in(edge_list_file);
    string line;
    while (getline(in, line)) {
        int u, v;
        sscanf(line.c_str(), "%d\t%d", &u, &v);
        edges.emplace_back(u, v);
        num_nodes = max(num_nodes, max(u, v));
    }

    // Construct the graph from the edge list
    Graph g(edges.begin(), edges.end(), num_nodes + 1);

    // Create the small-world generator
    minstd_rand gen;
    SWGen sw_gen(g, niter);

    // Generate the lattice reference and store the resulting edge list
    vector<pair<int, int>> lattice_edges;
    for (int i = 0; i < num_nodes; ++i) {
        auto [u, v] = *sw_gen;
        lattice_edges.emplace_back(u, v);
        ++sw_gen;
    }

    // convert the vector of pairs in a .tsv file called "lattice_reference.tsv"
    ofstream out("lattice_reference.tsv");
    for (const auto& [u, v] : lattice_edges) {
        out << u << "\t" << v << endl;
    }

    // return the vector of pairs
    return lattice_edges;

}

// main

int main(int argc, char* argv[]) {
    if (argc != 4) {
        cerr << "Usage: " << argv[0] << " <edge_list_file> <niter> <connectivity>" << endl;
        return 1;
    }

    string edge_list_file = argv[1];
    int niter = atoi(argv[2]);
    bool connectivity = atoi(argv[3]);

    lattice_reference(edge_list_file, niter, connectivity);

    return 0;
}

 