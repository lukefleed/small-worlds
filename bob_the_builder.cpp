#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <thread>
#include <functional>
#include <set>

using namespace std;

// It receives the file name as a string and returns a dictionary with the keys being the UserID and the values being a vector of VenueID associated with that UserID.

unordered_map<string, set<string>> createDictFromFile(string filename) {
    // Create an empty dictionary
    unordered_map<string, set<string>> dict;

    // Open the file
    ifstream file(filename);

    // Check if the file was opened successfully
    if (!file.is_open()) {
        cerr << "Error opening file " << filename << endl;
        return dict;
    }

    // Read the file line by line
    string userId, venueId;
    while (file.good()) {
        file >> userId >> venueId;
        // Add the venueId to the vector of venues associated with the userId
        dict[userId].insert(venueId);
    }

    cout << "Dict created" << endl;

    // Return the dictionary
    return dict;
    }

// void create_tsv_multi(unordered_map<string, vector<string>> dict, mutex& dict_mutex) {
//       // Create an output stream to write the file
//       ofstream out_file("output.tsv");
//       // Create a mutex to protect the output file
//       mutex out_file_mutex;
//       // Loop over all the key-value pairs in the map
//       for (const auto& kv1 : dict) {
//         for (const auto& kv2 : dict) {
//           // Check if the keys are the same
//           if (kv1.first == kv2.first) continue;
//           // Check if the values have elements in common
//           vector<string> common;
//           for (const auto& str1 : kv1.second) {
//             for (const auto& str2 : kv2.second) {
//               if (str1 == str2) common.push_back(str1);
//             }
//           }
//           // Write the keys and the number of common elements to the output file
//           if (!common.empty()) {
//             // Lock the mutexes before accessing the dict and the output file
//             lock_guard<mutex> dict_guard(dict_mutex);
//             lock_guard<mutex> out_file_guard(out_file_mutex);
//             out_file << kv1.first << "\t" << kv2.first << "\t" << common.size() << endl;
//           }
//         }
//       }
//     }

void create_tsv(unordered_map<string, set<string>> dict) {
    // Create an output stream to write the file
    ofstream out_file("data/foursquare/foursquareTKY_checkins_graph.tsv");

    // Loop over all the key-value pairs in the map
    unsigned long long i = 0;
    for (const auto& kv1 : dict) {
        i++;
        if (i%100 == 0) cout << (((double)i) * 100 / dict.size()) << "%" << "\r" << flush;

        for (const auto& kv2 : dict) {
            // Check if the keys are the same
            if(kv1.first >= kv2.first) continue;

            // Check if the values have elements in common
            set<string> common;
            set_intersection(kv1.second.begin(), kv1.second.end(), kv2.second.begin(), kv2.second.end(), inserter(common, common.begin()));

            // Write the keys and the number of common elements to the output file
            if (!common.empty()) {
                out_file << kv1.first << "\t" << kv2.first << "\t" << common.size() << endl;
                // cout << kv1.first << "\t" << kv2.first << "\t" << common.size() << endl;
            }
        }
    }
}

void print_help() {
  cout << "Usage: ./main [IN_FILE] [OUT_FILE]" << endl;
}

int main() {
    unordered_map<string, set<string>> dict = createDictFromFile("data/foursquare/foursquare_checkins_TKY.txt");
    create_tsv(dict);
}



// int main(int argc, const char* argv[]) {

//     if (argc == 3) {
//         string in_file = argv[1];
//         string out_file = argv[2];
//     if (in_file == "-h" || in_file == "--help" || out_file == "-h" || out_file == "--help") {
//         print_help();
//         return 0;
//         }
//     } else {
//         print_help();
//         return 0;
//     }
//     unordered_map<string, set<string>> dict = createDictFromFile("test.txt");
//     create_tsv(dict);

// }
