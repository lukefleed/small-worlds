#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <mutex>
#include <thread>
#include <functional>

using namespace std;

// It receives the file name as a string and returns a dictionary with the keys being the UserID and the values being a vector of VenueID associated with that UserID.

unordered_map<string, vector<string>> createDictFromFile(string filename) {
  // Create an empty dictionary
  unordered_map<string, vector<string>> dict;

  // Open the file
  ifstream file(filename);

  // Check if the file was opened successfully
  if (!file.is_open()) {
    cerr << "Error opening file " << filename << endl;
    return dict;
  }

  // Read the file line by line
  string userId, venueId;
  while (file >> userId >> venueId) {
    // Check if the userId is already in the dictionary
    if (dict.count(userId) == 0) {
      // If not, create an entry in the dictionary with an empty vector of venues
      dict[userId] = vector<string>();
    }

    // Add the venueId to the vector of venues associated with the userId
    dict[userId].push_back(venueId);
  }

  // Close the file
  file.close();

  cout << "Dict created" << endl;

  // Return the dictionary
  return dict;
}

void create_tsv(unordered_map<string, vector<string>> dict, mutex& dict_mutex) {
  // Create an output stream to write the file
  ofstream out_file("output.tsv");

  // Create a mutex to protect the output file
  mutex out_file_mutex;

  // Loop over all the key-value pairs in the map
  for (const auto& kv1 : dict) {
    for (const auto& kv2 : dict) {
      // Check if the keys are the same
      if (kv1.first == kv2.first) continue;

      // Check if the values have elements in common
      vector<string> common;
      for (const auto& str1 : kv1.second) {
        for (const auto& str2 : kv2.second) {
          if (str1 == str2) common.push_back(str1);
        }
      }

      // Write the keys and the number of common elements to the output file
      if (!common.empty()) {
        // Lock the mutexes before accessing the dict and the output file
        lock_guard<mutex> dict_guard(dict_mutex);
        lock_guard<mutex> out_file_guard(out_file_mutex);

        out_file << kv1.first << "\t" << kv2.first << "\t" << common.size() << endl;
      }
    }
  }

  // Close the output file
  out_file.close();
}


int main() {
  // Create a map of vectors
  unordered_map<string, vector<string>> dict = createDictFromFile("test.txt");

  // Create a mutex to protect the dict map
  mutex dict_mutex;

  // Create an array of threads
  const int num_threads = 12;
  thread threads[num_threads];

  // Launch the threads
  for (int i = 0; i < num_threads; i++) {
    threads[i] = thread(create_tsv, ref(dict), ref(dict_mutex));
  }

  // Wait for the threads to finish
  for (int i = 0; i < num_threads; i++) {
    threads[i].join();
  }

  return 0;
}
