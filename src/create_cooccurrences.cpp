#include <string>
#include <unordered_map>
#include <fstream>
#include <iostream>
#include <assert>
#include <rocksdb/db.h>

// Create vocabulary dictionary

#define corplen 17005207
#define windowsize = 15

bool create_index_map(std::map<std::string, int>& indices, std::string filename) {
    std::ifstream vocab_in_file(filename);
    
    if (!vocab_in_file.is_open()) {
        return false;
    }

    std::string in_string;
    int freq, i = 0;

    while (vocab_in_file >> in_string >> freq) {
        indices[in_string] = i++;
    }

    return true;
}

int main(int argc, char **argv) {
    if (argc != 4) {
        std::cerr << "USAGE: ./create_frequencies vocab.txt corpus.txt db_filename\n";
        return EXIT_FAILURE;
    }

    std::unordered_map<std::string, int> indices;

    if (!create_index_map(indices, argv[1])) {
        std::cerr << "ERROR: vocab.txt not opened. Exiting...";
        return EXIT_FAILURE;
    }

    rocksdb::DB* db;
    rocksdb::Options options;
    options.create_if_missing = true;
    rocksdb::Status status = rocksdb::DB::Open(options, argv[3], &db);

    if (!status.ok()) {
        std::cerr << "ERROR: opening \"" << argv[3] << "\" failed with status " << s.ToString() << ". Exiting...\n";
        return EXIT_FAILURE;
    }



    delete db;
    return EXIT_SUCCESS;
}