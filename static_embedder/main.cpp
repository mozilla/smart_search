
#include <fstream>
#include <vector>
#include <string>
#include <cstdint>
#include <iostream>
#include <unordered_set>
#include <unordered_map>
#include <sstream>
#include "tokenizer.h"
#include <thread>
#include <chrono>

struct EmbeddingHeader {
    char magic[4];
    uint32_t vocab_size;
    uint32_t embedding_dim;
};

int main() {
    std::ifstream infile("/Users/cgopal/Downloads/embeddings_dim_256.bin", std::ios::binary);
    if (!infile) {
        std::cerr << "Error opening file.\n";
        return 1;
    }
    
    // Read header
    EmbeddingHeader header;
    infile.read(header.magic, 4);
    if (std::string(header.magic, 4) != "EMBD") {
        std::cerr << "Invalid file format.\n";
        return 1;
    }
    infile.read(reinterpret_cast<char*>(&header.vocab_size), sizeof(header.vocab_size));
    infile.read(reinterpret_cast<char*>(&header.embedding_dim), sizeof(header.embedding_dim));

    // Read vocabulary
    std::vector<std::string> vocab;
    for (size_t i = 0; i < header.vocab_size; ++i) {
        uint32_t len;
        infile.read(reinterpret_cast<char*>(&len), sizeof(len));
        std::string word(len, '\0');
        infile.read(&word[0], len);
        vocab.push_back(word);
    }
    
    // Convert vector to map: token -> index
    std::unordered_map<std::string, size_t> vocab_map;
    for (size_t i = 0; i < vocab.size(); ++i) {
        vocab_map[vocab[i]] = i;
    }
    
    std::unordered_set<std::string> vocab_set(vocab.begin(), vocab.end());
    // Read embedding matrix
    size_t num_elements = header.vocab_size * header.embedding_dim;
    std::vector<float> embeddings(num_elements);
    infile.read(reinterpret_cast<char*>(embeddings.data()), num_elements * sizeof(float));
    
    std::cout << "Loaded " << header.vocab_size << " embeddings of dimension " << header.embedding_dim << "\n";
    
    
    auto start = std::chrono::high_resolution_clock::now();
    // Example input query.
    //    std::string query = "Hello, world! This is a test extraordinarily the simplest.";
    //    std::string query = "diabetes treatment";
    std::string query = "hello world this is a test extraordinarily the simplest";
    
    // Convert the query to lowercase.
    std::string lower_query = to_lower(query);

    // Tokenize by whitespace.
    std::vector<std::string> words = whitespace_tokenize(lower_query);

    // Optionally, remove punctuation from each word.
    std::vector<std::string> processed_words;
    for (const auto &word : words) {
        std::string cleaned = remove_punctuation(word);
        if (!cleaned.empty()) {
            processed_words.push_back(cleaned);
        }
    }

    // Prepare the final token list.
    std::vector<std::string> final_tokens;

    // Tokenize each word using WordPiece.
    for (const auto &word : processed_words) {
        auto sub_tokens = wordpiece_tokenize(word, vocab_set);
        final_tokens.insert(final_tokens.end(), sub_tokens.begin(), sub_tokens.end());
    }

    // Join final_tokens into a single string separated by spaces.
    std::ostringstream oss;
    for (size_t i = 0; i < final_tokens.size(); ++i) {
        if (i > 0) {
            oss << " ";
        }
        oss << final_tokens[i];
    }
    std::string joined_tokens = oss.str();
    
    for (const auto &tok : final_tokens) {
        std::cout << tok << ": " << vocab_map[tok] << std::endl;
    }
    
    // Compute the mean embedding for the tokens.
    std::vector<float> mean_embedding(header.embedding_dim, 0.0f);
    size_t tokenCount = 0;
    for (const auto &tok : final_tokens) {
        auto it = vocab_map.find(tok);
        if (it != vocab_map.end()) {
            size_t idx = it->second;
            // Each token's embedding is at position idx * header.embedding_dim.
            for (size_t d = 0; d < header.embedding_dim; d++) {
                mean_embedding[d] += embeddings[idx * header.embedding_dim + d];
            }
            tokenCount++;
        }
    }
    if (tokenCount > 0) {
        for (size_t d = 0; d < header.embedding_dim; d++) {
            mean_embedding[d] /= tokenCount;
        }
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    std::cout << "Execution time: " << duration.count() << " microseconds" << std::endl;
   
    // Print the first few dimensions of the mean embedding.
    std::cout << "Mean embedding (first 10 dimensions):" << std::endl;
    for (size_t d = 0; d < std::min((size_t)10, (size_t)header.embedding_dim); d++) {
        std::cout << mean_embedding[d] << " ";
    }
    std::cout << std::endl;
    
    // Now you can use the vocabulary and embeddings as needed.
    return 0;
}
