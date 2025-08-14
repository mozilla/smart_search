// tokenizer.cpp
#include "tokenizer.h"
#include <sstream>
#include <algorithm>
#include <cctype>

std::string to_lower(const std::string &input) {
    std::string out;
    out.reserve(input.size());
    for (char ch : input) {
        out.push_back(std::tolower(static_cast<unsigned char>(ch)));
    }
    return out;
}

std::string remove_punctuation(const std::string &input) {
    std::string output;
    std::copy_if(input.begin(), input.end(), std::back_inserter(output),
                 [](char c){ return !std::ispunct(static_cast<unsigned char>(c)); });
    return output;
}


std::vector<std::string> whitespace_tokenize(const std::string &text) {
    std::istringstream iss(text);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}

std::vector<std::string> wordpiece_tokenize(const std::string &word,
                                              const std::unordered_set<std::string> &vocab,
                                              const std::string &unk_token) {
    std::vector<std::string> output_tokens;
    size_t start = 0;
    size_t len = word.size();
    bool is_bad = false;
    
    while (start < len) {
        size_t end = len;
        std::string curr_substr;
        bool found = false;
        
        while (start < end) {
            std::string substr = word.substr(start, end - start);
            if (start > 0) {
                substr = "##" + substr;
            }
            if (vocab.find(substr) != vocab.end()) {
                curr_substr = substr;
                found = true;
                break;
            }
            end--;
        }
        if (!found) {
            is_bad = true;
            break;
        }
        output_tokens.push_back(curr_substr);
        start = end;
    }
    
    if (is_bad) {
        output_tokens.clear();
        output_tokens.push_back(unk_token);
    }
    return output_tokens;
}
