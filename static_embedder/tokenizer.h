// tokenizer.h
#ifndef TOKENIZER_H
#define TOKENIZER_H

#include <string>
#include <vector>
#include <unordered_set>

// Function declarations
std::string to_lower(const std::string &input);
std::string remove_punctuation(const std::string &input);
std::vector<std::string> whitespace_tokenize(const std::string &text);
std::vector<std::string> wordpiece_tokenize(const std::string &word,
                                              const std::unordered_set<std::string> &vocab,
                                              const std::string &unk_token = "[UNK]");

#endif // TOKENIZER_H
