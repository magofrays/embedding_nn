#include <vector>
#include <cmath>
#include <vector>
#include <iostream>
#include <map>
#include <string>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <deque>
#include <stdexcept>

class BPETokenizer
{
    std::vector<std::wstring> unique_words;
    std::map<size_t, std::wstring> vocab;
    std::map<std::wstring, size_t> inverse_vocab;
    std::map<std::pair<size_t, size_t>, size_t> bpe_merges;

    std::wstring text;
    std::vector<size_t> token_ids;

public:
    BPETokenizer() = default;
    std::vector<size_t> train(const std::wstring &text, size_t vocab_size);         // creates tokens from text and tokenizes it
    std::pair<size_t, size_t> find_freq_pair(const std::vector<size_t> &token_ids); // finds most freq pair
    std::vector<size_t> replace_pair(const std::vector<size_t> &token_ids,          // creates new token_ids where most freq pair is merged
                                     const std::pair<size_t, size_t> &pair_id,
                                     size_t new_id);

    std::wstring decode(const std::vector<size_t> &token_ids);    // gets token_ids and makes text from it
    std::vector<size_t> encode(const std::wstring &text);         // gets text and tokenizes it by using tokenize_word
    std::vector<size_t> tokenize_word(const std::wstring &token); // gets word and tokenizes it by using already made tokens
};