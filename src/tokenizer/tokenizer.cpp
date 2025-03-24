#include <vector>
#include <cmath>
#include <vector>
#include <iostream>
#include <map>
#include <string>
#include <algorithm>
#include <nlohmann/json.hpp>
#include <deque>

class BPETokenizer
{
    std::map<size_t, std::wstring> vocab;
    std::map<std::wstring, size_t> inverse_vocab;
    std::map<std::pair<size_t, size_t>, size_t> bpe_merges;
    std::vector<std::wstring> unique_words;
    std::wstring text;
    std::vector<size_t> token_ids;
    size_t vocab_size;

public:
    BPETokenizer()
    {
    }

    void train(std::wstring text, size_t vocab_size)
    {
        this->text = text;
        std::wstring result_text;
        std::vector<std::wstring> processed_text(text.size());
        size_t unique_counter = 0;
        for (size_t i = 0; i != text.size(); i++)
        {
            std::wstring unique_word(1, text[i]);
            if (unique_word == L" " && i != 0)
            {
                processed_text[i] = L"空";
            }
            {
                processed_text[i] = std::wstring(1, text[i]);
                if (std::find(unique_words.begin(), unique_words.end(), unique_word) == unique_words.end())
                {
                    unique_words.push_back(unique_word);
                    vocab.try_emplace(unique_counter, unique_word);
                    inverse_vocab.try_emplace(unique_word, unique_counter);
                    unique_counter++;
                }
            }
        }
        unique_words.push_back(L"空");
        vocab[unique_counter] = L"空";
        inverse_vocab[L"空"] = unique_counter;
        unique_counter++;

        std::vector<size_t> token_ids(processed_text.size());
        for (size_t i = 0; i != processed_text.size(); i++)
        {
            token_ids[i] = inverse_vocab[processed_text[i]];
        }

        for (size_t new_id = unique_words.size(); new_id != unique_words.size() + vocab_size; new_id++)
        {
            std::cout << new_id << "\n";
            std::pair<size_t, size_t> pair_id;
            token_ids = replace_pair(token_ids, pair_id, new_id);
            bpe_merges[pair_id] = new_id;
        }

        for (auto it = bpe_merges.begin(); it != bpe_merges.end(); ++it)
        {
            std::pair<size_t, size_t> new_pair = it->first;
            size_t new_id = it->second;
            std::wstring new_token = vocab[new_pair.first] + vocab[new_pair.second];
            vocab[new_id] = new_token;
            inverse_vocab[new_token] = new_id;
        }
    }

    static std::pair<size_t, size_t> find_freq_pair(const std::vector<size_t> &token_ids)
    {
        std::vector<std::pair<size_t, size_t>> pairs(token_ids.size());
        std::map<std::pair<size_t, size_t>, size_t> pairs_counter;
        std::pair<size_t, size_t> max_pair;
        size_t max_counter = 0;
        for (size_t i = 0; i != token_ids.size() - 1; i++)
        {
            pairs[i] = std::make_pair(token_ids[i], token_ids[i + 1]);
        }
        for (size_t i = 0; i != pairs.size(); i++)
        {
            if (pairs_counter.find(pairs[i]) == pairs_counter.end())
            {
                size_t counter = 0;
                for (size_t j = 0; j != pairs.size(); j++)
                {
                    if (pairs[i] == pairs[j])
                    {
                        counter++;
                    }
                }
                pairs_counter[pairs[i]] = counter;
                if (counter > max_counter)
                {
                    max_counter = counter;
                    max_pair = pairs[i];
                }
            }
        }
        return max_pair;
    }

    static std::vector<size_t> replace_pair(const std::vector<size_t> &token_ids,
                                            const std::pair<size_t, size_t> &pair_id,
                                            size_t new_id)
    {
        std::vector<size_t> new_token_ids;
        for (size_t i = 0; i != token_ids.size() - 1; i++)
        {
            if (token_ids[i] == pair_id.first && token_ids[i + 1] == pair_id.second)
            {
                new_token_ids.push_back(new_id);
            }
            else
            {
                new_token_ids.push_back(token_ids[i]);
            }
        }
    }
};
