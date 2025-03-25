#include "tokenizer.h"

std::vector<size_t> BPETokenizer::train(const std::wstring &text, size_t vocab_size)
{
    this->text = text;
    std::vector<std::wstring> processed_text(text.size());
    size_t unique_counter = 0;
    for (size_t i = 0; i != text.size(); i++)
    {
        std::wstring unique_word(1, text[i]);
        if (iswspace(text[i]) && i > 0)
        {
            processed_text[i] = L"空";
        }
        else
        {
            processed_text[i] = unique_word;
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
    for (size_t new_id = unique_words.size(); new_id != unique_words.size() + vocab_size && token_ids.size() != 1; new_id++)
    {
        std::wcout << new_id << "\n";
        auto pair_id = find_freq_pair(token_ids);

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
    return token_ids;
}

std::pair<size_t, size_t> BPETokenizer::find_freq_pair(const std::vector<size_t> &token_ids)
{
    std::vector<std::pair<size_t, size_t>> pairs(token_ids.size() - 1);
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
                if (pairs[i].first == pairs[j].first && pairs[i].second == pairs[j].second)
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

std::vector<size_t> BPETokenizer::replace_pair(const std::vector<size_t> &token_ids,
                                               const std::pair<size_t, size_t> &pair_id,
                                               size_t new_id)
{
    std::vector<size_t> new_token_ids;
    for (size_t i = 0; i < token_ids.size() - 1; i++)
    {
        if (token_ids[i] == pair_id.first && token_ids[i + 1] == pair_id.second)
        {
            new_token_ids.push_back(new_id);
            i++;
        }
        else
        {
            new_token_ids.push_back(token_ids[i]);
        }
        if (i == token_ids.size() - 2)
        {
            new_token_ids.push_back(token_ids[i + 1]);
        }
    }
    return new_token_ids;
}

std::wstring BPETokenizer::decode(const std::vector<size_t> &token_ids)
{
    std::wstring decoded_string;
    for (size_t token_id : token_ids)
    {
        if (vocab.find(token_id) == vocab.end())
        {
            throw std::logic_error("decode: no such token in vocab!");
        }
        std::wstring token = vocab[token_id];
        if (token[0] == L'空')
        {
            token.erase(0, 1);
            decoded_string += L" ";
        }
        decoded_string += token;
    }
    return decoded_string;
}

std::vector<size_t> BPETokenizer::encode(const std::wstring &text)
{
    std::vector<std::wstring> tokens;
    std::wstring word;
    for (size_t i = 0; i != text.size(); i++)
    {
        if (iswspace(text[i]) && i > 0)
        {
            tokens.push_back(word);
            word.clear();
            word += L"空";
        }
        else
        {
            word += text[i];
        }
    }
    tokens.push_back(word);
    std::vector<size_t> token_ids;
    for (auto &token : tokens)
    {
        if (inverse_vocab.find(token) != inverse_vocab.end())
        {
            size_t token_id = inverse_vocab[token];
            token_ids.push_back(token_id);
        }
        else
        {
            std::vector<size_t> tokenized_word = tokenize_word(token);
            token_ids.insert(token_ids.end(), tokenized_word.begin(), tokenized_word.end());
        }
    }
    return token_ids;
}

std::vector<size_t> BPETokenizer::tokenize_word(const std::wstring &token)
{
    std::vector<size_t> token_ids(token.size());
    for (size_t i = 0; i < token.size(); i++)
    {
        std::wstring symbol = std::wstring(1, token[i]);
        if (inverse_vocab.find(symbol) == inverse_vocab.end())
        {
            throw std::logic_error("tokenize_with_bpe: no such symbol in vocab");
        }
        token_ids[i] = inverse_vocab[symbol];
    }
    bool can_merge = true;
    while (can_merge && token_ids.size() > 1)
    {
        can_merge = false;
        std::vector<size_t> new_token_ids;
        size_t i = 0;
        while (i < token_ids.size() - 1)
        {
            std::pair<size_t, size_t> token_pair(token_ids[i], token_ids[i + 1]);
            if (bpe_merges.find(token_pair) != bpe_merges.end())
            {
                size_t merged_token_id = bpe_merges[token_pair];
                new_token_ids.push_back(merged_token_id);
                i += 2;
                can_merge = true;
            }
            else
            {
                new_token_ids.push_back(token_ids[i]);
                i += 1;
            }
        }
        if (i < token_ids.size())
        {
            new_token_ids.push_back(token_ids[i]);
        }
        token_ids = new_token_ids;
    }
    return token_ids;
}
