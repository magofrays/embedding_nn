#include "tokenizer.h"
#include <locale>

int main(int argc, const char **argv)
{
    std::locale::global(std::locale("")); // Или "en_US.UTF-8", "ru_RU.UTF-8" и т. д.
    std::wcout.imbue(std::locale());
    std::wstring simple_text = L"привет я вадим я самый крутой во вселенной";
    std::wcout << simple_text << std::endl;
    BPETokenizer tokenizer;
    auto tokenized_text = tokenizer.train(simple_text, 20);
    for (auto id : tokenized_text)
    {
        std::wcout << id << " ";
    }
    std::wcout << "\n";
    std::wstring decoded_text = tokenizer.decode(tokenized_text);
    std::wcout << decoded_text << std::endl;
    return 0;
}