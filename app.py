# copy from https://github.com/huggingface/transformers/blob/v4.22.1/src/transformers/models/clip/tokenization_clip.py

from functools import lru_cache
import regex as re
import ftfy

import streamlit as st

def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text

pat = re.compile(r"<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+", re.IGNORECASE,)

@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a mapping to unicode strings. We specifically avoids mapping to whitespace/control
    characters the bpe code barfs on.
    The reversible bpe codes work on unicode strings. This means you need a large # of unicode characters in your vocab
    if you want to avoid UNKs. When you're at something like a 10B token dataset you end up needing around 5K for
    decent coverage. This is a significant percentage of your normal, say, 32K bpe vocab. To avoid that, we want lookup
    tables between utf-8 bytes and unicode strings.
    """
    bs = (
        list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(2**8):
        if b not in bs:
            bs.append(b)
            cs.append(2**8 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    Word is represented as tuple of symbols (symbols being variable-length strings).
    """
    pairs = set()
    prev_char = word[0]
    for char in word[1:]:
        pairs.add((prev_char, char))
        prev_char = char
    return pairs


class BPE:
    def __init__(self):
        merges_file = "./merges.txt"
        with open(merges_file, encoding="utf-8") as merges_handle:
            bpe_merges = merges_handle.read().strip().split("\n")[1 : 49152 - 256 - 2 + 1]
        bpe_merges = [tuple(merge.split()) for merge in bpe_merges]
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        self.cache = {"<|startoftext|>": "<|startoftext|>", "<|endoftext|>": "<|endoftext|>"}

    def bpe(self, token):
        if token in self.cache:
            return self.cache[token]
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = get_pairs(word)

        if not pairs:
            return token + "</w>"

        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                else:
                    new_word.extend(word[i:j])
                    i = j

                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_word = tuple(new_word)
            word = new_word
            if len(word) == 1:
                break
            else:
                pairs = get_pairs(word)
        word = " ".join(word)
        self.cache[token] = word
        return word

byte_encoder = bytes_to_unicode()
bpe = BPE()

def get_token(text):
    bpe_tokens = []

    text = whitespace_clean(ftfy.fix_text(text)).lower()

    for token in re.findall(pat, text):
        token = "".join(
            byte_encoder[b] for b in token.encode("utf-8")
        )  # Maps all our bytes to unicode strings, avoiding control tokens of the BPE (spaces in our case)
        bpe_tokens.extend(bpe_token for bpe_token in bpe.bpe(token).split(" "))
    return bpe_tokens

def draw_html(text_list):
    html_list = []
    for i, word in enumerate(text_list):
        if i % 2==0:
            html_list.append(f'<div style="color:coral; display: inline-block; _display: inline;">{word}</div>')
        else:
            html_list.append(f'<div style="color:darkgray; display: inline-block; _display: inline;">{word}</div>')
    st.write(" | ".join(html_list), unsafe_allow_html=True)

def main():
    prompt = st.text_input("プロンプトを入力")

    bpe_tokens = get_token(prompt)
    model_max_length = 77
    text_size = len(bpe_tokens)
    max_size = model_max_length - 2
    st.text(f"size: {text_size} / {max_size}")
    draw_html(bpe_tokens[0:model_max_length-2])
    if len(bpe_tokens) >= model_max_length-2:
        st.text("--- over ---")
        draw_html(bpe_tokens[model_max_length-2:])

if __name__ == "__main__":
    main()
