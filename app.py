# copy from https://github.com/huggingface/transformers/blob/v4.22.1/src/transformers/models/clip/tokenization_clip.py

import difflib
import pytz
import datetime
from functools import lru_cache

import ftfy
import regex as re
import streamlit as st

from prompt_parser import parse_prompt_attention

def get_current_time():
    tdatetime = datetime.datetime.now(pytz.timezone('Asia/Tokyo'))
    tstr = tdatetime.strftime('%Y/%m/%d %H:%M')
    return tstr


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


def get_diff(a, b):
    d = difflib.Differ()
    diffs = list(d.compare(a, b))
    sentence = []
    flg_diff = diffs[0][0]
    cache = diffs[0][2]
    for diff in diffs[1:]:
        if diff[0] == flg_diff:
            cache += diff[2]
        else:
            sentence.append([cache, flg_diff])
            flg_diff = diff[0]
            cache = diff[2]
    sentence.append([cache, flg_diff])
    return sentence


def reformat_prompt(prompt):
    reformat = re.sub("\n", " ", prompt)
    reformat = re.sub(" +", " ", reformat)
    return reformat


def draw_html_prompt(text_list):
    html_list = []
    for i, word in enumerate(text_list):
        word = word.replace("</w>", "")
        if word == "_":
            word = "\_"
        if i % 2==0:
            # html_list.append(f'<div style="color:coral; display: inline-block; _display: inline;">{word}</div>')
            html_list.append(f'<span style="color:coral;">{word}</span>')
        else:
            # html_list.append(f'<div style="color:darkgray; display: inline-block; _display: inline;">{word}</div>')
            html_list.append(f'<span style="color:darkgray;">{word}</span>')
    st.write(" | ".join(html_list), unsafe_allow_html=True)


def draw_html_diff(text_diff):
    html_list = []
    for word, flg in text_diff:
        if word == "_":
            word = "\_"
        if flg in ["+", "-"]:
            if word == " ":
                word = "_"
        if flg == "-":
            html_list.append(f'<span style="color:royalblue;">{word}</span>')
        elif flg == "+":
            html_list.append(f'<span style="color:orangered;">{word}</span>')
        else:
            html_list.append(f'<span style="color:darkgray;">{word}</span>')
    st.write("".join(html_list), unsafe_allow_html=True)


# 最大履歴
max_history = 25

def main():
    # setting
    mode_input = st.radio("モード", ["Original", "AUTOMATIC1111", "NovelAI"], horizontal=True)
    if mode_input == "NovelAI":
        add_quality_tags = st.checkbox("Add Quality Tags (+5)", value=True)
    else:
        add_quality_tags = False
    prompt = st.text_area("プロンプトを入力（ボックス右下からサイズ変更可能）", height=130)
    if prompt != "":
        if mode_input == "AUTOMATIC1111":
            parsed_prompt = "".join([t for t, w in parse_prompt_attention(
                prompt, "automatic"
                )])
            max_size = 75 * 3
        elif mode_input == "NovelAI":
            parsed_prompt = "".join([t for t, w in parse_prompt_attention(
                prompt, "novelai"
                )])
            max_size = 75 * 3
        else:
            parsed_prompt = prompt
            max_size = 75

        bpe_tokens = get_token(parsed_prompt)
        text_size = len(bpe_tokens)
        if add_quality_tags:
            text_size += 5
        st.text(f"token数: {text_size} / {max_size}")
        draw_html_prompt(bpe_tokens[0:max_size])
        if len(bpe_tokens) >= max_size:
            st.text("--- over ---")
            draw_html_prompt(bpe_tokens[max_size:])
        st.text("\n")
        st.text("コピー用（改行を空白に変換、２つ以上の空白を１つに変換）")
        mode_replace = st.radio("プロンプトの変換", ["なし", "for AUTOMATIC1111: {}->()", "for NovelAI: ()->{}", "({[]})を消す"], horizontal=True)
        if mode_replace == "for AUTOMATIC1111: {}->()":
            prompt_replace = prompt.replace("{", "(").replace("}", ")")
        elif mode_replace == "for NovelAI: ()->{}":
            prompt_replace = prompt.replace("(", "{").replace(")", "}")
        elif mode_replace == "({[]})を消す":
            prompt_replace = prompt.replace("(", "").replace(")", "")
            prompt_replace = prompt_replace.replace("{", "").replace("}", "")
            prompt_replace = prompt_replace.replace("[", "").replace("]", "")
        else:
            prompt_replace = prompt
        reformat = reformat_prompt(prompt_replace)
        st.code(reformat, language="")

        # if st.button("履歴に一時保存"):
        #     data = {
        #         "date": get_current_time(),
        #         "prompt": reformat
        #     }
        #     if "storage" in st.session_state:
        #         st.session_state["storage"].append(data)
        #         if len(st.session_state["storage"]) > max_history:
        #             st.session_state["storage"].pop(0)
        #     else:
        #         st.session_state["storage"] = [data]
        # st.text("※一時保存はサーバには保存されず再接続時には消えるのでご注意ください")
        
    # if "storage" in st.session_state:
    #     st.markdown("---")
    #     st.markdown(f"### 履歴")
    #     st.text("※最大25個で古いものから消えていきます/差分では空白は_に変換されます")
    #     if prompt != "":
    #         reformat = reformat_prompt(prompt)
    #     for data in st.session_state["storage"][::-1]:
    #         st.markdown(f'**{data["date"]}**')
    #         if prompt != "":
    #             diff = get_diff(reformat, data["prompt"])
    #             draw_html_diff(diff)
    #         st.code(data["prompt"], language="")

if __name__ == "__main__":
    main()
