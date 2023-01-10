# copy from https://github.com/huggingface/transformers/blob/v4.22.1/src/transformers/models/clip/tokenization_clip.py

from math import ceil
import regex as re
import streamlit as st

from bpe import get_token
from prompt_parser import parse_prompt_attention
from footer import footer


def last_reformat(prompt):
    reformat = re.sub("\n", " ", prompt)
    reformat = re.sub(" +", " ", reformat)
    return reformat


def draw_html_prompt(text_list):
    html_list = []
    for i, word in enumerate(text_list):
        word = word.replace("</w>", "")
        if word == "_":
            word = "\_"
        if i % 2 == 0:
            # html_list.append(f'<div style="color:coral; display: inline-block; _display: inline;">{word}</div>')
            html_list.append(f'<span style="color:coral;">{word}</span>')
        else:
            # html_list.append(f'<div style="color:darkgray; display: inline-block; _display: inline;">{word}</div>')
            html_list.append(f'<span style="color:darkgray;">{word}</span>')
    st.write(" | ".join(html_list), unsafe_allow_html=True)


def reformat_nijijourney(prompt, mode_no_detail):
    clip_max = 75
    parsed_prompt = prompt
    max_size = clip_max

    # 正規表現での分離
    regex = r'::[+-]?(?:\d+\.?\d*|\.\d+)?'
    prompt_list = [p for p in re.split(regex, parsed_prompt) if p.strip() != ""]

    def convert_weight(w):
        if w == "::":
            return 1.0
        else:
            return float(w.replace("::", ""))

    weight_str_list = [weight for weight in re.findall(regex, parsed_prompt)]
    weight_list = [convert_weight(weight) for weight in weight_str_list]
    
    if len(prompt_list) > len(weight_list):
        weight_list.append(1.0)
    text_size_list = []
    for k, prompt in enumerate(prompt_list):
        bpe_tokens = get_token(prompt)
        text_size = len(bpe_tokens)
        text_size_list.append(text_size)
        st.text(f"{k+1}番目のプロンプト: weight={weight_list[k]}, token数: {text_size} / {max_size}")
        if not mode_no_detail:
            draw_html_prompt(bpe_tokens[0:max_size])
            if len(bpe_tokens) >= max_size:
                st.text("--- over ---")
                draw_html_prompt(bpe_tokens[max_size:])
            st.markdown("---")
    if mode_no_detail:
        st.markdown("---")
    
    st.text("プロンプトのコピー（一番下のボックスの右のアイコンからコピーできます）")
    add_size = st.radio("サイズを追加", ["なし", "--ar 2:3", "--ar 3:2"], horizontal=True)
    add_quality = st.radio("クオリティを追加", ["なし", "--q 0.25", "--q 0.5", "--q 2"], horizontal=True)
    fill_word = st.radio("次の文字で75トークンを埋める", ["なし", ";", ",", "+"], horizontal=True)
    if fill_word != "なし":
        prompt_list_new = []
        for p, w, s in zip(prompt_list, weight_list, text_size_list):
            if s < max_size:
                p = p + f" {fill_word}" * (max_size - s)
            prompt_list_new.append(f"{p}::{w}")
        parsed_prompt = " ".join(prompt_list_new)

    reformat_prompt = last_reformat(parsed_prompt)

    if add_size != "なし":
        reformat_prompt = reformat_prompt + f" {add_size}"
    if add_quality != "なし":
        reformat_prompt = reformat_prompt + f" {add_quality}"

    return reformat_prompt


def reformat(prompt, mode_input, mode_no_detail, mode_split_75, add_quality_tags):
    clip_max = 75
    if mode_input == "AUTOMATIC1111":
        parsed_prompt = "".join([t for t, w in parse_prompt_attention(
            prompt, mode_input
            )])
        max_size = None
    elif mode_input == "NovelAI":
        if add_quality_tags:
            prompt_temp = "masterpiece, best quality," + prompt
        else:
            prompt_temp = prompt
        parsed_prompt = "".join([t for t, w in parse_prompt_attention(
            prompt_temp, mode_input
            )])
        max_size = clip_max * 3            
    else:
        parsed_prompt = prompt
        max_size = clip_max

    bpe_tokens = get_token(parsed_prompt)
    text_size = len(bpe_tokens)
    if max_size is not None:
        st.text(f"token数: {text_size} / {max_size}")
    else:
        st.text(f"token数: {text_size} / ∞")

    if not mode_no_detail:
        if mode_split_75:
            n_row = ceil(len(bpe_tokens) / clip_max)
            for i0 in range(n_row):
                st.text(f"{clip_max * i0 + 1} ~ {clip_max * (i0 + 1)}")
                draw_html_prompt(bpe_tokens[clip_max * i0: clip_max * (i0 + 1)])
        else:
            if max_size is not None:
                draw_html_prompt(bpe_tokens[0:max_size])
                if len(bpe_tokens) >= max_size:
                    st.text("--- over ---")
                    draw_html_prompt(bpe_tokens[max_size:])
            else:
                draw_html_prompt(bpe_tokens)
    st.markdown("---")

    st.text("プロンプトのコピー（一番下のボックスの右のアイコンからコピーできます）")
    mode_replace = st.radio(
        "プロンプトの変換",
        ["なし", "for AUTOMATIC1111: {}->()", "for NovelAI: ()->{}", "({[]})を消す"],
        horizontal=True)
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

    reformat_prompt = last_reformat(prompt_replace)
    return reformat_prompt


def main():
    st.set_page_config(page_title="Prompt Counter")
    st.markdown(footer, unsafe_allow_html=True)

    # setting
    mode_input = st.radio("プロンプトを実行するサービスを選択してください",
        ["対象なし", "nijijourney", "AUTOMATIC1111", "NovelAI"],
        horizontal=True)
    if mode_input == "NovelAI":
        add_quality_tags = st.checkbox("Add Quality Tags (+5)", value=False)
    else:
        add_quality_tags = False
    prompt = st.text_area("プロンプトを入力（ボックス右下からサイズ変更可能）", height=300)

    col0, col1 = st.columns(2)
    mode_no_detail = col0.checkbox("プロンプト分割の詳細を非表示", value=False)
    if mode_input != "nijijourney":
        mode_split_75 = col1.checkbox("75トークンごとに分割", value=True)

    if prompt != "":
        st.markdown("---")
        if mode_input == "nijijourney":
            reformat_prompt = reformat_nijijourney(prompt, mode_no_detail)
        else:
            reformat_prompt = reformat(
                prompt, mode_input,
                mode_no_detail, mode_split_75, add_quality_tags)

        st.code(reformat_prompt, language="")


if __name__ == "__main__":
    main()
