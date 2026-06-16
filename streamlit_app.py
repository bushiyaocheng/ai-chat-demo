# -*- coding: utf-8 -*-
"""
Streamlit + DeepSeek chat demo.

Run locally:
    streamlit run outputs/35_AI_chat_complete.py

The API key is read from DEEPSEEK_API_KEY first, then DEEPSEEK_API_KEY_TEST.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any


DEFAULT_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
DEFAULT_SYSTEM_PROMPT = (
    "你是一个耐心、清晰的中文学习助手。回答时尽量给出步骤，"
    "必要时用简单示例帮助用户理解。"
)
ALLOWED_CHAT_ROLES = {"user", "assistant"}


def load_dotenv_if_available() -> None:
    """Load local .env values when python-dotenv is installed."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    load_dotenv()


def get_deepseek_api_key() -> str:
    return (
        os.getenv("DEEPSEEK_API_KEY", "").strip()
        or os.getenv("DEEPSEEK_API_KEY_TEST", "").strip()
    )


def normalize_messages(chat_messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    normalized: list[dict[str, str]] = []

    for message in chat_messages:
        role = str(message.get("role", "")).strip()
        content = str(message.get("content", "")).strip()

        if role in ALLOWED_CHAT_ROLES and content:
            normalized.append({"role": role, "content": content})

    return normalized


def build_api_messages(
    system_prompt: str,
    chat_messages: list[dict[str, Any]],
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    clean_system_prompt = system_prompt.strip()

    if clean_system_prompt:
        messages.append({"role": "system", "content": clean_system_prompt})

    messages.extend(normalize_messages(chat_messages))
    return messages


def build_history_export(
    system_prompt: str,
    model: str,
    chat_messages: list[dict[str, Any]],
) -> str:
    payload = {
        "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model,
        "system_prompt": system_prompt,
        "messages": normalize_messages(chat_messages),
    }

    return json.dumps(payload, ensure_ascii=False, indent=2)


def get_download_file_name() -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"deepseek_chat_history_{timestamp}.json"


def ensure_session_state(st: Any) -> None:
    if "messages" not in st.session_state:
        old_messages = st.session_state.get("message", [])
        st.session_state.messages = normalize_messages(old_messages)

    if "system_prompt" not in st.session_state:
        st.session_state.system_prompt = DEFAULT_SYSTEM_PROMPT

    if "model_name" not in st.session_state:
        st.session_state.model_name = DEFAULT_MODEL

    if "temperature" not in st.session_state:
        st.session_state.temperature = 0.7

    if "history_rounds" not in st.session_state:
        st.session_state.history_rounds = 10


def main() -> None:
    load_dotenv_if_available()

    import streamlit as st
    from openai import OpenAI

    st.set_page_config(
        page_title="AI Partner",
        page_icon="AI",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={},
    )
    ensure_session_state(st)

    st.title("AI Partner")

    with st.sidebar:
        st.header("聊天设置")
        st.text_area("系统提示词", key="system_prompt", height=180)
        st.text_input("模型名称", key="model_name")
        st.slider(
            "携带最近多少轮对话",
            min_value=1,
            max_value=30,
            key="history_rounds",
        )
        st.slider(
            "温度",
            min_value=0.0,
            max_value=1.5,
            step=0.1,
            key="temperature",
        )

        st.divider()
        export_json = build_history_export(
            system_prompt=st.session_state.system_prompt,
            model=st.session_state.model_name,
            chat_messages=st.session_state.messages,
        )
        st.download_button(
            "下载当前对话 JSON",
            data=export_json,
            file_name=get_download_file_name(),
            mime="application/json",
            use_container_width=True,
        )

        if st.button("清空当前对话", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        api_key = get_deepseek_api_key()
        if api_key:
            st.success("已读取环境变量中的 API Key")
        else:
            st.warning("未读取到 DEEPSEEK_API_KEY 或 DEEPSEEK_API_KEY_TEST")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "请输入你的问题",
        disabled=not get_deepseek_api_key(),
    )

    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    max_messages = st.session_state.history_rounds * 2
    recent_messages = st.session_state.messages[-max_messages:]
    api_messages = build_api_messages(
        system_prompt=st.session_state.system_prompt,
        chat_messages=recent_messages,
    )

    client = OpenAI(
        api_key=get_deepseek_api_key(),
        base_url=DEEPSEEK_BASE_URL,
    )

    try:
        with st.chat_message("assistant"):
            with st.spinner("正在生成回复..."):
                response = client.chat.completions.create(
                    model=st.session_state.model_name.strip() or DEFAULT_MODEL,
                    messages=api_messages,
                    temperature=float(st.session_state.temperature),
                    stream=False,
                )

                answer = response.choices[0].message.content or "模型没有返回内容。"
                st.markdown(answer)

        st.session_state.messages.append(
            {"role": "assistant", "content": answer}
        )
    except Exception as exc:
        st.error(f"调用 DeepSeek 失败：{exc}")


if __name__ == "__main__":
    main()
