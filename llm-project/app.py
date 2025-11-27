import streamlit as st
from qa_chain import ZhipuQAChain


def main():
    st.set_page_config(page_title="智谱大模型问答助手", page_icon="🦜")
    st.markdown("### 🦜🔗 智谱大模型问答助手")

    # 初始化会话状态
    if "qa_chain" not in st.session_state:
        st.session_state.qa_chain = ZhipuQAChain(
            model_name="glm-4-plus",
            temperature=0.7,
            top_k=4
        )

    if "expanded" not in st.session_state:
        st.session_state.expanded = False

    # 侧边栏配置
    with st.sidebar:
        st.title("配置")
        temperature = st.slider(
            "温度系数", 0.0, 1.0,
            st.session_state.qa_chain.temperature, 0.1
        )
        top_k = st.slider(
            "检索文档数量", 1, 10,
            st.session_state.qa_chain.top_k, 1
        )

        if st.button("清空对话历史"):
            st.session_state.qa_chain.clear_history()
            st.success("已清空对话历史")

        if st.button("截断历史记录"):
            max_len = st.number_input("保留对话轮次", 1, 10, 5)
            st.session_state.qa_chain.truncate_history(max_len)
            st.success(f"已保留最近{max_len}轮对话")

        # 更新配置
        if temperature != st.session_state.qa_chain.temperature:
            st.session_state.qa_chain.temperature = temperature
        if top_k != st.session_state.qa_chain.top_k:
            st.session_state.qa_chain.top_k = top_k

    # 显示对话历史
    messages_container = st.container(height=500)
    with messages_container:
        for i, (human_msg, ai_msg) in enumerate(st.session_state.qa_chain.chat_history):
            with st.chat_message("human"):
                st.write(human_msg)
            with st.chat_message("ai"):
                st.write(ai_msg)

    # 处理用户输入
    if prompt := st.chat_input("请输入你的问题..."):
        # 显示用户消息
        with messages_container:
            with st.chat_message("human"):
                st.write(prompt)

        # 获取并显示AI回答
        with messages_container:
            with st.chat_message("ai"):
                response = st.write_stream(
                    st.session_state.qa_chain.stream_answer(prompt)
                )


if __name__ == "__main__":
    main()