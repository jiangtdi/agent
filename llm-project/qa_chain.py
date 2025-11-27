from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableBranch, RunnablePassthrough, Runnable
from langchain_community.vectorstores import Chroma
from zhipuai_llm import ZhipuaiLLM
from zhipuai_embedding import ZhipuAIEmbeddings


class ZhipuQAChain:
    """
    智谱API专用的带历史记录的问答链
    - model_name: 智谱模型名称，如"glm-4-plus"
    - temperature: 温度系数，控制生成随机性
    - top_k: 检索返回的前k个文档
    - persist_directory: 向量数据库持久化路径
    - chat_history: 对话历史记录
    """

    def __init__(self,
                 model_name: str = "glm-4-plus",
                 temperature: float = 0.0,
                 top_k: int = 4,
                 persist_directory: str = "chroma2",
                 chat_history: list = None):

        self.model_name = model_name
        self.temperature = temperature
        self.top_k = top_k
        self.persist_directory = persist_directory
        self.chat_history = chat_history if chat_history is not None else []
        self.vectordb = self._get_vectordb()
        self.chain = self._build_chain()

    def _get_vectordb(self) -> Chroma:
        """获取向量数据库"""
        embedding = ZhipuAIEmbeddings()
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embedding
        )

    def _build_chain(self) -> Runnable:
        """构建问答链"""
        llm = ZhipuaiLLM(model_name=self.model_name, temperature=self.temperature)

        # 问题浓缩提示
        condense_question_system_template = (
            "请根据聊天记录总结用户最近的问题，"
            "如果没有多余的聊天记录则返回用户的问题。"
        )
        condense_question_prompt = ChatPromptTemplate([
            ("system", condense_question_system_template),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

        # 检索文档逻辑
        retriever = self.vectordb.as_retriever(
            search_type="similarity",
            search_kwargs={'k': self.top_k}
        )

        retrieve_docs = RunnableBranch(
            (lambda x: not x.get("chat_history", False),
             (lambda x: x["input"]) | retriever),
            condense_question_prompt | llm | StrOutputParser() | retriever,
        )

        # 回答生成提示
        system_prompt = (
            "你是一个问答任务的助手。 "
            "请使用检索到的上下文片段回答这个问题。 "
            "如果你不知道答案就说不知道。 "
            "请使用简洁的话语回答用户。"
            "\n\n"
            "{context}"
        )
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
        ])

        # 组合链
        def combine_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs["context"])

        qa_chain = (
                RunnablePassthrough().assign(context=combine_docs)
                | qa_prompt
                | llm
                | StrOutputParser()
        )

        return RunnablePassthrough().assign(
            context=retrieve_docs
        ).assign(answer=qa_chain)

    def stream_answer(self, question: str):
        """流式获取回答"""
        if not question.strip():
            return

        # 转换历史记录格式为LangChain可识别的格式
        formatted_history = []
        for human, ai in self.chat_history:
            formatted_history.append(("human", human))
            formatted_history.append(("ai", ai))

        response = self.chain.stream({
            "input": question,
            "chat_history": formatted_history
        })

        full_answer = ""
        for res in response:
            if "answer" in res:
                full_answer += res["answer"]
                yield res["answer"]

        # 更新历史记录
        self.chat_history.append((question, full_answer))

    def clear_history(self):
        """清空对话历史"""
        self.chat_history.clear()
        return True

    def truncate_history(self, max_length: int = 5):
        """截断历史记录，保留最近的max_length轮对话"""
        if len(self.chat_history) > max_length:
            self.chat_history = self.chat_history[-max_length:]
        return self.chat_history
