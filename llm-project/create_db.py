import os
import re
import logging
from datetime import datetime
from dotenv import load_dotenv, find_dotenv

# -------------------------- 1. 环境配置与依赖导入 --------------------------
# 加载环境变量（包含ZHIPUAI_API_KEY）
_ = load_dotenv(find_dotenv())

# 创建日志目录
LOG_DIR = "log"
os.makedirs(LOG_DIR, exist_ok=True)  # 确保日志目录存在

# 生成带时间戳的日志文件名（格式：年-月-日_时-分-秒.log）
current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
log_filename = os.path.join(LOG_DIR, f"{current_time}.log")

# 配置日志（同时输出到控制台和带时间戳的文件）
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler(log_filename),  # 日志保存到log目录下的时间戳文件
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)

# 配置代理（如需）
# os.environ['HTTPS_PROXY'] = 'http://127.0.0.1:7890'
# os.environ["HTTP_PROXY"] = 'http://127.0.0.1:7890'

# 导入文档加载器（新增txt和docx加载器）
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredMarkdownLoader,
    UnstructuredFileLoader,
    UnstructuredWordDocumentLoader
)

# 导入文本分割器
from langchain_text_splitters import RecursiveCharacterTextSplitter

# 导入嵌入模型（智谱）
from zhipuai_embedding import ZhipuAIEmbeddings

# 导入向量数据库
from langchain_community.vectorstores import Chroma

# -------------------------- 2. 路径配置（可根据自己需求修改） --------------------------
# 原始知识库文档路径（放你的PDF/Markdown/TXT/DOCX文件）
KNOWLEDGE_DB_PATH = "data_base/knowledge_db1"
# 向量数据库持久化路径
VECTOR_DB_PATH = "data_base/vector_db/chroma2"


# -------------------------- 3. 加载多类型文档（新增txt和docx支持） --------------------------
def load_documents(folder_path):
    """加载指定目录下的所有PDF、Markdown、TXT和DOCX文件"""
    file_paths = []
    # 遍历目录获取所有文件路径，支持的文件类型扩展为pdf、md、txt、docx
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_ext = file.split('.')[-1].lower()
            if file_ext in ['pdf', 'md', 'txt', 'docx']:  # 新增txt和docx类型
                file_paths.append(os.path.join(root, file))

    # 初始化加载器并加载文档
    loaders = []
    for file_path in file_paths:
        file_ext = file_path.split('.')[-1].lower()
        try:
            if file_ext == 'pdf':
                loaders.append(PyMuPDFLoader(file_path))
                logger.info(f"添加PDF加载器：{file_path}")
            elif file_ext == 'md':
                loaders.append(UnstructuredMarkdownLoader(file_path))
                logger.info(f"添加Markdown加载器：{file_path}")
            elif file_ext == 'txt':  # 新增TXT文件处理
                loaders.append(UnstructuredFileLoader(file_path))
                logger.info(f"添加TXT加载器：{file_path}")
            elif file_ext == 'docx':  # 新增DOCX文件处理
                loaders.append(UnstructuredWordDocumentLoader(file_path))
                logger.info(f"添加DOCX加载器：{file_path}")
        except Exception as e:
            logger.error(f"创建加载器失败 {file_path}：{str(e)}")

    # 加载所有文档
    docs = []
    for loader in loaders:
        try:
            docs.extend(loader.load())
            logger.info(f"已加载文件：{loader.file_path}")
        except Exception as e:
            logger.error(f"加载文件失败 {loader.file_path}：{str(e)}")

    return docs


# -------------------------- 4. 文档预处理与分块 --------------------------
def process_and_split_docs(docs):
    """文档预处理（清理）+ 分块"""
    # 清理文档内容（去除多余换行、特殊符号）
    for doc in docs:
        # 清理中英文混排时的多余换行
        doc.page_content = re.sub(r'[^\u4e00-\u9fff](\n)[^\u4e00-\u9fff]',
                                 lambda m: m.group(0).replace('\n', ''),
                                 doc.page_content)
        doc.page_content = doc.page_content.replace('•', '').replace('  ', ' ')

    # 分块配置
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # 每个块的字符数
        chunk_overlap=50,  # 块之间的重叠字符数
        length_function=len  # 长度计算函数
    )

    # 执行分块
    split_docs = text_splitter.split_documents(docs)
    logger.info(f"文档分块完成，共生成 {len(split_docs)} 个文本块")
    return split_docs


# -------------------------- 5. 生成向量并存储到数据库（保持智谱嵌入模型） --------------------------
def build_vector_db(split_docs, persist_dir):
    """将分块后的文档转为向量并存储到Chroma，使用智谱嵌入模型"""
    # 初始化智谱嵌入模型（保持不变）
    embedding = ZhipuAIEmbeddings()

    # 创建并持久化向量库
    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embedding,
        persist_directory=persist_dir
    )

    # 强制持久化
    vectordb.persist()
    logger.info(f"向量数据库构建完成！存储路径：{persist_dir}")
    return vectordb


# -------------------------- 主函数：一键生成数据库 --------------------------
if __name__ == "__main__":
    try:
        # 1. 加载文档（支持多种类型）
        logger.info("开始加载知识库文档...")
        raw_docs = load_documents(KNOWLEDGE_DB_PATH)
        if not raw_docs:
            logger.warning("未找到任何PDF/Markdown/TXT/DOCX文档，请检查KNOWLEDGE_DB_PATH路径！")
            exit(1)

        # 2. 文档分块
        split_docs = process_and_split_docs(raw_docs)

        # 3. 构建向量库（使用智谱嵌入）
        vectordb = build_vector_db(split_docs, VECTOR_DB_PATH)

        logger.info("新数据库生成成功！可直接用于RAG检索")

    except Exception as e:
        logger.error(f"生成数据库失败：{str(e)}", exc_info=True)