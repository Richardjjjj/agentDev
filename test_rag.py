import os
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings  # Updated import
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

import openai

# Configure OpenAI - make sure to use a valid API key if needed
os.environ["OPENAI_API_KEY"] = "NULL"  # Replace with actual key or keep NULL if using local
os.environ["OPENAI_API_BASE"] = "http://192.168.10.137:31002/v1"

def query_llm(prompt, model="gpt-3.5-turbo-16k"):
    response = openai.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

def main():
    # 加载文档
    loader = TextLoader("xxl.txt")
    documents = loader.load()

    # 分割文档
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)

    try:
        # 创建向量数据库
        embeddings = OpenAIEmbeddings()
        vector_db = Chroma.from_documents(chunks, embeddings)

        # 检索相关信息
        query = "郭沛华的职位"
        docs = vector_db.similarity_search(query, k=3)

        # 结合检索结果与LLM生成回答
        context = "\n".join([doc.page_content for doc in docs])
        prompt = f"基于以下信息回答问题: {query}\n\n信息: {context}"
        response = query_llm(prompt)
        print(response)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()