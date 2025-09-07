from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.llms.openai import OpenAI
# from langchain_community.llms.openai import OpenAIChat
import os

class RAGAgent:
    def __init__(
        self,
        document_paths: list,
        embedding_model=None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        vector_store_class=FAISS,
        k: int = 2
    ):
        self.document_paths = document_paths
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.vector_store_class = vector_store_class
        self.k = k
        self.vector_store = self._load_vector_store()

    def _load_vector_store(self):
        documents = []
        for document_path in self.document_paths:
            with open(document_path, "r", encoding="utf-8") as file:
                raw_text = file.read()
            
            # use the `RecursiveCharacterTextSplitter` algorithm to split each document into chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )

            # append the chunks into the `documents` collection
            documents.extend(splitter.create_documents([raw_text]))

        # convert documents to embeddings
        return self.vector_store_class.from_documents(documents, self.embedding_model)
    

    def retrieve(self, query: str):
        docs = self.vector_store.similarity_search(query, k=self.k) # k indicates top-k matching documents to retrieve
        
        # Note we are just appending everything to the content object.
        # Issue: If the page_content is quite big it might bloat up the context window.
        # but for now it's fine.
        context = [doc.page_content for doc in docs]
        return context
    
    def generate(
        self,
        query: str, 
        retrieved_docs: list, 
        llm_model=None, 
        prompt_template=None
    ):
        context = "\n".join(retrieved_docs) # consolidate docs for context
        model = llm_model or OpenAI(temperature=0)

        prompt = prompt_template or (
            "Answer the query using the context below.\n\nContext:\n{context}\n\nQuery:\n{query}"
            "Only use information from the context. If nothing relevant is found, respond with: 'No relevant information available.'"
        )

        prompt = prompt.format(context=context, query=query)

        return model(prompt)

    def answer(
        self, 
        query: str,
        llm_model=None, 
        prompt_template=None
    ):
        retrieved_docs = self.retrieve(query=query)

        # generate answer
        generated_answer = self.generate(query=query, retrieved_docs=retrieved_docs, llm_model=llm_model, prompt_template=prompt_template)

        return generated_answer, retrieved_docs


base_path = "./datasets/Theranos"
documents = ["theranos.txt"]
documents = [os.path.join(base_path, document_name) for document_name in documents]

prompt_template = """
You are a helpful assistant. Use the context below to answer the user's query. 
Format your response strictly as a JSON object with the following structure:

{{
  "answer": "<a concise, complete answer to the user's query>",
  "citations": [
    "<relevant quoted snippet or summary from source 1>",
    "<relevant quoted snippet or summary from source 2>",
    ...
  ]
}}

Only include information that appears in the provided context. Do not make anything up.
Only respond in JSON — No explanations needed. Only use information from the context. If 
nothing relevant is found, respond with: 

{{
  "answer": "No relevant information available.",
  "citations": []
}}


Context:
{context}

Query:
{query}
"""
agent = RAGAgent(documents)
# print (agent.vector_store)


# after retreiving the docs, we will feed them as context to generate the final answer
# query = "Who's the CEO of theranos?"
# query = "What is theranos flagship product?"
query = "What is the TheraCloud health portal? Explain in simple words."


generated_answer, retrieved_docs = agent.answer(query, prompt_template=prompt_template)

print (generated_answer)

