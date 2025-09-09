from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from deepeval.test_case import LLMTestCaseParams, LLMTestCase
from langchain.llms.openai import OpenAI
from deepeval.tracing import observe, update_current_span
from metrics import (
    answer_correctness,
    citation_accuracy,
    recall,
    relevancy,
    precision
)
from deepeval.tracing import observe
import os


class RAGAgent:
    def __init__(
        self,
        document_paths: list,
        embedding_model=None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        vector_store_class=FAISS,
        k: int = 2,
        prompt_template: str = None
    ):
        self.document_paths = document_paths
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = embedding_model or OpenAIEmbeddings()
        self.vector_store_class = vector_store_class
        self.k = k
        self.vector_store = self._load_vector_store()
        self.prompt_template = prompt_template or (
            "Answer the query using the context below.\n\nContext:\n{context}\n\nQuery:\n{query}"
            "Only use information from the context. If nothing relevant is found, respond with: 'No relevant information available.'"
        )


    def _load_vector_store(self):
        print (f'Loading and chunking docs into vectordb...')
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
        print (f'Docs loaded successfully.')
        return self.vector_store_class.from_documents(documents, self.embedding_model)
    
    @observe(metrics=[recall, relevancy, precision], name="Retriever")
    def retrieve(self, query: str):
        docs = self.vector_store.similarity_search(query, k=self.k) # k indicates top-k matching documents to retrieve
        
        # Note we are just appending everything to the content object.
        # Issue: If the page_content is quite big it might bloat up the context window.
        # but for now it's fine.
        context = [doc.page_content for doc in docs]
        update_current_span(
            test_case=LLMTestCase(
                input=query,
                retrieval_context=context
            )
        )
        return context
    
    @observe(metrics=[answer_correctness, citation_accuracy], name="Generator")
    def generate(
        self,
        query: str,
        retrieved_docs: list, 
        llm_model=None, 
    ):
        context = "\n".join(retrieved_docs) # consolidate docs for context
        model = llm_model or OpenAI(temperature=0)

        prompt = self.prompt_template.format(context=context, query=query)

        answer = model(prompt)

        update_current_span(
            test_case=LLMTestCase(
                input=query,
                actual_output=answer,
                retrieval_context=retrieved_docs
            )
        )

        return answer

    @observe(type="agent")
    def answer(
        self, 
        query: str,
        llm_model=None, 
    ):
        retrieved_docs = self.retrieve(query=query)

        # generate answer
        generated_answer = self.generate(query=query, retrieved_docs=retrieved_docs, llm_model=llm_model)

        # update_current_span(
        #     test_case=LLMTestCase(
        #         input=query,
        #         actual_output=generated_answer,
        #         retrieval_context=retrieved_docs
        #     ),
        #     output=generated_answer
        # )

        return generated_answer, retrieved_docs


if __name__ == "__main__":
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
    agent = RAGAgent(documents, prompt_template=prompt_template)
    # print (agent.vector_store)


    # after retreiving the docs, we will feed them as context to generate the final answer
    # query = "Who's the CEO of theranos?"
    # query = "What is theranos flagship product?"
    query = "What is the TheraCloud health portal? Explain in simple words."


    generated_answer, retrieved_docs = agent.answer(query)

    print (generated_answer)

