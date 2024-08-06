
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain.prompts import PromptTemplate
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter




def get_pdf_text(pdf_docs):
    documents=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            documents+= page.extract_text()
    return  documents

# def get_pdf_text(pdf_path):
#     loader=PyPDFLoader(pdf_path)
#     documents=loader.load()
#     return documents


# def get_text_chunk(documents):
#     text_splitter=CharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
#     docs = text_splitter.split_documents(documents=documents)
#     return docs


def get_text_chunk(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks





def get_vector_store(docs,text_chunk):
    embedding_model_name="sentence-transformers/all-mpnet-base=v2"
    model_kwargs={"device":"cuda"}
    embedding=HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs
    )
    vector_store=FAISS.from_documents(docs,embedding)
    vector_store.save_local("faiss_index")



def get_conversational_chain(retriever):

    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """
    llm = Ollama(model='llama3.1',
                 temperature=0.3)
    
    prompt = PromptTemplate(template=prompt_template,input_variables=["Context","Question"])
    chain= RetrievalQA.from_chain_type(llm=llm,chain_type="stuff",retriever=retriever)
    return chain



def user_input(user_question):
    embedding_model_name="sentence-transformers/all-mpnet-base=v2"
    model_kwargs={"device":"cuda"}
    embedding=HuggingFaceEmbeddings(
        model_name=embedding_model_name,
        model_kwargs=model_kwargs
    )

    new_db=FAISS.load_local("faiss_index",embedding,allow_dangerous_deserialization=True)
    
    docs= new_db.similarity_search(user_question)

    chain =get_conversational_chain()

    response = chain(
        {
            "input_documents":docs,
            "question": user_question
        },return_only_outputs=True
    )

    print(response)
    st.write("Reply: ", response["output_text"])



def main():
    # st.set_page_config("Chat PDF")
    st.header("Question-Ansering with PDF ")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)
    

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunk(raw_text)
                get_vector_store(raw_text,text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()

