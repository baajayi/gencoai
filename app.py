import os
import gradio as gr
from dotenv import load_dotenv, find_dotenv

from llama_index.readers.youtube_transcript import YoutubeTranscriptReader
from langchain_community.vectorstores import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough

_ = load_dotenv(find_dotenv())
google_api_key = os.getenv("GOOGLE_API_KEY")
docs = []
loader = YoutubeTranscriptReader()
docs.extend(loader.load_data(ytlinks = ["https://www.youtube.com/watch?v=oSLZE0rm2Pg", 
                                        "https://www.youtube.com/watch?v=A7oiCFnlLr4", 
                                        "https://www.youtube.com/watch?v=K7x4VsjguH8", 
                                        "https://www.youtube.com/watch?v=nD8gGZzHZ7Y", 
                                        "https://www.youtube.com/watch?v=1p5pbCyvKvg"]))
# print(docs)

context = "\n\n".join(doc.text for doc in docs)
# print(context)

split_text = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = split_text.split_text(context)
# print(len(texts))
# for i in texts:
#     print(i,"\n\n")

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=google_api_key)
vector_index = Chroma.from_texts(texts, embeddings).as_retriever()

# print(len(vector_index.get_relevant_documents("Bible")))

# question = input("What is your question?")

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=google_api_key,
                 temperature=0.7, top_p=0.85)



# Prompt template to query Gemini
llm_prompt_template = """You are an intelligent assistant for question-answering tasks.
Use the following context, take a deep breath and lets think step by step to answer the question.
Keep the answer concise.\n
Question: {question} \nContext: {context} \nAnswer:"""

llm_prompt = PromptTemplate.from_template(llm_prompt_template)

# print(llm_prompt)

rag_chain = (
    {"context": vector_index, "question": RunnablePassthrough() }
    | llm_prompt
    | llm
    | StrOutputParser()
)
def answer_question(question):
    answer = rag_chain.invoke(question)
    return answer

iface = gr.Interface(
    fn=answer_question, 
    inputs=gr.Textbox(lines=2, placeholder="Enter your question here..."), 
    outputs="text",
    title="VIVE Church Q&A",
    description="This app answers your questions based on the content of selected YouTube video transcripts."
)

if __name__ == "__main__":
    iface.launch(share=True)