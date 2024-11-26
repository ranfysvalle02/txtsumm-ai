import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
from langchain_text_splitters import CharacterTextSplitter

class TextSummarizer:
    def __init__(self, llm=None):
        if llm is None:
            # requires LLM, throw exception
            raise Exception("LLM is not initialized")
        self.llm = llm

        self.map_prompt = ChatPromptTemplate.from_messages([
            ("human", "Write a concise summary of the following:\n\n{context}")
        ])

        self.reduce_prompt = ChatPromptTemplate.from_messages([
            ("human", "Combine these summaries into a final summary:\n\n{summaries}")
        ])
        # StrOutputParser: OutputParser that parses LLMResult into the top likely string.
        # https://python.langchain.com/api_reference/core/output_parsers/langchain_core.output_parsers.string.StrOutputParser.html
        self.map_chain = self.map_prompt | self.llm | StrOutputParser()
        self.reduce_chain = self.reduce_prompt | self.llm | StrOutputParser()

    def summarize_text(self, text):
        text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=1000, chunk_overlap=0
        )
        docs = text_splitter.create_documents([text])

        summaries = [self.map_chain.invoke({"context": doc.page_content}) for doc in docs]
        final_summary = self.reduce_chain.invoke({"summaries": "\n\n".join(summaries)})

        return final_summary

os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""
my_llm = AzureChatOpenAI(
    openai_api_version="2024-10-01",
    azure_deployment="gpt-4o-mini",
    temperature=0.0,
    max_tokens=150
)
# Example usage
summarizer = TextSummarizer(llm=my_llm)
long_text = "hello world"
summary = summarizer.summarize_text(long_text)
print("Final Summary:")
print(summary)
