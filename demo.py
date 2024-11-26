import os
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import AzureChatOpenAI
# from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import CharacterTextSplitter
from multiprocessing import Pool, current_process
from functools import partial
from criticalvectors import CriticalVectors

# Global configuration for Azure OpenAI
AZURE_API_VERSION = ""
AZURE_DEPLOYMENT = "gpt-4o-mini"
# Set environment variables (ensure these are set appropriately)
os.environ["AZURE_OPENAI_API_KEY"] = ""
os.environ["AZURE_OPENAI_ENDPOINT"] = ""

def init_critical_vectors():
    """
    Initialize the critical vectors selector.
    """
    return CriticalVectors(
        strategy='kmeans',
        num_clusters='auto',
        chunk_size=10000,
        chunks_per_cluster=1,  # Set the desired number of chunks per cluster here
        split_method='sentences',
        max_tokens_per_chunk=3000,  # Adjust as needed
        use_faiss=True  # Enable FAISS if desired
    )
def initialize_llm():
    """
    Initialize the preferred LLM instance.
    This function is called within each worker process.
    """
    return AzureChatOpenAI( # or OllamaLLM or whatever you prefer
        openai_api_version=AZURE_API_VERSION,
        azure_deployment=AZURE_DEPLOYMENT,
        temperature=0.0,
        max_tokens=150
    )

def worker_map_func(doc_content, map_prompt_template):
    """
    Worker function to generate a summary for a single document.
    This function is picklable and initializes its own LLM instance.
    """
    try:
        # Initialize LLM within the worker
        llm = initialize_llm()
        
        # Create the map chain
        map_chain = map_prompt_template | llm | StrOutputParser()
        
        # Invoke the chain with the document content
        summary = map_chain.invoke({"context": doc_content})
        return summary
    except Exception as e:
        print(f"Error in process {current_process().name}: {e}")
        return ""

class TextSummAI:
    def __init__(self, strategy='map_reduce', parallelization=True):
        self.allowed_strategies = ['map_reduce','none','critical_vectors']
        if strategy not in self.allowed_strategies:
            raise Exception(f"Strategy {strategy} is not allowed. Allowed strategies: {self.allowed_strategies}")
        self.strategy = strategy
        if not isinstance(parallelization, bool):
            raise ValueError("parallelization must be a boolean value")
        self.parallelization = parallelization

        # Define map and reduce prompts
        self.map_prompt = ChatPromptTemplate.from_messages([
            ("human", "Write a concise summary of the following:\n\n{context}")
        ])

        self.reduce_prompt = ChatPromptTemplate.from_messages([
            ("human", "Combine these summaries into a final summary:\n\n{summaries}")
        ])

        # Initialize reduce chain (single process)
        self.reduce_llm = initialize_llm()
        self.reduce_chain = self.reduce_prompt | self.reduce_llm | StrOutputParser()

        # Initialize critical vectors selector (single process)
        if self.strategy == 'critical_vectors':
            self.critical_vectors = init_critical_vectors()

        print(f"TextSummAI initialized with strategy: {self.strategy}, parallelization: {self.parallelization}")

    def summarize_text(self, text, chunk_size=1000, chunk_overlap=0):
        if self.strategy == 'map_reduce':
            # Split the text into chunks
            text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
                chunk_size=chunk_size, chunk_overlap=chunk_overlap
            )
            docs = text_splitter.create_documents([text])

            # Prepare the map function with the map_prompt
            map_func = partial(worker_map_func, map_prompt_template=self.map_prompt)

            # Generate summaries using parallel or sequential processing
            if self.parallelization:
                with Pool() as p:
                    summaries = p.map(map_func, [doc.page_content for doc in docs])
            else:
                summaries = [map_func(doc.page_content) for doc in docs]

            # Combine the summaries into a final summary
            final_summary = self.reduce_chain.invoke({"summaries": "\n\n".join(summaries)})

            return final_summary
        elif self.strategy == 'none':
            # Return the original text, summarized with no chunking
            # Prepare the map function with the map_prompt
            map_func = partial(worker_map_func, map_prompt_template=self.map_prompt)
            return map_func(str(text))
        elif self.strategy == 'critical_vectors':
            # Use the critical vectors selector
            if self.critical_vectors is None:
                raise Exception("CriticalVectors selector is not initialized.")
            relevant_chunks, first_part, last_part = self.critical_vectors.get_relevant_chunks(str(text))
            map_func = partial(worker_map_func, map_prompt_template=self.map_prompt)
            return map_func(f"""
[first part]
{first_part}
[/first part]

[context]
{str(relevant_chunks)}
[/context]

[last part]
{last_part}
[/last part]
""")

if __name__ == "__main__":
    # Ensure that the multiprocessing code runs only when the script is executed directly

    # Initialize the summarizer with parallelization enabled
    summarizer_parallel = TextSummAI(strategy='map_reduce', parallelization=True)

    # Initialize the summarizer with parallelization disabled (sequential processing)
    summarizer_sequential = TextSummAI(strategy='map_reduce', parallelization=False)
    
    # Initialize the summarizer with no strategy
    summarizer_none = TextSummAI(strategy='none')

    # Example with CriticalVectors
    summarizer_cv = TextSummAI(strategy='critical_vectors')

    # load text file as string variable
    long_text = f"""
    hello world
    """

    # Using parallel processing
    summary_parallel = summarizer_cv.summarize_text(long_text)
    print("Final Summary:")
    print(summary_parallel)
