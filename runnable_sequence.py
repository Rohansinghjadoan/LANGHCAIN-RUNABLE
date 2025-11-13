from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

load_dotenv()

llm1 = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation"
)
model1 = ChatHuggingFace(llm=llm1)

# FIX 1: Correct parameter name
promt1 = PromptTemplate(
    template='Write a joke about {topic}',
    input_variables=['topic']
)

# FIX 2: create parser instance
parser = StrOutputParser()

prompt2 = PromptTemplate(
    template='Explain the following joke - {text}',
    input_variables=['text']
)

# FIX 3: Proper sequence chaining
chain = RunnableSequence(
    promt1,
    model1,
    parser,
    prompt2,
    model1,
    parser
)

print(chain.invoke({'topic': 'AI'}))
