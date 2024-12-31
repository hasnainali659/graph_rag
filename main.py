import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.graph_qa.cypher import GraphCypherQAChain
from langchain_core.runnables import (
    RunnableBranch,
    RunnableLambda,
    RunnableParallel,
    RunnablePassthrough,
)
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import Tuple, List
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.graphs import Neo4jGraph
from langchain_openai import ChatOpenAI
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_community.vectorstores import Neo4jVector
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores.neo4j_vector import remove_lucene_chars
from prompts import _template, template

load_dotenv()

class Entities(BaseModel):
    """Identifying information about entities."""

    names: List[str] = Field(
        ...,
        description="All the person, organization, or business entities that "
        "appear in the text",
    )

def load_docs(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
    docs = text_splitter.split_documents(pages)
    return docs

def add_graph_documents(docs, graph, llm):
    
    llm_transformer = LLMGraphTransformer(llm=llm)

    graph_documents = llm_transformer.convert_to_graph_documents(docs)

    graph.add_graph_documents(
    graph_documents, 
    baseEntityLabel=True, 
    include_source=True
    )
    
def generate_full_text_query(input: str) -> str:
    """
    Generate a full-text search query for a given input string.

    This function constructs a query string suitable for a full-text search.
    It processes the input string by splitting it into words and appending a
    similarity threshold (~2 changed characters) to each word, then combines
    them using the AND operator. Useful for mapping entities from user questions
    to database values, and allows for some misspelings.
    """
    full_text_query = ""
    words = [el for el in remove_lucene_chars(input).split() if el]
    for word in words[:-1]:
        full_text_query += f" {word}~2 AND"
    full_text_query += f" {words[-1]}~2"
    return full_text_query.strip()

# Fulltext index query
def structured_retriever(question, graph, entity_chain):
    """
    Collects the neighborhood of entities mentioned
    in the question
    """
    result = ""
    entities = entity_chain.invoke({"question": question})
    for entity in entities.names:
        response = graph.query(
            """CALL db.index.fulltext.queryNodes('entity', $query, {limit:2})
            YIELD node,score
            CALL {
              WITH node
              MATCH (node)-[r:!MENTIONS]->(neighbor)
              RETURN node.id + ' - ' + type(r) + ' -> ' + neighbor.id AS output
              UNION ALL
              WITH node
              MATCH (node)<-[r:!MENTIONS]-(neighbor)
              RETURN neighbor.id + ' - ' + type(r) + ' -> ' +  node.id AS output
            }
            RETURN output LIMIT 50
            """,
            {"query": generate_full_text_query(entity)},
        )
        result += "\n".join([el['output'] for el in response])
    return result

def retriever(question, graph, entity_chain, vector_index):
    print(f"Search query: {question}")
    structured_data = structured_retriever(question, graph, entity_chain)
    unstructured_data = [el.page_content for el in vector_index.similarity_search(question)]
    final_data = f"""Structured data:
    {structured_data}
    Unstructured data:
    {"#Document ". join(unstructured_data)}
        """
    return final_data

def _format_chat_history(chat_history: List[Tuple[str, str]]) -> List:
    buffer = []
    for human, ai in chat_history:
        buffer.append(HumanMessage(content=human))
        buffer.append(AIMessage(content=ai))
    return buffer
    
def main():
    
    docs = load_docs('docs/Evan Patel.pdf')
    graph = Neo4jGraph()
    llm = ChatOpenAI(model_name='gpt-4o', temperature=0)
    add_graph_documents(docs, graph, llm)
    
    vector_index = Neo4jVector.from_existing_graph(
    OpenAIEmbeddings(),
    search_type="hybrid",
    node_label="Document",
    text_node_properties=["text"],
    embedding_node_property="embedding"
    )
    
    graph.query("CREATE FULLTEXT INDEX entity IF NOT EXISTS FOR (e:__Entity__) ON EACH [e.id]")
    
    prompt_1 = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are extracting organization and person entities from the text.",
        ),
        (
            "human",
            "Use the given format to extract information from the following "
            "input: {question}",
        ),
    ]
    )

    entity_chain = prompt_1 | llm.with_structured_output(Entities)
    
    CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(_template)
    
    _search_query_chain = CONDENSE_QUESTION_PROMPT | llm | StrOutputParser()
    
    prompt_2 = ChatPromptTemplate.from_template(template)
    
    chain = prompt_2 | llm | StrOutputParser()
    

    
    chat_history = []
    
    while True:
        question = input("Enter your question: ")
        
        response_1 = _search_query_chain.invoke({"question": question, "chat_history": chat_history})
        
        retriever_response = retriever(response_1, graph, entity_chain, vector_index)
    
        print(retriever_response)
    
        response = chain.invoke({"context": retriever_response, "question": question})
        
        print(response)
    
if __name__ == "__main__":
    main()
