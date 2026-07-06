# Multi-Agent RAG System

This project is an intelligent assistant that combines multiple AI agents to answer questions more accurately. Depending on the query, the system decides whether to retrieve information from a database or from the web.

## How it works

When a user asks a question, a supervisor agent analyzes the intent and chooses the best path:

- If the query requires structured data, it is sent to an SQL database.
- If it requires general or up-to-date information, it triggers a web search.

The selected agent processes the request and generates a response. All context is stored in memory to improve future interactions.

## Features

- Multi-agent architecture  
- Intelligent query routing  
- Natural language to SQL translation  
- Web search integration  
- Session-based conversational memory  
- Fast responses powered by Groq  
- SQLite database (Chinook Database)  
- Context-aware interactions  

## Tech Stack

- Python  
- LangGraph  
- LangChain  
- Groq (Llama 3.3 70B)  
- Tavily API  
- SQLite  
- Chinook Database  

## Workflow

1. The user sends a question.  
2. A supervisor agent decides whether to use the database or the web.  
3. The appropriate agent processes the query.  
4. A final response is generated.  
5. The conversation is stored to maintain context.  

## Use Cases

- Natural language database querying  
- Multi-agent AI systems  
- Retrieval-Augmented Generation (RAG) applications  
- Conversational assistants  
- Context-aware question answering systems  
