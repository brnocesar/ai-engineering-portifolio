# ai-engineering-portifolio

## Contents

- [Travel Assistant Agent](travel-assistant-agent/README.md)

## Travel Assistant Agent

**Key-words:** Generative AI, AI Agents, LangChain, LangGraph, Prompt Chaining, Conversational Memory, Conditional Routing, Retrieval-Augmented Generation (RAG), FAISS, Vector Search, Python, OpenAI API, OpenAI Embeddings.

An experimental collection of implementations for a travel assistant, covering destination recommendations, conversational memory, specialized routing, and document-grounded answers about travel benefits.

- **Recommendations:** LangChain chains suggest a destination based on the user's interests, then provide restaurant and cultural activity recommendations.
- **Conversational assistant:** a travel guide for Brazilian destinations that keeps conversation history in memory for each session.
- **Specialized routing:** a LangGraph workflow classifies the request and routes it to a beach or mountain travel specialist.
- **Document-based answers:** a RAG workflow retrieves relevant passages from travel-benefit documents using OpenAI embeddings and FAISS, then generates a context-grounded response.

Development and experimentation are documented in the notebooks; the corresponding Python scripts present concise versions of the final implementations.

[Click here to access the project](travel-assistant-agent/README.md)
