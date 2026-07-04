from langchain_ollama import ChatOllama

llm = ChatOllama(model='llama3', temperature=0)
response = llm.invoke('Reply with exactly this phrase and nothing else: Ollama OK')

print('Response:', response.content)
print('Ollama + LangChain OK')
