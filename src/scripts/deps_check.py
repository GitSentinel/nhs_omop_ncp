import deepagents
import langchain_mcp_adapters
import langgraph

from langchain_mcp_adapters.client import MultiServerMCPClient
from deepagents import create_deep_agent

print('deepagents             :', deepagents.__version__)
print('langchain-mcp-adapters : installed OK')
print('langgraph              : installed OK')
print('MultiServerMCPClient   : imported OK')
print('create_deep_agent      : imported OK')
print()
print('All deps OK')
