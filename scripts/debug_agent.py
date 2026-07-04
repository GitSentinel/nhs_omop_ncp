import asyncio
import sys
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama
from deepagents import create_deep_agent
from src.config.settings import settings

SERVER_PATH = str(Path('src/mcp_server/server.py').resolve())
PYTHON_BIN = sys.executable

async def run():
    llm = ChatOllama(model=settings.ollama_model, base_url=settings.ollama_base_url, temperature=0)
    client = MultiServerMCPClient({
        'omop': {'command': PYTHON_BIN, 'args': [SERVER_PATH], 'transport': 'stdio'}
    })
    tools = await client.get_tools()
    print(f'Tools loaded: {len(tools)}')

    agent = create_deep_agent(model=llm, tools=tools)

    print('Streaming agent response:')
    print('=' * 60)
    async for chunk in agent.astream(
        {'messages': [{'role': 'user', 'content': 'Patient person_id=17247. What conditions does this patient have? Call get_patient_conditions with person_id=17247.'}]},
        stream_mode='values'
    ):
        messages = chunk.get('messages', [])
        if messages:
            last = messages[-1]
            content = getattr(last, 'content', '')
            if content and isinstance(content, str):
                print(content[:500])
                print('---')

asyncio.run(run())
