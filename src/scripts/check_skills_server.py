import asyncio
import sys
from pathlib import Path
from langchain_mcp_adapters.client import MultiServerMCPClient

SKILLS_PATH = str(Path('src/mcp_server/skills_server.py').resolve())
PYTHON_BIN  = sys.executable

async def check():
    client = MultiServerMCPClient({
        'skills': {
            'command': PYTHON_BIN,
            'args': [SKILLS_PATH],
            'transport': 'stdio',
        }
    })
    tools = await client.get_tools()
    print(f'Skills tools registered: {len(tools)}')
    for t in tools:
        print(f'  {t.name}')

asyncio.run(check())
