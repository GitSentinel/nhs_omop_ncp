import asyncio
from src.mcp_server.server import mcp

async def check():
    tools = await mcp.list_tools()
    print(f'Tools registered: {len(tools)}')
    for tool in tools:
        print(f'  {tool.name}')

asyncio.run(check())
