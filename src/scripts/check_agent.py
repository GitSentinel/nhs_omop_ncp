from src.agents.omop_agent import ask, run_agent
print('omop_agent imports OK')
print('SERVER_PATH resolves to:')
from pathlib import Path
import sys
p = str(Path('src/mcp_server/server.py').resolve())
print(' ', p)
