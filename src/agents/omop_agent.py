# Run an OMOP clinical agent using Ollama and FastMCP tools.
import argparse
import asyncio
import sys
from pathlib import Path

from deepagents import create_deep_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_ollama import ChatOllama

from src.config.settings import settings


# Define the MCP server file and Python executable
SERVER_PATH = str(
    Path(__file__).resolve().parents[2]
    / "src"
    / "mcp_server"
    / "server.py"
)
PYTHON_BIN = sys.executable


# Define the instructions followed by the clinical agent
SYSTEM_PROMPT = """
You are a clinical AI assistant working with synthetic OMOP CDM v5.4
patient data at Lancashire Teaching Hospitals NHS Foundation Trust.

You can retrieve structured demographics, conditions, medications,
visits, measurements, observations, clinical notes, and procedures.

When answering a patient question:
1. Call get_patient_summary first.
2. Call the relevant clinical-domain tools.
3. Analyse only the retrieved data.
4. Provide a concise and structured response.
5. State which tools were called.
6. Report missing or null information clearly.
7. State that the data are synthetic.
"""


def _validate_request(person_id: int, query: str) -> tuple[int, str]:
    # Validate the patient identifier and query.
    person_id = int(person_id)
    query = query.strip()

    if person_id <= 0:
        raise ValueError("person_id must be a positive integer.")

    if not query:
        raise ValueError("query must not be empty.")

    return person_id, query


def _extract_response(result: dict) -> str:
    # Extract the final assistant response from the agent result.
    messages = result.get("messages", [])

    for message in reversed(messages):
        content = getattr(message, "content", None)

        if content:
            return str(content)

    return str(result)


async def run_agent(person_id: int, query: str) -> str:
    llm = ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=0,
    )

    client = MultiServerMCPClient(
        {
            "omop": {
                "command": PYTHON_BIN,
                "args": [SERVER_PATH],
                "transport": "stdio",
            }
        }
    )

    tools = await client.get_tools()

    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    full_query = f"Patient person_id={person_id}. {query}"

    result = await agent.ainvoke({
        "messages": [{"role": "user", "content": full_query}]
    })

    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content") and msg.content:
            return msg.content

    return str(result)


def ask(person_id: int, query: str) -> str:
    # Run the asynchronous agent from synchronous Python code.
    return asyncio.run(
        run_agent(person_id, query)
    )


def main() -> None:
    # Run the OMOP agent from the command line.
    parser = argparse.ArgumentParser(
        description="OMOP clinical agent command-line interface"
    )

    parser.add_argument(
        "--pid",
        type=int,
        default=17247,
        help="OMOP patient identifier"
    )

    parser.add_argument(
        "--query",
        type=str,
        default=(
            "Summarise this patient's medical history, including "
            "conditions, medications, and recent visits."
        ),
        help="Question to ask about the patient"
    )

    args = parser.parse_args()

    print(f"\nRunning OMOP agent for person_id={args.pid}")
    print(f"Query: {args.query}")
    print("=" * 60)

    response = ask(args.pid, args.query)
    print(response)


# Run the command-line interface when executed directly
if __name__ == "__main__":
    main()