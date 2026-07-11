# Run an OMOP clinical agent using Ollama and FastMCP tools.
import argparse
import asyncio
import sys
from pathlib import Path

import mlflow
from deepagents import create_deep_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from src.config.settings import settings


# Define the OMOP MCP server file and Python executable
SERVER_PATH = str(
    Path(__file__).resolve().parents[2]
    / "src"
    / "mcp_server"
    / "server.py"
)

# Define the skills MCP server file
SKILLS_SERVER_PATH = str(
    Path(__file__).resolve().parents[2] 
    / "src" 
    / "mcp_server" 
    / "skills_server.py"
)

# Define the Python executable
PYTHON_BIN = sys.executable


# Define the instructions followed by the clinical agent
SYSTEM_PROMPT = """You are a clinical AI assistant working with synthetic patient data. You have access to tools that retrieve structured patient data and clinical protocol documents.

At the start of every patient assessment:
1. Call get_omop_reasoning_guide to load OMOP data quality rules
2. Call get_patient_summary to get demographics
3. Call the relevant clinical domain tools based on the question
4. Apply the reasoning guide rules before drawing conclusions
5. For PIFU questions, call get_skill_for_condition or get_skill for the relevant specialty

Important:
- This is synthetic data only — no real patients
- Always state which tools you called and what data you retrieved
- Apply all data quality rules from the reasoning guide before concluding
- If data is missing or null, say so explicitly and apply the relevant rule
- Keep responses concise and structured
"""

# Setup MLflow for logging
def _setup_mlflow() -> None:
    Path("mlflow_runs").mkdir(exist_ok=True)

    # Configure the MLflow tracking location
    mlflow.set_tracking_uri(
        settings.mlflow_tracking_uri
    )

    # Select or create the configured experiment
    mlflow.set_experiment(
        settings.mlflow_experiment_name
    )

# Make the LLM client
def _make_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.azure_openai_deployment,
        base_url=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        temperature=0,
    )

# Validate the patient identifier and user query
def _validate_request(person_id: int, query: str) -> tuple[int, str]:
    # Validate the patient identifier and user query
    person_id = int(person_id)
    query = query.strip()

    if person_id <= 0:
        raise ValueError("person_id must be a positive integer.")

    if not query:
        raise ValueError("query must not be empty.")

    return person_id, query

# Run the agent
async def run_agent(person_id: int, query: str) -> str:
    # Validate the patient identifier and user query
    person_id, query = _validate_request(
        person_id,
        query
    )

    # Create the LLM client
    llm = _make_llm()

    # Create the MCP client
    client = MultiServerMCPClient(
        {
            # Define the OMOP MCP server
            "omop": {
                "command": PYTHON_BIN,
                "args": [SERVER_PATH],
                "transport": "stdio",
            },
            
            # Define the skills MCP server
            "skills": {
                "command": PYTHON_BIN,
                "args": [SKILLS_SERVER_PATH],
                "transport": "stdio",
            },
        }
    )

    # Get the tools from the MCP server
    tools = await client.get_tools()

    mlflow.log_param("tools_count", len(tools))

    # Create the agent
    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=SYSTEM_PROMPT,
    )

    full_query = f"Patient person_id={person_id}. {query}"
    result = ""
    tool_calls_made = []

    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": full_query}]},
        stream_mode="values"
    ):
        # Process the chunks of the agent response
        messages = chunk.get("messages", [])

        # Process the last message in the chunk
        if messages:
            last = messages[-1]
            content = getattr(last, "content", "")
            tool_calls = getattr(last, "tool_calls", [])

            # Process the tool calls in the last message
            if tool_calls:
                for tc in tool_calls:
                    tool_calls_made.append(tc.get("name", "unknown"))
            
            # Process the content of the last message
            if content and isinstance(content, str):
                result = content

    mlflow.log_param("tool_calls_sequence", " → ".join(tool_calls_made))
    mlflow.log_param("tool_calls_count", len(tool_calls_made))
    mlflow.log_text(result, "agent_response.txt")
    mlflow.log_text(query, "query.txt")

    return result


def ask(person_id: int, query: str) -> str:
    _setup_mlflow()

    with mlflow.start_run(
        run_name=f"patient_{person_id}",
        tags={
            "person_id": str(person_id),
            "sprint": "sprint_2",
            "placement": "lancashire_teaching_hospitals",
            "backend" : "azure_openai",
            "model" : settings.azure_openai_deployment,
            "dataset" : "delphi-100k",
            "omop_version" : "5.4",
        }
    ):
        # Log the parameters
        mlflow.log_param("person_id", person_id)
        mlflow.log_param("query", query)
        mlflow.log_param("model", settings.azure_openai_deployment)

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
            "Summarise this patient's medical history, including conditions, medications, and recent visits."
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