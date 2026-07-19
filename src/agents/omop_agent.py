import argparse
import asyncio
from pathlib import Path

import mlflow
from deepagents import create_deep_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from config import (
    PYTHON_BIN,
    OMOP_SERVER_PATH,
    SKILLS_SERVER_PATH,
    AGENT_SYSTEM_PROMPT,
    MLFLOW_EXPERIMENT_AGENT,
)

from src.config.settings import settings


def setup_mlflow() -> None:
    # MLflow directory setup
    Path("mlflow_runs").mkdir(exist_ok=True)

    # MLflow experiment setup
    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_AGENT)


def make_llm() -> ChatOpenAI:
    # Azure OpenAI model setup
    return ChatOpenAI(
        model=settings.azure_openai_deployment,
        base_url=settings.azure_openai_endpoint,
        api_key=settings.azure_openai_api_key,
        temperature=0,
    )


def make_mcp_client() -> MultiServerMCPClient:
    # MCP server configuration
    return MultiServerMCPClient(
        {
            "omop": {
                "command": PYTHON_BIN,
                "args": [str(OMOP_SERVER_PATH)],
                "transport": "stdio",
            },
            "skills": {
                "command": PYTHON_BIN,
                "args": [str(SKILLS_SERVER_PATH)],
                "transport": "stdio",
            },
        }
    )


async def run_agent(person_id: int, query: str) -> str:
    # Model and Tool Setup
    llm = make_llm()
    client = make_mcp_client()
    tools = await client.get_tools()

    # Tool metadata logging
    tool_names = [tool.name for tool in tools]

    mlflow.log_param("tools_count", len(tool_names))
    mlflow.log_text("\n".join(tool_names), "available_tools.txt")

    # Agent creation
    agent = create_deep_agent(
        model=llm,
        tools=tools,
        system_prompt=AGENT_SYSTEM_PROMPT,
    )

    # Query construction
    full_query = f"Patient person_id={person_id}. {query}"

    final_response = ""
    tool_calls_made = []

    # Agent streaming
    async for chunk in agent.astream(
        {"messages": [{"role": "user", "content": full_query}]},
        stream_mode="values",
    ):
        messages = chunk.get("messages", [])

        if not messages:
            continue

        last_message = messages[-1]
        content = getattr(last_message, "content", "")
        tool_calls = getattr(last_message, "tool_calls", [])

        if tool_calls:
            tool_calls_made.extend(
                tool_call.get("name", "unknown")
                for tool_call in tool_calls
            )

        if content and isinstance(content, str):
            final_response = content

    # Run output logging
    mlflow.log_param("tool_calls_count", len(tool_calls_made))
    mlflow.log_text(" -> ".join(tool_calls_made), "tool_calls_sequence.txt")
    mlflow.log_text(query, "query.txt")
    mlflow.log_text(final_response, "agent_response.txt")

    return final_response


def ask(person_id: int, query: str) -> str:
    # MLflow setup
    setup_mlflow()

    # Tracked agent run
    with mlflow.start_run(
        run_name=f"patient_{person_id}",
        tags={
            "person_id": str(person_id),
            "sprint": "sprint_2",
            "placement": "lancashire_teaching_hospitals",
            "backend": "azure_openai",
            "model": settings.azure_openai_deployment,
            "dataset": "delphi-100k",
            "omop_version": "5.4",
        },
    ):
        # Run metadata logging
        mlflow.log_param("person_id", person_id)
        mlflow.log_param("model", settings.azure_openai_deployment)
        mlflow.log_text(query, "input_query.txt")

        return asyncio.run(run_agent(person_id, query))


def parse_args() -> argparse.Namespace:
    # CLI argument setup
    parser = argparse.ArgumentParser(
        description="Run the OMOP Deep Agent for a synthetic patient."
    )

    parser.add_argument(
        "--pid",
        type=int,
        default=17247,
        help="Synthetic OMOP person_id to analyse.",
    )

    parser.add_argument(
        "--query",
        type=str,
        default="Summarise this patient's medical history.",
        help="Question to ask the agent.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    # CLI execution
    args = parse_args()

    print(f"\nRunning OMOP agent for person_id={args.pid}")
    print(f"Query: {args.query}")
    print("=" * 60)

    response = ask(args.pid, args.query)

    print(response)