"""
Entry point for the Parking Space Reservation Chatbot.

Usage:
    python main.py ingest      # Load and embed parking documents
    python main.py initdb      # Create and seed the SQLite database
    python main.py chat        # Start the interactive chatbot
    python main.py evaluate    # Run the RAG evaluation suite
    python main.py --help      # Show this help message
"""

import sys

from src.config import get_logger, settings

logger = get_logger(__name__)


def cmd_ingest() -> None:
    """Ingest parking documents into ChromaDB."""
    from src.ingestion.embedder import ingest_documents

    logger.info("Starting document ingestion...")
    settings.ensure_directories()
    ingest_documents()
    logger.info("Ingestion complete.")


def cmd_initdb() -> None:
    """Create and seed the SQLite database."""
    from src.database.schema import initialise_database

    logger.info("Initialising database...")
    settings.ensure_directories()
    initialise_database()
    logger.info("Database ready.")


def cmd_chat() -> None:
    """Start the interactive chatbot CLI."""
    from src.chatbot.graph import build_graph

    logger.info("Starting chatbot...")
    settings.ensure_directories()
    graph = build_graph()

    print("\nParking Chatbot — type 'quit' or 'exit' to stop.\n")
    state: dict = {"messages": [], "reservation": {}, "intent": None}

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if user_input.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        if not user_input:
            continue

        prev_rid = state.get("pending_reservation_id")
        state = graph.invoke({**state, "user_input": user_input})
        response = state.get("response", "")
        if response:
            print(f"\nBot: {response}\n")

        # Stage 4: start workflow if a reservation was just saved
        new_rid = state.get("pending_reservation_id")
        if new_rid and new_rid != prev_rid:
            try:
                from src.workflow.graph import start_reservation_workflow
                start_reservation_workflow(new_rid, state.get("reservation", {}))
                logger.info("Workflow started for reservation %d.", new_rid)
            except Exception as wf_exc:
                logger.warning("Could not start workflow for reservation %d: %s", new_rid, wf_exc)


def cmd_evaluate() -> None:
    """Run the RAG evaluation suite and generate a report."""
    from src.evaluation.report import run_evaluation

    logger.info("Running evaluation...")
    settings.ensure_directories()
    run_evaluation()
    logger.info("Evaluation complete. See reports/ directory.")


_COMMANDS: dict = {
    "ingest": cmd_ingest,
    "initdb": cmd_initdb,
    "chat": cmd_chat,
    "evaluate": cmd_evaluate,
}

_HELP = __doc__


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"--help", "-h"}:
        print(_HELP)
        return

    command = sys.argv[1].lower()
    if command not in _COMMANDS:
        print(f"Unknown command: {command}")
        print(f"Available commands: {', '.join(_COMMANDS)}")
        sys.exit(1)

    _COMMANDS[command]()


if __name__ == "__main__":
    main()
