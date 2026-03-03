"""Generate a PNG visualization of the LangGraph pipeline."""

from pathlib import Path

from orchestrator.graph import build_graph


def main():
    graph = build_graph()
    png_bytes = graph.get_graph().draw_mermaid_png()
    out = Path("graph.png")
    out.write_bytes(png_bytes)
    print(f"Graph saved to {out.resolve()}")


if __name__ == "__main__":
    main()
