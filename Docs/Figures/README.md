# Thesis Figures (Mermaid sources)

Diagram sources for the thesis, written in [Mermaid](https://mermaid.js.org/).
Each `.mmd` file renders to an image (SVG recommended for print) that is
inserted at the corresponding `[TODO: Insert Figure...]` marker in `THESIS.md`.

| File | Thesis section | Figure |
| :--- | :--- | :--- |
| `fig1_ingestion_pipeline.mmd` | §4.1 | Offline ingestion pipeline (5 stages) |
| `fig2_base_query_pipeline.mmd` | §4.2 | Base linear online query pipeline |
| `fig3a_baseline.mmd` | §4.3 | Retrieval config (a): bi-encoder only (`baseline`) |
| `fig3b_reranked.mmd` | §4.3 | Retrieval config (b): \+ cross-encoder rerank (`reranked`) |
| `fig3c_expanded.mmd` | §4.3 | Retrieval config (c): \+ contiguous page expansion (`expanded`) |
| `fig4_agentic_graph.mmd` | §6.1 | Agentic cyclic query graph (CRAG + Self-RAG loops) |

## Rendering

**Easiest — online, no install:** open https://mermaid.live, paste a file's
contents, then export PNG (2–3× scale) or SVG.

**Local CLI** (requires Node.js + mermaid-cli):

```powershell
# one-time setup
winget install OpenJS.NodeJS
npm install -g @mermaid-js/mermaid-cli

# render all figures (run from the repo root)
Get-ChildItem Docs/Figures/*.mmd | ForEach-Object {
  mmdc -i $_.FullName -o ($_.FullName -replace '\.mmd$', '.svg') -b white
}
```

The three `fig3*` panels are separate images by design: place them side by side
(or stacked) in the document as sub-figures (a), (b), (c) of Figure 4.3, with a
single shared caption. Panel styling is consistent, and the stage *added* by
each configuration is drawn with a bold outline so the progression is obvious.

**VS Code:** install the "Markdown Preview Mermaid Support" (or "Mermaid Editor")
extension for live preview and export.

## Note on fig4 (agentic graph)

`fig4_agentic_graph.mmd` mirrors the real compiled LangGraph. It can be
regenerated from the implementation at any time and then re-styled:

```powershell
python -c "from alexandria.agent.graph import app; print(app.get_graph().draw_mermaid())"
```

This guarantees the figure never drifts from the code.
