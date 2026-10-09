# Terminal demo for the README. Regenerate with: vhs docs/demo.tape
# Needs the API running on :8000 with a local Ollama model (make run).
Output docs/demo.gif
Set FontSize 13
Set Width 1400
Set Height 540
Set Theme "Catppuccin Mocha"
Set TypingSpeed 25ms

Type "# /analyze: signals, retrieved definitions, LLM explanation, guardrail"
Enter
Type `curl -s "localhost:8000/analyze?ticker=PETR4" | python3 -m json.tool`
Enter
Sleep 25s
Type "# offline evals scored from recorded LLM output (no network)"
Enter
Type "uv run python -m quantlens.evals | head -8"
Enter
Sleep 6s
