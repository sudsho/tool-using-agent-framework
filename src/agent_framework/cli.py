"""Tiny CLI. Mostly used for the dashboard and quick agent runs."""
from __future__ import annotations

import argparse
import sys

from . import AgentState, Tracer
from .config import get, load
from .llm import make_client
from .templates import build_plan_act_agent, build_react_agent
from .tools import default_registry


def cmd_run(args: argparse.Namespace) -> int:
    cfg = load(args.config)
    client = make_client(get(cfg, "llm.provider", "openai"), get(cfg, "llm.model", "gpt-4o-mini"))
    tools = default_registry()
    if args.template == "react":
        graph = build_react_agent(client, tools)
    elif args.template == "plan":
        graph = build_plan_act_agent(client, tools)
    else:
        print(f"unknown template {args.template!r}", file=sys.stderr)
        return 2
    tracer = Tracer(out_dir=get(cfg, "tracer.out_dir", "./traces"))
    compiled = graph.compile(tracer=tracer, recursion_limit=get(cfg, "graph.recursion_limit", 25))
    state = compiled.invoke(AgentState(input=args.input))
    print(state.output or "(no output)")
    print(f"trace: {tracer.current_trace}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="agent-framework")
    sub = parser.add_subparsers(dest="cmd")

    p_run = sub.add_parser("run", help="Run an agent")
    p_run.add_argument("--config", default="configs/default.yaml")
    p_run.add_argument("--template", choices=["react", "plan"], default="react")
    p_run.add_argument("input", help="user input / goal")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
