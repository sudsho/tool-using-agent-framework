from .base import FuncTool, Tool, ToolRegistry, ToolSchema
from .calculator import CalculatorTool
from .code_executor import CodeExecutorTool
from .file_io import FileIOTool
from .web_search import WebSearchTool


def default_registry(
    web_search: bool = True,
    calculator: bool = True,
    code_executor: bool = True,
    file_io: bool = True,
    web_search_provider: str = "tavily",
    sandbox_root: str = "./sandbox",
) -> ToolRegistry:
    """Convenience: a registry pre-populated with the built-in tools."""
    reg = ToolRegistry()
    if web_search:
        reg.register(WebSearchTool(provider=web_search_provider))
    if calculator:
        reg.register(CalculatorTool())
    if code_executor:
        reg.register(CodeExecutorTool())
    if file_io:
        reg.register(FileIOTool(sandbox_root=sandbox_root))
    return reg


__all__ = [
    "CalculatorTool",
    "CodeExecutorTool",
    "FileIOTool",
    "FuncTool",
    "Tool",
    "ToolRegistry",
    "ToolSchema",
    "WebSearchTool",
    "default_registry",
]
