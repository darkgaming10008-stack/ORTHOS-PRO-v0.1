import pytest
from core.tool_registry import tool, ToolRegistry, _to_ollama_declaration, _convert_type, _TOOLS


# ── _convert_type ────────────────────────────────────────────────────────────


class TestConvertType:
    def test_maps_string(self):
        assert _convert_type("STRING") == "string"

    def test_maps_object(self):
        assert _convert_type("OBJECT") == "object"

    def test_passes_through_lowercase(self):
        assert _convert_type("string") == "string"

    def test_handles_non_string(self):
        assert _convert_type(None) is None


# ── _to_ollama_declaration ───────────────────────────────────────────────────


class TestToOllamaDeclaration:
    def test_generates_correct_format(self):
        params = {
            "type": "object",
            "properties": {
                "name": {"type": "STRING", "description": "The name"},
            },
            "required": ["name"],
        }
        decl = _to_ollama_declaration("greet", "Greet someone", params)
        assert decl["type"] == "function"
        assert decl["function"]["name"] == "greet"
        assert decl["function"]["description"] == "Greet someone"
        assert decl["function"]["parameters"]["type"] == "object"
        assert decl["function"]["parameters"]["properties"]["name"]["type"] == "string"

    def test_no_required(self):
        decl = _to_ollama_declaration("noop", "Does nothing", {})
        assert "required" not in decl["function"]["parameters"]


# ── @tool decorator ──────────────────────────────────────────────────────────


class TestToolDecorator:
    def _clear_globals(self):
        _TOOLS.clear()

    def test_decorator_registers_function(self):
        self._clear_globals()

        @tool("my_tool", description="A test tool")
        def my_tool(parameters=None, response=None, player=None):
            return "hello"

        registry = ToolRegistry()
        fn = registry.get("my_tool")
        assert fn is not None
        assert fn(parameters={}, response=None, player=None) == "hello"

    def test_decorator_with_parameters(self):
        self._clear_globals()

        @tool("calc", description="Calculate", parameters={
            "type": "object",
            "properties": {"x": {"type": "INTEGER"}},
            "required": ["x"],
        })
        def calc(parameters=None, response=None, player=None):
            return parameters.get("x", 0) * 2

        registry = ToolRegistry()
        decls = registry.get_declarations()
        calc_decls = [d for d in decls if d["function"]["name"] == "calc"]
        assert len(calc_decls) == 1


# ── ToolRegistry ─────────────────────────────────────────────────────────────


class TestToolRegistry:
    def _clear_globals(self):
        _TOOLS.clear()

    def test_register_and_get(self):
        self._clear_globals()
        registry = ToolRegistry()

        def my_fn(parameters=None, response=None, player=None):
            return "ok"

        registry.register(my_fn, "greet", "A greeting", {"properties": {}})
        fn = registry.get("greet")
        assert fn is not None
        assert fn(parameters={}, response=None, player=None) == "ok"

    def test_get_returns_none_for_unknown(self):
        registry = ToolRegistry()
        assert registry.get("nonexistent") is None

    def test_list_tools(self):
        self._clear_globals()
        registry = ToolRegistry()

        def fn1(parameters=None, response=None, player=None):
            return 1

        def fn2(parameters=None, response=None, player=None):
            return 2

        registry.register(fn1, "one", "First")
        registry.register(fn2, "two", "Second")
        tools = registry.list_tools()
        names = {t["name"] for t in tools}
        assert "one" in names
        assert "two" in names

    def test_execute_calls_with_parameters(self):
        self._clear_globals()
        registry = ToolRegistry()

        def echo(parameters=None, response=None, player=None):
            return parameters

        registry.register(echo, "echo")
        result = registry.execute("echo", {"msg": "hi"})
        assert result == {"msg": "hi"}

    def test_execute_raises_on_unknown(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError, match="unknown_tool"):
            registry.execute("unknown_tool", {})

    def test_get_declarations_format(self):
        self._clear_globals()
        registry = ToolRegistry()

        def fn(parameters=None, response=None, player=None):
            pass

        registry.register(
            fn, "test_tool", "A test",
            {
                "type": "object",
                "properties": {"arg": {"type": "STRING"}},
                "required": ["arg"],
            },
        )
        decls = registry.get_declarations()
        assert len(decls) == 1
        d = decls[0]
        assert d["type"] == "function"
        assert d["function"]["name"] == "test_tool"
        assert d["function"]["parameters"]["properties"]["arg"]["type"] == "string"


# ── discover ─────────────────────────────────────────────────────────────────


class TestDiscover:
    def _clear_globals(self):
        _TOOLS.clear()

    def test_raises_on_missing_dir(self):
        with pytest.raises(NotADirectoryError):
            from core.tool_registry import discover
            discover("C:/nonexistent_path_xyz_123")

    def test_discovers_tools(self, tmp_path):
        self._clear_globals()
        from core.tool_registry import discover
        actions_dir = tmp_path / "actions"
        actions_dir.mkdir()
        mod_file = actions_dir / "test_module.py"
        mod_file.write_text(
            'from core.tool_registry import tool\n'
            '@tool("disc_tool", description="Discovered", parameters={})\n'
            'def disc_tool(parameters=None, response=None, player=None):\n'
            '    return "found"\n'
        )
        registry = discover(str(actions_dir))
        fn = registry.get("disc_tool")
        assert fn is not None
