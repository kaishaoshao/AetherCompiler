"""
Compiler Infrastructure: Unified Pass Manager & Dynamic Pass Registry (借鉴 MLIR PassManager & LLVM PassRegistry).

Core Architectural Innovations:
1. Dynamic Pass Registry (借鉴 LLVM/MLIR PassRegistry & RegisterPass):
   - Central catalog of all compiler optimization and lowering passes.
   - Decorator @register_pass and dynamic discovery with alias support.
   - Self-documenting pass table generator for compiler documentation.
2. Textual Pipeline Specification & Parsing (借鉴 MLIR Textual Pass Pipeline):
   - Supports pipeline strings like "const-fold{sram_capacity_bytes=16384},h-fusion,fusion,dce".
   - Supports nested dialect wrappers like "builtin.module(const-fold,fusion,dce)".
   - Type-safe option parsing (int, float, bool, None, string).
3. Pass Performance Telemetry:
   - Tracks millisecond-level execution latency across graph rewriting, memory planning, and lowering passes.
4. Observability & IR Dumps (--print-after-all):
   - Intermediate representation dumps after every optimization pass.
"""

import time
import re
import inspect
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple, Union


class CompilerPass:
    """Base class for all compiler passes."""
    @property
    def name(self) -> str:
        return self.__class__.__name__

    @property
    def description(self) -> str:
        return self.__doc__ or "No description provided."

    def run(self, target: Any) -> Any:
        raise NotImplementedError


@dataclass
class PassInfo:
    """Metadata describing a registered compiler pass."""
    name: str
    pass_cls: type
    description: str = ""
    dialect_stage: str = "graph"  # "graph", "flow", "tir", "hal", "rv64"
    aliases: List[str] = field(default_factory=list)
    options_schema: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        alias_str = f", aliases={self.aliases}" if self.aliases else ""
        return f"PassInfo(name='{self.name}', cls={self.pass_cls.__name__}, stage='{self.dialect_stage}'{alias_str})"


def _cast_value(val_str: str) -> Any:
    """Auto-casts string option values to appropriate Python types."""
    v_clean = val_str.strip()
    v_lower = v_clean.lower()
    if v_lower in ("true", "yes", "on"):
        return True
    if v_lower in ("false", "no", "off"):
        return False
    if v_lower in ("none", "null"):
        return None
    # Integer
    try:
        return int(v_clean)
    except ValueError:
        pass
    # Float
    try:
        return float(v_clean)
    except ValueError:
        pass
    # Quoted string
    if (v_clean.startswith('"') and v_clean.endswith('"')) or (v_clean.startswith("'") and v_clean.endswith("'")):
        return v_clean[1:-1]
    return v_clean


def parse_pipeline(pipeline_spec: str) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Parses an LLVM/MLIR-style textual pipeline specification.
    
    Examples:
        "const-fold,h-fusion,fusion,dce"
        "const-fold{sram_capacity_bytes=32768},dce"
        "builtin.module(const-fold,fusion{patterns=all},dce)"
    
    Returns:
        List of (pass_name, options_dict)
    """
    spec = pipeline_spec.strip()
    if not spec:
        return []

    # Strip outer wrapper if present: e.g. "builtin.module(...)" or "func.func(...)" or "graph(...)"
    wrapper_match = re.match(r"^[a-zA-Z0-9_\.]+\((.*)\)$", spec, re.DOTALL)
    if wrapper_match:
        spec = wrapper_match.group(1).strip()

    # Split top-level commas (ignoring commas inside {} or ())
    tokens: List[str] = []
    current_token: List[str] = []
    brace_depth = 0
    paren_depth = 0

    for ch in spec:
        if ch == '{':
            brace_depth += 1
            current_token.append(ch)
        elif ch == '}':
            brace_depth -= 1
            current_token.append(ch)
        elif ch == '(':
            paren_depth += 1
            current_token.append(ch)
        elif ch == ')':
            paren_depth -= 1
            current_token.append(ch)
        elif ch == ',' and brace_depth == 0 and paren_depth == 0:
            tok = "".join(current_token).strip()
            if tok:
                tokens.append(tok)
            current_token = []
        else:
            current_token.append(ch)

    if current_token:
        tok = "".join(current_token).strip()
        if tok:
            tokens.append(tok)

    results: List[Tuple[str, Dict[str, Any]]] = []
    for token in tokens:
        # Match pass_name or pass_name{opt1=val1, opt2=val2}
        m = re.match(r"^([a-zA-Z0-9_\-]+)(?:\{([^}]*)\})?$", token)
        if not m:
            raise ValueError(f"Invalid pass specification syntax: '{token}' in pipeline '{pipeline_spec}'")
        pass_name = m.group(1).strip()
        opts_str = m.group(2)
        opts: Dict[str, Any] = {}
        if opts_str:
            for kv in opts_str.split(','):
                kv = kv.strip()
                if not kv:
                    continue
                if '=' not in kv:
                    opts[kv] = True
                else:
                    k, v = kv.split('=', 1)
                    opts[k.strip()] = _cast_value(v)
        results.append((pass_name, opts))
    return results


class PassRegistry:
    """
    Central Pass Registry (借鉴 LLVM/MLIR PassRegistry).
    Maintains all available passes, their schemas, dialect stages, and aliases.
    """
    _registry: Dict[str, PassInfo] = {}
    _aliases: Dict[str, str] = {}

    @classmethod
    def _ensure_builtins_loaded(cls):
        """Lazily imports middle-end passes to populate registry if empty."""
        if not cls._registry:
            try:
                import compiler.midend as midend  # noqa: F401
            except ImportError:
                pass

    @classmethod
    def register(
        cls,
        name: str,
        pass_cls: type,
        description: str = "",
        dialect_stage: str = "graph",
        aliases: Optional[List[str]] = None,
        options_schema: Optional[Dict[str, Any]] = None
    ):
        """Registers a pass class with metadata."""
        aliases = aliases or []
        info = PassInfo(
            name=name,
            pass_cls=pass_cls,
            description=description or getattr(pass_cls, "__doc__", "") or "",
            dialect_stage=dialect_stage,
            aliases=aliases,
            options_schema=options_schema or {}
        )
        cls._registry[name] = info
        for alias in aliases:
            cls._aliases[alias] = name

    @classmethod
    def register_pass(
        cls,
        name: str,
        description: str = "",
        dialect_stage: str = "graph",
        aliases: Optional[List[str]] = None,
        options_schema: Optional[Dict[str, Any]] = None
    ):
        """Class decorator for pass registration."""
        def decorator(subclass: type):
            cls.register(
                name=name,
                pass_cls=subclass,
                description=description,
                dialect_stage=dialect_stage,
                aliases=aliases,
                options_schema=options_schema
            )
            return subclass
        return decorator

    @classmethod
    def get(cls, name: str) -> PassInfo:
        """Retrieves PassInfo by name or alias."""
        cls._ensure_builtins_loaded()
        if name in cls._registry:
            return cls._registry[name]
        if name in cls._aliases:
            canon = cls._aliases[name]
            return cls._registry[canon]
        available = list(cls._registry.keys()) + list(cls._aliases.keys())
        raise KeyError(f"Pass '{name}' not found in PassRegistry. Available passes: {sorted(available)}")

    @classmethod
    def has_pass(cls, name: str) -> bool:
        """Checks if a pass name or alias is registered."""
        cls._ensure_builtins_loaded()
        return (name in cls._registry) or (name in cls._aliases)

    @classmethod
    def list_passes(cls) -> List[PassInfo]:
        """Returns all registered PassInfo objects."""
        cls._ensure_builtins_loaded()
        return list(cls._registry.values())

    @classmethod
    def create_pass(cls, name: str, **kwargs) -> Any:
        """
        Instantiates a pass by name, passing only the kwargs accepted by its constructor.
        """
        info = cls.get(name)
        cls_obj = info.pass_cls
        sig = inspect.signature(cls_obj.__init__)
        accepted_params = set(sig.parameters.keys()) - {"self"}
        has_var_keyword = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

        merged_kwargs = dict(info.options_schema)
        merged_kwargs.update(kwargs)

        if has_var_keyword:
            filtered_kwargs = merged_kwargs
        else:
            filtered_kwargs = {k: v for k, v in merged_kwargs.items() if k in accepted_params}

        return cls_obj(**filtered_kwargs)

    @classmethod
    def generate_table(cls) -> str:
        """Generates a comprehensive Markdown table summarizing all registered passes."""
        cls._ensure_builtins_loaded()
        headers = [
            "Pass Name (CLI Flag)",
            "Class Name",
            "Dialect Stage",
            "Aliases",
            "Parameters & Defaults",
            "Description & Optimization Invariant"
        ]
        rows: List[List[str]] = []
        for info in sorted(cls._registry.values(), key=lambda p: (p.dialect_stage, p.name)):
            alias_str = ", ".join(f"`{a}`" for a in info.aliases) if info.aliases else "-"
            opts_str = ", ".join(f"`{k}={v}`" for k, v in info.options_schema.items()) if info.options_schema else "None"
            desc = " ".join(info.description.strip().splitlines()) if info.description else "No description"
            if len(desc) > 80:
                desc = desc[:77] + "..."
            rows.append([
                f"`{info.name}`",
                f"`{info.pass_cls.__name__}`",
                f"**{info.dialect_stage}**",
                alias_str,
                opts_str,
                desc
            ])

        col_widths = [len(h) for h in headers]
        for row in rows:
            for i, val in enumerate(row):
                col_widths[i] = max(col_widths[i], len(val))

        def format_row(cols: List[str]) -> str:
            return "| " + " | ".join(c.ljust(col_widths[i]) for i, c in enumerate(cols)) + " |"

        separator = "| " + " | ".join("-" * col_widths[i] for i in range(len(headers))) + " |"

        lines = [
            format_row(headers),
            separator
        ]
        for row in rows:
            lines.append(format_row(row))
        return "\n".join(lines)


# Standalone decorator shorthand
register_pass = PassRegistry.register_pass


class PassManager:
    """
    Unified Pipeline Manager (借鉴 MLIR PassManager).
    Sequences optimization and lowering passes with timing telemetry,
    dynamic text pipeline parsing, and intermediate IR inspection.
    """
    def __init__(self, name: str = "ZenNPU_PassPipeline", print_after_all: bool = False):
        self.name = name
        self.print_after_all = print_after_all
        self.passes: List[Any] = []
        self.pass_timings: Dict[str, float] = {}

    def add_pass(self, compiler_pass: Any) -> 'PassManager':
        """Appends a compiler pass instance or callable to the pipeline."""
        self.passes.append(compiler_pass)
        return self

    def add_pass_by_name(self, name: str, **kwargs) -> 'PassManager':
        """Instantiates and appends a registered pass by name."""
        pass_inst = PassRegistry.create_pass(name, **kwargs)
        self.passes.append(pass_inst)
        return self

    def add_pipeline(self, pipeline_spec: str, **default_kwargs) -> 'PassManager':
        """
        Parses a textual pipeline specification string and appends passes.
        
        Example:
            pm.add_pipeline("const-fold{sram_capacity_bytes=16384},h-fusion,fusion,dce")
        """
        parsed = parse_pipeline(pipeline_spec)
        for pass_name, opts in parsed:
            kw = dict(default_kwargs)
            kw.update(opts)
            self.add_pass_by_name(pass_name, **kw)
        return self

    @classmethod
    def from_pipeline_string(
        cls,
        pipeline_spec: str,
        name: str = "ZenNPU_TextualPipeline",
        print_after_all: bool = False,
        **default_kwargs
    ) -> 'PassManager':
        """
        Constructs a PassManager directly from an MLIR-style textual pipeline specification.
        """
        pm = cls(name=name, print_after_all=print_after_all)
        pm.add_pipeline(pipeline_spec, **default_kwargs)
        return pm

    def run(self, target: Any) -> Any:
        """
        Executes all passes in order on the target module/graph.
        """
        current = target
        self.pass_timings.clear()

        for idx, p in enumerate(self.passes, start=1):
            pass_name = getattr(p, "name", p.__class__.__name__)
            t0 = time.perf_counter()

            # Execute pass
            if hasattr(p, "run"):
                res = p.run(current)
                # Some passes return tuple (e.g. MemoryPlanningPass returns (naive, planned, liveness))
                # or (graph, metadata)
                if isinstance(res, tuple):
                    # Convention: if first element is a Graph or IR module, update current
                    if hasattr(res[0], "nodes") or hasattr(res[0], "functions"):
                        current = res[0]
                    else:
                        current = res
                else:
                    current = res
            elif callable(p):
                current = p(current)
            else:
                raise TypeError(f"Invalid pass object in PassManager: {p}")

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            self.pass_timings[pass_name] = elapsed_ms

            if self.print_after_all:
                print("\n" + "=" * 80)
                print(f" // ----- Dump After Pass [{idx}/{len(self.passes)}]: {pass_name} ({elapsed_ms:.2f} ms) -----")
                print("=" * 80)
                if hasattr(current, "print_graph"):
                    current.print_graph(f"After {pass_name}")
                elif hasattr(current, "print_ir"):
                    current.print_ir()
                elif hasattr(current, "to_string"):
                    print(current.to_string())
                print("=" * 80 + "\n")

        return current

    def print_pipeline_summary(self):
        """Prints execution time breakdown for all passes."""
        total_ms = sum(self.pass_timings.values())
        print("\n" + "-" * 60)
        print(f" PassManager [{self.name}] Performance Telemetry:")
        print("-" * 60)
        for name, ms in self.pass_timings.items():
            pct = (ms / total_ms * 100.0) if total_ms > 0 else 0.0
            print(f"  * {name:<35}: {ms:>6.2f} ms ({pct:>5.1f}%)")
        print("-" * 60)
        print(f"  Total Optimization Pass Latency: {total_ms:>6.2f} ms")
        print("-" * 60 + "\n")
