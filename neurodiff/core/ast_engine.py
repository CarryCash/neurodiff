"""AST engine for semantic code analysis."""
from __future__ import annotations

from dataclasses import dataclass

from .semantic_events import (
    ClassAdded,
    ClassModified,
    FunctionAdded,
    FunctionModified,
    FunctionRemoved,
    ImportAdded,
    ImportRemoved,
    NeuroDiffError,
    SemanticEvent,
)


def calculate_cyclomatic_complexity(code: str) -> int:
    """Calculate cyclomatic complexity of code."""
    count = 1  # Base complexity
    keywords = ["if", "elif", "for", "while", "except", "and", "or"]

    # Simple token-based counting
    tokens = code.split()
    for token in tokens:
        clean_token = token.strip("(){}[];,:")
        if clean_token in keywords:
            count += 1

    return max(1, count)


class ASTEngine:
    """Engine for extracting semantic events from AST analysis."""

    SUPPORTED_LANGUAGES = {"python", "javascript", "typescript"}

    def __init__(self) -> None:
        """Initialize the AST engine."""
        try:
            import tree_sitter
            import tree_sitter_languages

            self.tree_sitter = tree_sitter
            self.tree_sitter_languages = tree_sitter_languages
        except ImportError as e:
            raise NeuroDiffError(
                "tree-sitter not installed. Install with: pip install tree-sitter tree-sitter-languages"
            ) from e

    def extract_events(
        self,
        content_before: str,
        content_after: str,
        language: str,
        file_path: str,
    ) -> list[SemanticEvent]:
        """Extract semantic events from code changes."""
        if language not in self.SUPPORTED_LANGUAGES:
            return []

        events: list[SemanticEvent] = []

        try:
            # Parse both versions
            tree_before = self._parse_code(content_before, language)
            tree_after = self._parse_code(content_after, language)

            # Extract functions, classes, and imports
            functions_before = self._extract_functions(tree_before, language)
            functions_after = self._extract_functions(tree_after, language)
            classes_before = self._extract_classes(tree_before, language)
            classes_after = self._extract_classes(tree_after, language)
            imports_before = self._extract_imports(tree_before, language)
            imports_after = self._extract_imports(tree_after, language)

            # Detect function changes
            events.extend(
                self._detect_function_changes(
                    functions_before,
                    functions_after,
                    content_before,
                    content_after,
                    language,
                    file_path,
                )
            )

            # Detect class changes
            events.extend(
                self._detect_class_changes(
                    classes_before, classes_after, language, file_path
                )
            )

            # Detect import changes
            events.extend(
                self._detect_import_changes(
                    imports_before, imports_after, language, file_path
                )
            )

            return events
        except Exception as e:
            # Graceful degradation
            return []

    def _parse_code(self, code: str, language: str) -> object:
        """Parse code using tree-sitter."""
        if not code.strip():
            class EmptyTree:
                root_node = None
            return EmptyTree()

        try:
            lang_module = getattr(self.tree_sitter_languages, language)
            parser = self.tree_sitter.Parser()
            parser.set_language(lang_module.language)
            tree = parser.parse(code.encode())
            return tree
        except Exception:
            return type("EmptyTree", (), {"root_node": None})()

    def _extract_functions(
        self, tree: object, language: str
    ) -> dict[str, dict]:
        """Extract function definitions from AST."""
        functions: dict[str, dict] = {}

        if not hasattr(tree, "root_node") or tree.root_node is None:
            return functions

        try:
            self._traverse_functions(tree.root_node, functions, language)
        except Exception:
            pass

        return functions

    def _traverse_functions(
        self, node: object, functions: dict, language: str
    ) -> None:
        """Traverse AST and collect function definitions."""
        if not node:
            return

        if language == "python":
            if hasattr(node, "type") and node.type == "function_definition":
                func_name = self._extract_function_name(node, language)
                if func_name:
                    calls = self._extract_calls(node, language)
                    functions[func_name] = {
                        "start_line": node.start_point[0],
                        "end_line": node.end_point[0],
                        "calls": calls,
                        "node": node,
                    }
        else:  # javascript / typescript
            node_type = getattr(node, "type", "")

            # Standard: function foo() {}
            if node_type == "function_declaration":
                func_name = self._extract_function_name(node, language)
                if func_name:
                    calls = self._extract_calls(node, language)
                    functions[func_name] = {
                        "start_line": node.start_point[0],
                        "end_line": node.end_point[0],
                        "calls": calls,
                        "node": node,
                    }

            # const foo = () => {} / const foo = function() {}
            elif node_type in ("lexical_declaration", "variable_declaration"):
                self._traverse_js_variable_func(node, functions, language)

            # method_definition inside a class body (handled separately via classes)
            # but also capture standalone method_definition if encountered
            elif node_type == "method_definition":
                func_name = self._extract_function_name(node, language)
                if func_name:
                    calls = self._extract_calls(node, language)
                    functions[func_name] = {
                        "start_line": node.start_point[0],
                        "end_line": node.end_point[0],
                        "calls": calls,
                        "node": node,
                    }

        if hasattr(node, "children"):
            for child in node.children:
                self._traverse_functions(child, functions, language)

    def _traverse_js_variable_func(
        self, node: object, functions: dict, language: str
    ) -> None:
        """Handle JS/TS: const/let foo = () => {} or const foo = function() {}."""
        if not hasattr(node, "children"):
            return
        for child in node.children:
            if getattr(child, "type", "") == "variable_declarator":
                try:
                    name_node = child.child_by_field_name("name")
                    value_node = child.child_by_field_name("value")
                    if name_node and value_node:
                        vtype = getattr(value_node, "type", "")
                        if vtype in ("arrow_function", "function_expression", "function"):
                            func_name = name_node.text.decode()
                            calls = self._extract_calls(value_node, language)
                            functions[func_name] = {
                                "start_line": child.start_point[0],
                                "end_line": child.end_point[0],
                                "calls": calls,
                                "node": value_node,
                            }
                except Exception:
                    pass

    def _extract_function_name(self, node: object, language: str) -> str | None:
        """Extract function name from AST node."""
        try:
            if hasattr(node, "child_by_field_name"):
                name_node = node.child_by_field_name("name")
                if name_node:
                    return name_node.text.decode()
        except Exception:
            pass
        return None
        
    def _extract_calls(self, node: object, language: str) -> list[str]:
        """Extract function calls within a node."""
        calls: list[str] = []

        def traverse(n: object) -> None:
            if not n:
                return
            ntype = getattr(n, "type", "")
            # Python: call  | JS/TS: call_expression
            if ntype in ("call", "call_expression"):
                try:
                    if hasattr(n, "child_by_field_name"):
                        func_node = n.child_by_field_name("function")
                        if func_node and hasattr(func_node, "text"):
                            raw = func_node.text.decode()
                            # Keep only the rightmost name for member expressions
                            # e.g. "self.foo" → "self.foo", "obj.method" → "obj.method"
                            calls.append(raw)
                except Exception:
                    pass
            if hasattr(n, "children"):
                for child in n.children:
                    traverse(child)

        traverse(node)
        return list(dict.fromkeys(calls))

    def _extract_classes(
        self, tree: object, language: str
    ) -> dict[str, dict]:
        """Extract class definitions from AST."""
        classes: dict[str, dict] = {}
        if not hasattr(tree, "root_node") or tree.root_node is None:
            return classes
        try:
            self._traverse_classes(tree.root_node, classes, language)
        except Exception:
            pass
        return classes

    def _traverse_classes(
        self, node: object, classes: dict, language: str
    ) -> None:
        """Traverse AST and collect class definitions."""
        if not node:
            return
        if language == "python":
            class_type = "class_definition"
        else:
            class_type = "class_declaration"

        if hasattr(node, "type") and node.type == class_type:
            class_name = self._extract_class_name(node, language)
            if class_name:
                methods = self._extract_class_methods(node, language)
                inherits_from = self._extract_class_inheritance(node, language)
                classes[class_name] = {
                    "start_line": node.start_point[0],
                    "end_line": node.end_point[0],
                    "methods": methods,
                    "inherits_from": inherits_from,
                    "node": node,
                }
        if hasattr(node, "children"):
            for child in node.children:
                self._traverse_classes(child, classes, language)

    def _extract_class_name(self, node: object, language: str) -> str | None:
        """Extract class name from AST node."""
        try:
            if hasattr(node, "child_by_field_name"):
                name_node = node.child_by_field_name("name")
                if name_node:
                    return name_node.text.decode()
        except Exception:
            pass
        return None

    def _extract_class_methods(self, node: object, language: str) -> list[str]:
        """Extract method names from a class definition."""
        methods: list[str] = []
        if not hasattr(node, "children"):
            return methods
        try:
            if language == "python":
                method_type = "function_definition"
            else:
                # JS/TS classes use method_definition
                method_type = "method_definition"

            body_node = node.child_by_field_name("body")
            if body_node and hasattr(body_node, "children"):
                for child in body_node.children:
                    if hasattr(child, "type") and child.type == method_type:
                        method_name = self._extract_function_name(child, language)
                        if method_name:
                            methods.append(method_name)
        except Exception:
            pass
        return methods
        
    def _extract_class_inheritance(self, node: object, language: str) -> list[str]:
        """Extract base classes / interfaces from a class node."""
        inherits: list[str] = []
        try:
            if language == "python":
                superclasses_node = node.child_by_field_name("superclasses")
                if superclasses_node and hasattr(superclasses_node, "children"):
                    for child in superclasses_node.children:
                        if getattr(child, "type", "") in ("identifier", "attribute"):
                            inherits.append(child.text.decode())
            else:  # javascript / typescript
                # heritage_clause contains extends / implements
                for child in getattr(node, "children", []):
                    ctype = getattr(child, "type", "")
                    if ctype == "class_heritage":
                        for heritage_child in getattr(child, "children", []):
                            htype = getattr(heritage_child, "type", "")
                            if htype in ("identifier", "extends_clause", "implements_clause"):
                                try:
                                    inherits.append(heritage_child.text.decode())
                                except Exception:
                                    pass
        except Exception:
            pass
        return inherits


    def _extract_imports(
        self, tree: object, language: str
    ) -> dict[str, dict]:
        """Extract import statements from AST."""
        imports: dict[str, dict] = {}
        if not hasattr(tree, "root_node") or tree.root_node is None:
            return imports
        try:
            self._traverse_imports(tree.root_node, imports, language)
        except Exception:
            pass
        return imports

    def _traverse_imports(
        self, node: object, imports: dict, language: str
    ) -> None:
        """Traverse AST and collect import statements."""
        if not node:
            return
        try:
            if language == "python":
                import_types = {"import_statement", "import_from_statement"}
            else:
                import_types = {"import_statement"}

            if hasattr(node, "type") and node.type in import_types:
                module = self._extract_import_module(node, language)
                if module:
                    symbols = self._extract_import_symbols(node, language)
                    imports[module] = {
                        "line": node.start_point[0],
                        "symbols": symbols,
                    }
            if hasattr(node, "children"):
                for child in node.children:
                    self._traverse_imports(child, imports, language)
        except Exception:
            pass

    def _extract_import_module(self, node: object, language: str) -> str | None:
        """Extract module name from import statement."""
        try:
            if hasattr(node, "text"):
                statement = node.text.decode()
                if language == "python":
                    if node.type == "import_from_statement":
                        module_name_node = node.child_by_field_name("module_name")
                        if module_name_node:
                            return module_name_node.text.decode()
                    elif node.type == "import_statement":
                        # simplified handling
                        parts = statement.split()
                        if len(parts) > 1:
                            return parts[1].split('.')[0]
                else:
                    if "from" in statement:
                        start = statement.find("'")
                        end = statement.rfind("'")
                        if start == -1:
                            start = statement.find('"')
                            end = statement.rfind('"')
                        if start != -1 and end > start:
                            return statement[start + 1 : end]
        except Exception:
            pass
        return None
        
    def _extract_import_symbols(self, node: object, language: str) -> list[str]:
        """Extract named symbols from an import statement."""
        symbols: list[str] = []
        try:
            if language == "python":
                if getattr(node, "type", "") == "import_from_statement":
                    for child in getattr(node, "children", []):
                        ctype = getattr(child, "type", "")
                        if ctype in ("dotted_name", "aliased_import", "identifier"):
                            symbols.append(child.text.decode())
            else:  # javascript / typescript
                # import { foo, bar } from 'mod'
                # AST: import_statement -> import_clause -> named_imports -> import_specifier*
                for child in getattr(node, "children", []):
                    if getattr(child, "type", "") == "import_clause":
                        for clause_child in getattr(child, "children", []):
                            if getattr(clause_child, "type", "") == "named_imports":
                                for spec in getattr(clause_child, "children", []):
                                    if getattr(spec, "type", "") == "import_specifier":
                                        try:
                                            name_node = spec.child_by_field_name("name")
                                            if name_node:
                                                symbols.append(name_node.text.decode())
                                        except Exception:
                                            pass
        except Exception:
            pass
        return symbols

    def _detect_function_changes(
        self,
        functions_before: dict[str, dict],
        functions_after: dict[str, dict],
        content_before: str,
        content_after: str,
        language: str,
        file_path: str,
    ) -> list[SemanticEvent]:
        """Detect function additions, modifications, and removals."""
        events: list[SemanticEvent] = []

        # Find added functions
        for name, info in functions_after.items():
            if name not in functions_before:
                func_code = self._extract_function_body(
                    content_after, info["start_line"], info["end_line"]
                )
                complexity = calculate_cyclomatic_complexity(func_code)
                body_lines = info["end_line"] - info["start_line"] + 1
                events.append(
                    FunctionAdded(
                        name=name,
                        file=file_path,
                        start_line=info["start_line"] + 1,  # 1-indexed
                        body_lines=body_lines,
                        calls=info.get("calls", []),
                        cyclomatic_complexity=complexity,
                    )
                )

        # Find removed functions
        for name, info in functions_before.items():
            if name not in functions_after:
                events.append(
                    FunctionRemoved(
                        name=name,
                        file=file_path,
                    )
                )

        # Find modified functions
        for name in functions_before:
            if name in functions_after:
                before_info = functions_before[name]
                after_info = functions_after[name]
                before_code = self._extract_function_body(
                    content_before, before_info["start_line"], before_info["end_line"]
                )
                after_code = self._extract_function_body(
                    content_after, after_info["start_line"], after_info["end_line"]
                )

                if before_code != after_code:
                    complexity_before = calculate_cyclomatic_complexity(before_code)
                    complexity_after = calculate_cyclomatic_complexity(after_code)
                    
                    calls_before = set(before_info.get("calls", []))
                    calls_after = set(after_info.get("calls", []))
                    
                    events.append(
                        FunctionModified(
                            name=name,
                            file=file_path,
                            start_line=after_info["start_line"] + 1,
                            lines_before=before_info["end_line"] - before_info["start_line"] + 1,
                            lines_after=after_info["end_line"] - after_info["start_line"] + 1,
                            signature_changed=False, # Basic heuristic, could compare signature string
                            calls_added=list(calls_after - calls_before),
                            calls_removed=list(calls_before - calls_after),
                            complexity_before=complexity_before,
                            complexity_after=complexity_after,
                        )
                    )

        return events

    def _detect_class_changes(
        self,
        classes_before: dict[str, dict],
        classes_after: dict[str, dict],
        language: str,
        file_path: str,
    ) -> list[SemanticEvent]:
        """Detect class additions, modifications, and removals."""
        events: list[SemanticEvent] = []

        # Find added classes
        for name, info in classes_after.items():
            if name not in classes_before:
                events.append(
                    ClassAdded(
                        name=name,
                        file=file_path,
                        methods=info.get("methods", []),
                        inherits_from=info.get("inherits_from", []),
                    )
                )

        # Find modified classes
        for name in classes_before:
            if name in classes_after:
                before_info = classes_before[name]
                after_info = classes_after[name]
                before_methods = set(before_info.get("methods", []))
                after_methods = set(after_info.get("methods", []))

                methods_added = list(after_methods - before_methods)
                methods_removed = list(before_methods - after_methods)

                if methods_added or methods_removed:
                    events.append(
                        ClassModified(
                            name=name,
                            file=file_path,
                            methods_added=methods_added,
                            methods_removed=methods_removed,
                        )
                    )

        return events

    def _detect_import_changes(
        self,
        imports_before: dict[str, dict],
        imports_after: dict[str, dict],
        language: str,
        file_path: str,
    ) -> list[SemanticEvent]:
        """Detect import additions and removals."""
        events: list[SemanticEvent] = []

        # Find added imports
        for module, info in imports_after.items():
            if module not in imports_before:
                events.append(
                    ImportAdded(
                        module=module,
                        file=file_path,
                        symbols=info.get("symbols", []),
                    )
                )

        # Find removed imports
        for module, info in imports_before.items():
            if module not in imports_after:
                events.append(
                    ImportRemoved(
                        module=module,
                        file=file_path,
                    )
                )

        return events

    def _extract_function_body(
        self, content: str, start_line: int, end_line: int
    ) -> str:
        """Extract function body from source code."""
        lines = content.split("\\n")
        start = max(0, start_line)
        end = min(len(lines), end_line + 1)
        return "\\n".join(lines[start:end])
