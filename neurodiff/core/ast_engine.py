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
    """Calculate cyclomatic complexity of code.

    Counts control flow constructs: if, elif, else, for, while, except, and, or
    Formula: complexity = 1 + count(decision_points)

    Args:
        code: The source code as a string.

    Returns:
        The cyclomatic complexity value (minimum 1).
    """
    count = 1  # Base complexity
    keywords = ["if", "elif", "for", "while", "except", "and", "or"]

    # Simple token-based counting
    tokens = code.split()
    for token in tokens:
        # Remove common punctuation but keep the word
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
    ) -> list[SemanticEvent]:
        """Extract semantic events from code changes.

        Args:
            content_before: The code content before changes.
            content_after: The code content after changes.
            language: The programming language (python, javascript, typescript).

        Returns:
            List of semantic events extracted from the changes.

        Raises:
            NeuroDiffError: If language is not supported.
        """
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
                )
            )

            # Detect class changes
            events.extend(
                self._detect_class_changes(
                    classes_before, classes_after, language
                )
            )

            # Detect import changes
            events.extend(
                self._detect_import_changes(
                    imports_before, imports_after, language
                )
            )

            return events
        except Exception as e:
            # Graceful degradation
            return []

    def _parse_code(self, code: str, language: str) -> object:
        """Parse code using tree-sitter.

        Args:
            code: The source code to parse.
            language: The programming language.

        Returns:
            The parse tree object.
        """
        if not code.strip():
            # Return an empty tree-like object
            class EmptyTree:
                root_node = None

            return EmptyTree()

        try:
            lang_module = getattr(self.tree_sitter_languages, language)
            parser = self.tree_sitter.Parser()
            parser.set_language(lang_module.language)
            tree = parser.parse(code.encode())
            return tree
        except Exception as e:
            # Graceful degradation
            return type("EmptyTree", (), {"root_node": None})()

    def _extract_functions(
        self, tree: object, language: str
    ) -> dict[str, dict]:
        """Extract function definitions from AST.

        Args:
            tree: The parse tree object.
            language: The programming language.

        Returns:
            Dictionary mapping function names to their metadata.
        """
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
        """Traverse AST and collect function definitions.

        Args:
            node: The AST node to traverse.
            functions: Dictionary to accumulate functions.
            language: The programming language.
        """
        if not node:
            return

        # Check for function definition nodes based on language
        if language == "python":
            function_type = "function_definition"
        else:  # javascript, typescript
            function_type = "function_declaration"

        if hasattr(node, "type") and node.type == function_type:
            func_name = self._extract_function_name(node, language)
            if func_name:
                functions[func_name] = {
                    "start_line": node.start_point[0],
                    "end_line": node.end_point[0],
                    "node": node,
                }

        # Recursively traverse children
        if hasattr(node, "children"):
            for child in node.children:
                self._traverse_functions(child, functions, language)

    def _extract_function_name(self, node: object, language: str) -> str | None:
        """Extract function name from AST node.

        Args:
            node: The function definition node.
            language: The programming language.

        Returns:
            The function name or None.
        """
        try:
            if hasattr(node, "child_by_field_name"):
                name_node = node.child_by_field_name("name")
                if name_node:
                    return name_node.text.decode()
        except Exception:
            pass
        return None

    def _extract_classes(
        self, tree: object, language: str
    ) -> dict[str, dict]:
        """Extract class definitions from AST.

        Args:
            tree: The parse tree object.
            language: The programming language.

        Returns:
            Dictionary mapping class names to their metadata.
        """
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
        """Traverse AST and collect class definitions.

        Args:
            node: The AST node to traverse.
            classes: Dictionary to accumulate classes.
            language: The programming language.
        """
        if not node:
            return

        # Check for class definition nodes
        if language == "python":
            class_type = "class_definition"
        else:  # javascript, typescript
            class_type = "class_declaration"

        if hasattr(node, "type") and node.type == class_type:
            class_name = self._extract_class_name(node, language)
            if class_name:
                methods = self._extract_class_methods(node, language)
                classes[class_name] = {
                    "start_line": node.start_point[0],
                    "end_line": node.end_point[0],
                    "methods": methods,
                    "node": node,
                }

        # Recursively traverse children
        if hasattr(node, "children"):
            for child in node.children:
                self._traverse_classes(child, classes, language)

    def _extract_class_name(self, node: object, language: str) -> str | None:
        """Extract class name from AST node.

        Args:
            node: The class definition node.
            language: The programming language.

        Returns:
            The class name or None.
        """
        try:
            if hasattr(node, "child_by_field_name"):
                name_node = node.child_by_field_name("name")
                if name_node:
                    return name_node.text.decode()
        except Exception:
            pass
        return None

    def _extract_class_methods(self, node: object, language: str) -> list[str]:
        """Extract method names from a class definition.

        Args:
            node: The class definition node.
            language: The programming language.

        Returns:
            List of method names in the class.
        """
        methods: list[str] = []

        if not hasattr(node, "children"):
            return methods

        try:
            if language == "python":
                method_type = "function_definition"
            else:
                method_type = "function_declaration"

            for child in node.children:
                if hasattr(child, "type") and child.type == method_type:
                    method_name = self._extract_function_name(child, language)
                    if method_name:
                        methods.append(method_name)
        except Exception:
            pass

        return methods

    def _extract_imports(
        self, tree: object, language: str
    ) -> dict[str, dict]:
        """Extract import statements from AST.

        Args:
            tree: The parse tree object.
            language: The programming language.

        Returns:
            Dictionary mapping module names to their metadata.
        """
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
        """Traverse AST and collect import statements.

        Args:
            node: The AST node to traverse.
            imports: Dictionary to accumulate imports.
            language: The programming language.
        """
        if not node:
            return

        try:
            if language == "python":
                import_types = {"import_statement", "import_from_statement"}
            else:  # javascript, typescript
                import_types = {"import_statement"}

            if hasattr(node, "type") and node.type in import_types:
                module = self._extract_import_module(node, language)
                if module:
                    imports[module] = {
                        "line": node.start_point[0],
                        "statement": node.text.decode() if hasattr(node, "text") else "",
                    }

            # Recursively traverse children
            if hasattr(node, "children"):
                for child in node.children:
                    self._traverse_imports(child, imports, language)
        except Exception:
            pass

    def _extract_import_module(self, node: object, language: str) -> str | None:
        """Extract module name from import statement.

        Args:
            node: The import statement node.
            language: The programming language.

        Returns:
            The module name or None.
        """
        try:
            if hasattr(node, "text"):
                statement = node.text.decode()
                if language == "python":
                    if statement.startswith("import"):
                        # import module or from module import
                        parts = statement.split()
                        if "from" in parts:
                            idx = parts.index("from")
                            if idx + 1 < len(parts):
                                return parts[idx + 1].rstrip(".,;")
                        else:
                            idx = parts.index("import")
                            if idx + 1 < len(parts):
                                return parts[idx + 1].rstrip(".,;")
                else:  # javascript, typescript
                    if "from" in statement:
                        # import x from 'module'
                        start = statement.find("'")
                        end = statement.rfind("'")
                        if start != -1 and end > start:
                            return statement[start + 1 : end]
        except Exception:
            pass
        return None

    def _detect_function_changes(
        self,
        functions_before: dict[str, dict],
        functions_after: dict[str, dict],
        content_before: str,
        content_after: str,
        language: str,
    ) -> list[SemanticEvent]:
        """Detect function additions, modifications, and removals.

        Args:
            functions_before: Functions in the before version.
            functions_after: Functions in the after version.
            content_before: The code before changes.
            content_after: The code after changes.
            language: The programming language.

        Returns:
            List of function-related semantic events.
        """
        events: list[SemanticEvent] = []

        # Find added functions
        for name, info in functions_after.items():
            if name not in functions_before:
                # Calculate complexity for the function body
                func_code = self._extract_function_body(
                    content_after, info["start_line"], info["end_line"]
                )
                complexity = calculate_cyclomatic_complexity(func_code)
                events.append(
                    FunctionAdded(
                        name=name,
                        language=language,
                        line_start=info["start_line"],
                        line_end=info["end_line"],
                        cyclomatic_complexity=complexity,
                    )
                )

        # Find removed functions
        for name, info in functions_before.items():
            if name not in functions_after:
                func_code = self._extract_function_body(
                    content_before, info["start_line"], info["end_line"]
                )
                complexity = calculate_cyclomatic_complexity(func_code)
                events.append(
                    FunctionRemoved(
                        name=name,
                        language=language,
                        line_start=info["start_line"],
                        line_end=info["end_line"],
                        cyclomatic_complexity=complexity,
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
                    events.append(
                        FunctionModified(
                            name=name,
                            language=language,
                            line_start_before=before_info["start_line"],
                            line_end_before=before_info["end_line"],
                            line_start_after=after_info["start_line"],
                            line_end_after=after_info["end_line"],
                            complexity_before=complexity_before,
                            complexity_after=complexity_after,
                            changes_summary=f"Function modified from line {before_info['start_line']} to {after_info['start_line']}",
                        )
                    )

        return events

    def _detect_class_changes(
        self,
        classes_before: dict[str, dict],
        classes_after: dict[str, dict],
        language: str,
    ) -> list[SemanticEvent]:
        """Detect class additions, modifications, and removals.

        Args:
            classes_before: Classes in the before version.
            classes_after: Classes in the after version.
            language: The programming language.

        Returns:
            List of class-related semantic events.
        """
        events: list[SemanticEvent] = []

        # Find added classes
        for name, info in classes_after.items():
            if name not in classes_before:
                events.append(
                    ClassAdded(
                        name=name,
                        language=language,
                        line_start=info["start_line"],
                        line_end=info["end_line"],
                        methods=info.get("methods", []),
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
                methods_modified = [
                    m for m in before_methods & after_methods
                ]  # Simplified

                if methods_added or methods_removed or methods_modified:
                    events.append(
                        ClassModified(
                            name=name,
                            language=language,
                            line_start_before=before_info["start_line"],
                            line_end_before=before_info["end_line"],
                            line_start_after=after_info["start_line"],
                            line_end_after=after_info["end_line"],
                            methods_added=methods_added,
                            methods_removed=methods_removed,
                            methods_modified=methods_modified,
                        )
                    )

        return events

    def _detect_import_changes(
        self,
        imports_before: dict[str, dict],
        imports_after: dict[str, dict],
        language: str,
    ) -> list[SemanticEvent]:
        """Detect import additions and removals.

        Args:
            imports_before: Imports in the before version.
            imports_after: Imports in the after version.
            language: The programming language.

        Returns:
            List of import-related semantic events.
        """
        events: list[SemanticEvent] = []

        # Find added imports
        for module, info in imports_after.items():
            if module not in imports_before:
                events.append(
                    ImportAdded(
                        module=module,
                        language=language,
                        line=info.get("line", 0),
                        full_statement=info.get("statement", ""),
                    )
                )

        # Find removed imports
        for module, info in imports_before.items():
            if module not in imports_after:
                events.append(
                    ImportRemoved(
                        module=module,
                        language=language,
                        line=info.get("line", 0),
                        full_statement=info.get("statement", ""),
                    )
                )

        return events

    def _extract_function_body(
        self, content: str, start_line: int, end_line: int
    ) -> str:
        """Extract function body from source code.

        Args:
            content: The full source code.
            start_line: The starting line number.
            end_line: The ending line number.

        Returns:
            The function body as a string.
        """
        lines = content.split("\n")
        # Convert 0-indexed to 1-indexed and extract range
        start = max(0, start_line)
        end = min(len(lines), end_line + 1)
        return "\n".join(lines[start:end])
