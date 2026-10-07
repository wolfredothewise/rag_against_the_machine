import ast
from typing import List, Tuple


class SuperChunker:
    """Split source files into smaller chunks for later indexing and search.

    The class centralizes the logic required to split Python and Markdown
    documents into consistent segments that can be indexed by the retrieval
    pipeline. It keeps a shared list of chunk boundaries in character offsets,
    which are later consumed by the search index.
    """

    def __init__(self) -> None:
        """Initialize the chunk accumulator and auxiliary metadata.

        This stores the generated chunk ranges and a separate Python-specific
        mapping structure for downstream processing.
        """
        self.all_chunks: List[Tuple[str, int, int]] = []
        self.map_py: List[Tuple[str, int, int]] = []

    def chunk_orchestator(
        self,
        f_path: str,
        max_chunk_size: int,
    ) -> None:
        """Route a file to the appropriate chunking strategy.

        Args:
            f_path: Path to the source file that needs to be
                chunked.
            max_chunk_size: Maximum size, in characters, allowed for each
                chunk.

        Raises:
            ValueError: If the file extension is not supported by the
                chunker.
        """
        f = f_path.lower()
        if f.endswith(".py"):
            self.chunk_python_file(f_path, max_chunk_size)
        elif f.endswith((".md", ".markdown")):
            self.chunkeator_md(f_path, max_chunk_size)
        else:
            raise ValueError(f"ERROR: Unsupported format for file: {f_path}")

    def chunk_python_file(
        self,
        file_path: str,
        max_chunk_size: int = 2000,
    ) -> None:
        """Chunk a Python source file by traversing its AST.

        The method reads the file, parses it with Python's AST module, and
        creates chunk boundaries around statements, classes, and functions.
        When an AST node is too large for the configured maximum size, it falls
        back to a recursive split at newline boundaries.

        Args:
            file_path: Path to the Python source file to process.
            max_chunk_size: Maximum number of characters allowed for each
                generated chunk.

        Raises:
            FileNotFoundError: If the target Python file does not exist.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                file_py = file.read()
                tree = ast.parse(file_py)

                for nodo in tree.body:
                    size, fci, lci = self._line_to_char(
                        file_py,
                        nodo.lineno,
                        nodo.end_lineno,
                    )

                    if isinstance(nodo, (ast.Import, ast.ImportFrom)):
                        if size <= max_chunk_size:
                            self.all_chunks.append((file.name, fci, lci))
                        else:
                            self.fallback_last_line(
                                file_py,
                                fci,
                                lci,
                                max_chunk_size,
                                file.name,
                            )

                    elif isinstance(nodo, ast.ClassDef):
                        if size <= max_chunk_size:
                            self.all_chunks.append((file.name, fci, lci))
                        else:
                            for method in nodo.body:
                                if isinstance(
                                    method,
                                    (ast.FunctionDef, ast.AsyncFunctionDef),
                                ):
                                    method_size, m_fci, m_lci = (
                                        self._line_to_char(
                                            file_py,
                                            method.lineno,
                                            method.end_lineno,
                                        )
                                    )

                                    if method_size < max_chunk_size:
                                        self.all_chunks.append(
                                            (file.name, m_fci, m_lci)
                                        )
                                    else:
                                        self.fallback_last_line(
                                            file_py,
                                            m_fci,
                                            m_lci,
                                            max_chunk_size,
                                            file.name,
                                        )

                    elif isinstance(
                        nodo,
                        (ast.FunctionDef, ast.AsyncFunctionDef),
                    ):
                        if size < max_chunk_size:
                            self.all_chunks.append((file.name, fci, lci))
                        else:
                            self.fallback_last_line(
                                file_py,
                                fci,
                                lci,
                                max_chunk_size,
                                file.name,
                            )

                    else:
                        if size <= max_chunk_size:
                            self.all_chunks.append((file.name, fci, lci))
                        else:
                            self.fallback_last_line(
                                file_py,
                                fci,
                                lci,
                                max_chunk_size,
                                file.name,
                            )

        except FileNotFoundError as exc:
            print("Archive not found in storage", exc)

    def _line_to_char(
        self,
        raw_text_py: str,
        lineno: int,
        end_lineno: int | None,
    ) -> tuple[int, int, int]:
        """Convert a line interval into character offsets in the raw text.

        This is used to translate AST node positions back into absolute offsets
        inside the original file content, so the chunk boundaries correspond to
        actual character spans rather than only line numbers.

        Args:
            raw_text_py: Full source text of the Python file.
            lineno: Starting line number of the node.
            end_lineno: Ending line number of the node.

        Returns:
            A tuple with the following values:
                - size: length of the node in characters,
                - first_character_index: start offset in the file,
                - last_character_index: end offset in the file.
        """
        lines = raw_text_py.splitlines(keepends=True)
        first_character_index = len(
            "".join(lines[:lineno - 1])
        )
        last_character_index = len("".join(lines[:end_lineno]))
        return (
            last_character_index - first_character_index,
            first_character_index,
            last_character_index,
        )

    def chunkeator_md(
        self,
        file_path: str,
        max_chunk_size: int = 2000,
    ) -> None:
        """Chunk a Markdown file into paragraph-based segments.

        The method splits the content by blank-line paragraph separators
        and applies a fallback strategy when a paragraph exceeds the
        configured maximum size. This ensures longer Markdown documents still
        produce valid chunks without losing structural readability.

        Args:
            file_path: Path to the Markdown file to process.
            max_chunk_size: Maximum number of characters allowed for each
                generated chunk.

        Raises:
            FileNotFoundError: If the Markdown file cannot be found.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            start_chunk = 0
            p_start = 0

            paragraphs = content.split("\n\n")

            for p in paragraphs:
                p_end = p_start + len(p)

                if (p_end - p_start) > max_chunk_size:
                    if p_start > start_chunk:
                        self.all_chunks.append(
                            (file_path, start_chunk, p_start - 2)
                        )

                    self.fallback_md(
                        content,
                        p_start,
                        p_end,
                        max_chunk_size,
                        file_path,
                    )
                    start_chunk = p_end + 2

                elif (p_end - start_chunk) > max_chunk_size:
                    self.all_chunks.append(
                        (file_path, start_chunk, p_start - 2)
                    )
                    start_chunk = p_start

                p_start = p_end + 2

            if start_chunk < len(content):
                self.all_chunks.append(
                    (file_path, start_chunk, len(content))
                )
        except FileNotFoundError as exc:
            print("RESPONSE: Archive not found in storage matrix", exc)

    def fallback_last_line(
        self,
        file_py: str,
        fci: int,
        lci: int,
        max_chunk_size: int,
        file_path: str,
    ) -> None:
        """Recursively split a large Python block on newline boundaries.

        This fallback is used when a block is larger than the configured
        maximum size and cannot be safely kept as a single chunk. It walks
        backward from the maximum allowed boundary until it finds a newline and
        then continues recursively on the remaining segment.

        Args:
            file_py: Full contents of the Python file.
            fci: Starting character index of the current block.
            lci: Ending character index of the current block.
            max_chunk_size: Maximum allowed size for a chunk.
            file_path: File path associated with the chunk.
        """
        if (lci - fci) <= max_chunk_size:
            self.all_chunks.append((file_path, fci, lci))
            return

        for i in range(fci + max_chunk_size - 1, fci, -1):
            if file_py[i] == "\n":
                limit = i
                self.all_chunks.append((file_path, fci, limit))
                self.fallback_last_line(
                    file_py,
                    limit + 1,
                    lci,
                    max_chunk_size,
                    file_path,
                )
                return

    def fallback_md(
        self,
        content: str,
        fci: int,
        lci_limit: int,
        max_chunk_size: int,
        file_path: str,
    ) -> None:
        """Split a large Markdown segment around natural sentence boundaries.

        When a paragraph exceeds the configured size, this method tries to
        cut at punctuation such as periods, commas, or newlines to preserve
        readable chunks. If no punctuation is found, it falls back to a plain
        fixed-size boundary.

        Args:
            content: Full Markdown content of the file.
            fci: Starting character index of the current segment.
            lci_limit: Upper character limit for the segment.
            max_chunk_size: Maximum allowed chunk length in characters.
            file_path: File path associated with the chunk.
        """
        while fci < lci_limit:
            lci = min(fci + max_chunk_size, lci_limit)

            if lci == lci_limit:
                self.all_chunks.append((file_path, fci, lci_limit))
                break

            cut = max(
                content.rfind(".", fci, lci),
                content.rfind(",", fci, lci),
                content.rfind("\n", fci, lci),
            )

            if cut != -1 and cut >= fci:
                self.all_chunks.append((file_path, fci, cut + 1))
                fci = cut + 1
            else:
                self.all_chunks.append((file_path, fci, lci))
                fci = lci
