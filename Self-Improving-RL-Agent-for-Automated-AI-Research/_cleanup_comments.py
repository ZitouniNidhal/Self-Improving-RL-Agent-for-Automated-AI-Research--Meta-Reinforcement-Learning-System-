import ast
import io
import os
import tokenize

root = os.path.dirname(os.path.abspath(__file__))

for dirpath, _, filenames in os.walk(root):
    for filename in filenames:
        if not filename.endswith('.py'):
            continue
        if filename == os.path.basename(__file__):
            continue
        path = os.path.join(dirpath, filename)
        with open(path, 'r', encoding='utf-8') as f:
            src = f.read()
        try:
            tree = ast.parse(src)
        except SyntaxError:
            continue

        lines = src.splitlines(True)
        offsets = [0]
        for line in lines:
            offsets.append(offsets[-1] + len(line))

        def pos_to_index(line, col):
            return offsets[line - 1] + col

        ranges = []
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                body = getattr(node, 'body', None)
                if not body:
                    continue
                first = body[0]
                if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
                    start = pos_to_index(first.lineno, first.col_offset)
                    end = pos_to_index(first.end_lineno, first.end_col_offset)
                    ranges.append((start, end))

        for start, end in sorted(ranges, reverse=True):
            src = src[:start] + ' ' * (end - start) + src[end:]

        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
        cleaned = tokenize.untokenize([tok for tok in tokens if tok.type != tokenize.COMMENT])
        with open(path, 'w', encoding='utf-8', newline='') as f:
            f.write(cleaned)
