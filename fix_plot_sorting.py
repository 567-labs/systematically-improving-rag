#!/usr/bin/env python3
"""Fix plotting code to sort by k value in all affected notebooks."""

import json
import sys
from pathlib import Path


def fix_notebook_cell(source_lines):
    """Add .sort_values("k") to data filtering lines."""
    fixed = []
    for line in source_lines:
        # Look for pattern: data = some_data[...] followed by ]
        if 'data = ' in line and '[' in line and '&' in line:
            # Check if this is a multi-line filter expression
            # We need to add .sort_values("k") after the closing ]
            if line.rstrip().endswith(']'):
                # Single line case
                line = line.rstrip()[:-1] + '].sort_values("k")\n'
            elif line.rstrip().endswith(']\\n",'):
                # Last line of notebook cell (with escaped newline and quote)
                line = line.rstrip()[:-4] + '].sort_values(\\"k\\")\\n",\n'
        elif line.strip() == ']' and fixed and 'data = ' in fixed[-1]:
            # Multi-line case - closing bracket on its own line
            if line.rstrip().endswith('\\n",'):
                line = '].sort_values("k")\\n",\n'
            else:
                line = '].sort_values("k")\n'
        fixed.append(line)
    return fixed


def fix_notebook(notebook_path):
    """Fix all plotting cells in a notebook."""
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)

    modified = False
    for cell in notebook.get('cells', []):
        if cell.get('cell_type') != 'code':
            continue

        source = cell.get('source', [])
        if not source:
            continue

        # Check if this cell contains plotting code with data["k"]
        source_text = ''.join(source)
        if 'ax1.plot' in source_text or 'ax2.plot' in source_text or 'plt.plot' in source_text:
            if 'data["k"]' in source_text and '.sort_values' not in source_text:
                # This cell needs fixing
                fixed_source = fix_notebook_cell(source)
                cell['source'] = fixed_source
                modified = True

    if modified:
        with open(notebook_path, 'w') as f:
            json.dump(notebook, f, indent=1, ensure_ascii=False)
        print(f"Fixed: {notebook_path}")
        return True
    else:
        print(f"No changes needed: {notebook_path}")
        return False


def main():
    notebooks = [
        "cohort_2/week1/2. benchmark_retrieval.ipynb",
        "cohort_2/week1/2. benchmark_retrieval_logfire.ipynb",
    ]

    root = Path("/Users/jasonliu/dev/systematically-improving-rag")

    fixed_count = 0
    for notebook_path in notebooks:
        full_path = root / notebook_path
        if full_path.exists():
            if fix_notebook(full_path):
                fixed_count += 1
        else:
            print(f"Not found: {full_path}", file=sys.stderr)

    print(f"\nFixed {fixed_count} notebooks")


if __name__ == "__main__":
    main()
