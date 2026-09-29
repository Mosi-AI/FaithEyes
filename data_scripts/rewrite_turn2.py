#!/usr/bin/env python3
"""
Rewrite turn[2] code in len(response)==5 SFT data so that each code block
is FULLY STANDALONE: correct imports, variables, image loading, and absolute coordinates.

Background:
  In len=5 data, the conversation is:
    turn[0] (assistant): thinking + code (crop on original image, gets REJECT)
    turn[1] (user):      sandbox_output [REJECT]
    turn[2] (assistant): thinking + code (refined crop, gets ACCEPT)
    turn[3] (user):      sandbox_output [ACCEPT]
    turn[4] (assistant): final answer

  turn[2] code has three problems for RL:
    1. It often loads from `processed_path` (turn[0]'s output) → coords are
       RELATIVE to the first crop, meaningless on the original image.
    2. It inherits variables (`image`, `temp_dir`, `filename`, etc.) from turn[0]'s
       code block, which don't exist when RL runs each code block independently.
    3. It inherits imports (`os`, `cv2`, `uuid4`, etc.) from turn[0].

  This script rewrites turn[2] code to be fully standalone:
    - Carry over missing imports from turn[0].
    - Carry over missing variable definitions from turn[0] (temp_dir, filename, etc.).
    - Category A (loads processed_path): replace with image_path loading,
      convert relative crop coords to absolute (offset + relative).
    - Category B (loads image_path): already standalone, just add missing deps.
    - Category C (uses inherited `image` var): add image loading + missing deps.
    - Category D / unparseable: skip.

Usage:
    python3 rewrite_turn2.py
"""

import ast
import json
import re
import os
import sys
from typing import Optional, Tuple, List, Set

# ============================================================================
# Configuration
# ============================================================================
INPUT_FILE = '/path/to/input.jsonl'
OUTPUT_FILE = '/path/to/output.jsonl'
STATS_FILE = '/path/to/rewrite_stats.txt'

NUM = r'-?\d+(?:\.\d+)?'

# ============================================================================
# Coordinate Extraction
# ============================================================================

def extract_coords(code: str) -> Optional[Tuple[float, float, float, float]]:
    """
    Extract crop coordinates from code as (y1, y2, x1, x2).

    Handles 5 patterns:
      1. numpy slice:   var[y1:y2, x1:x2]
      2. PIL crop:      .crop((x1, y1, x2, y2))
      3. var assign:    *x1, *y1, *x2, *y2 = n, n, n, n  (then var[y1:y2, x1:x2])
      4. tuple var:     var = (x1, y1, x2, y2); .crop(var)
      5. tuple bare:    var = (x1, y1, x2, y2)  (used elsewhere)
    """
    # 1. numpy slice: var[y1:y2, x1:x2]
    m = re.findall(
        r'\w+\s*\[\s*(' + NUM + r')\s*:\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*:\s*(' + NUM + r')\s*\]',
        code)
    if m:
        return (float(m[0][0]), float(m[0][1]), float(m[0][2]), float(m[0][3]))

    # 2. PIL crop literal: .crop((x1, y1, x2, y2))
    m = re.search(
        r'\.crop\(\s*\(\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*\)',
        code)
    if m:
        return (float(m.group(2)), float(m.group(4)), float(m.group(1)), float(m.group(3)))

    # 3. Variable assignment: *x1, *y1, *x2, *y2 = n, n, n, n
    m = re.search(
        r'\w*x1\s*,\s*\w*y1\s*,\s*\w*x2\s*,\s*\w*y2\s*=\s*'
        r'(' + NUM + r')\s*,\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*,\s*(' + NUM + r')',
        code)
    if m:
        return (float(m.group(2)), float(m.group(4)), float(m.group(1)), float(m.group(3)))

    # 4. Tuple variable: var = (x1, y1, x2, y2); .crop(var)
    tuples = re.findall(
        r'(\w+)\s*=\s*\(\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*,\s*(' + NUM + r')\s*\)',
        code)
    for varname, x1, y1, x2, y2 in tuples:
        if re.search(r'\.crop\(\s*' + re.escape(varname), code):
            return (float(y1), float(y2), float(x1), float(x2))

    # 5. Bare tuple: var = (x1, y1, x2, y2)
    if tuples:
        return (float(tuples[0][2]), float(tuples[0][4]), float(tuples[0][1]), float(tuples[0][3]))

    return None


# ============================================================================
# Image Source Classification
# ============================================================================

PROCESSED_PATH_NAMES = [
    'processed_path', 'cropped_image_path', 'initial_cropped_path',
    'first_cropped_path', 'cropped_path', 'first_cropped_image_path',
]

def classify_turn2(c2: str) -> str:
    """
    Classify how turn[2] gets its image.

    Returns:
      'A' - loads from processed_path (RELATIVE coords, needs offset)
      'B' - loads from image_path (ABSOLUTE coords, already standalone)
      'C' - uses inherited `image` var (ABSOLUTE coords, needs image loading)
      'D' - other / unparseable
    """
    # Check if loads from processed_path variants
    pp_pattern = r'(?:imread|Image\.open)\s*\(\s*(?:' + '|'.join(
        re.escape(n) for n in PROCESSED_PATH_NAMES) + r')'
    if re.search(pp_pattern, c2, re.I):
        return 'A'

    # Check if loads from image_path
    if re.search(r'(?:imread|Image\.open)\s*\(\s*image_path', c2):
        return 'B'

    # Check if uses bare `image` variable in crop (inherited from turn[0])
    if re.search(r'\bimage\s*\[', c2):
        return 'C'

    # Also check for PIL .crop() on inherited `image` var
    if re.search(r'\bimage\.crop\s*\(', c2) or re.search(r'\bimg\.crop\s*\(', c2):
        return 'C'

    return 'D'


# ============================================================================
# Dependency Analysis
# ============================================================================

PYTHON_KEYWORDS = {
    'if', 'else', 'elif', 'for', 'while', 'def', 'class', 'return', 'import',
    'from', 'as', 'in', 'not', 'and', 'or', 'is', 'None', 'True', 'False',
    'with', 'try', 'except', 'finally', 'raise', 'pass', 'break', 'continue',
    'lambda', 'global', 'nonlocal', 'yield', 'del', 'assert', 'print',
}

PYTHON_BUILTINS = {
    'str', 'int', 'float', 'len', 'range', 'open', 'list', 'dict',
    'tuple', 'set', 'abs', 'min', 'max', 'sum', 'sorted', 'enumerate',
    'zip', 'map', 'round', 'isinstance', 'filter', 'any', 'all',
}

# Variables that are provided by the RL sandbox at execution time.
# NOTE: image_path is intentionally NOT here. In SFT data, image_path is
# hardcoded in turn[0] (image_path = "/.../xxx.jpg"), and the model
# learns to write it. We carry this definition to turn[2] so the code is
# self-contained and the model learns the same pattern.
SANDBOX_VARS = {'temp_output_dir'}


def get_import_lines(code: str) -> List[str]:
    """Extract import statements from code, preserving order."""
    lines = []
    for m in re.finditer(r'^\s*((?:import|from)\s+.+)$', code, re.MULTILINE):
        lines.append(m.group(1).strip())
    return lines


def _ast_parse_safe(code: str):
    """Parse code with AST, return None on failure."""
    try:
        return ast.parse(code)
    except SyntaxError:
        return None


def get_defined_names_ast(code: str) -> Set[str]:
    """
    Get all names defined in code via AST (imports, assignments, tuple unpacking).
    Properly handles f-strings and tuple unpacking like `x1, y1 = 0, 0`.
    """
    defined = set()
    tree = _ast_parse_safe(code)
    if tree is None:
        return defined

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                defined.add(alias.asname or alias.name.split('.')[0])
        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                defined.add(alias.asname or alias.name)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    defined.add(target.id)
                elif isinstance(target, ast.Tuple):
                    for elt in target.elts:
                        if isinstance(elt, ast.Name):
                            defined.add(elt.id)
        elif isinstance(node, ast.AnnAssign):
            if isinstance(node.target, ast.Name):
                defined.add(node.target.id)
    defined.update(SANDBOX_VARS)
    return defined


def get_used_names_ast(code: str) -> Set[str]:
    """
    Get all Name nodes that are loaded (true variable references) via AST.
    Properly captures names inside f-strings (JoinedStr) and nested expressions.
    """
    used = set()
    tree = _ast_parse_safe(code)
    if tree is None:
        return used

    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            used.add(node.id)
    return used


def get_assignment_lines(code: str) -> dict:
    """
    Get variable assignments with their full source lines.
    Handles both simple assignments (var = ...) and tuple unpacking (a, b = ...).
    Returns {var_name: (full_line, value_string)}.
    """
    assigns = {}
    lines = code.split('\n')
    for line in lines:
        stripped = line.strip()
        # Skip comments and empty lines
        if not stripped or stripped.startswith('#'):
            continue
        # Skip imports
        if stripped.startswith('import ') or stripped.startswith('from '):
            continue

        # Try to parse as assignment
        m = re.match(r'^(\w+(?:\s*,\s*\w+)*)\s*=\s*(.+)$', stripped)
        if m:
            var_part = m.group(1)
            val = m.group(2).strip()
            # Split tuple unpacking: "x1, y1, x2, y2" → individual names
            var_names = [v.strip() for v in var_part.split(',')]
            for var in var_names:
                assigns[var] = (stripped, val)
    return assigns


def compute_missing_deps(c0: str, c2: str) -> Tuple[List[str], List[str]]:
    """
    Given turn[0] code and turn[2] code, compute what turn[2] is missing.
    Uses AST for accurate name resolution (handles f-strings, tuple unpacking).
    Runs iteratively: carried-over variables may introduce new import dependencies.

    Returns:
      (missing_import_lines, missing_variable_lines)
      - missing_import_lines: import statements from turn[0] that turn[2] needs
      - missing_variable_lines: assignment statements from turn[0] that turn[2] needs
        (excludes image, processed_path, and variables that depend on them)
    """
    t0_imports = get_import_lines(c0)
    t0_imports_set = set(t0_imports)
    t0_assigns = get_assignment_lines(c0)

    t2_defined = get_defined_names_ast(c2)
    t2_used = get_used_names_ast(c2)

    # Names used in t2 but not defined in t2
    missing_names = t2_used - t2_defined - PYTHON_BUILTINS

    EXCLUDED_VARS = {'image', 'img', 'processed_path', 'cropped_image',
                     'first_cropped_path', 'initial_cropped_path', 'cropped_image_path',
                     'first_cropped_image', 'initial_cropped_image'}

    missing_import_lines = []
    missing_var_lines = []
    seen_imports = set()
    seen_vars = set()

    # Iterative: carried-over variables may introduce new name dependencies
    # (e.g., filename = os.path.basename(image_path) introduces 'os' dependency)
    for _ in range(5):  # max 5 iterations to avoid infinite loop
        # Find missing imports from turn[0]
        for imp_line in t0_imports:
            if imp_line in seen_imports:
                continue
            imp_aliases = set()
            m1 = re.match(r'import\s+(\S+)(?:\s+as\s+(\w+))?', imp_line)
            if m1:
                mod = m1.group(1).split('.')[0]
                alias = m1.group(2) or mod
                imp_aliases.add(alias)
            m2 = re.match(r'from\s+\S+\s+import\s+(.+)', imp_line)
            if m2:
                for name in m2.group(1).split(','):
                    name = name.strip()
                    if ' as ' in name:
                        name = name.split(' as ')[-1].strip()
                    if name:
                        imp_aliases.add(name)

            if imp_aliases & missing_names and not (imp_aliases & t2_defined):
                missing_import_lines.append(imp_line)
                seen_imports.add(imp_line)
                t2_defined.update(imp_aliases)

        # Find missing variables from turn[0]
        new_vars_added = False
        for name in list(missing_names):
            if name in EXCLUDED_VARS or name in seen_vars:
                continue
            if name in t0_assigns:
                full_line, val = t0_assigns[name]
                if any(ex in val for ex in ('processed_path', 'cropped_image', 'first_cropped')):
                    continue
                missing_var_lines.append(full_line)
                seen_vars.add(name)
                t2_defined.add(name)
                # The carried-over variable may use new names (e.g., os, basename)
                # Add those to missing_names for the next iteration
                var_used = get_used_names_ast(full_line)
                new_missing = var_used - t2_defined - PYTHON_BUILTINS
                if new_missing:
                    missing_names.update(new_missing)
                    new_vars_added = True
            elif name == 'image_path':
                # image_path might be defined under a different name in turn[0]
                # (e.g., original_image_path, input_path). Find any path-like
                # variable and create an image_path alias.
                for var_name, (full_line, val) in t0_assigns.items():
                    if var_name in seen_vars or var_name in EXCLUDED_VARS:
                        continue
                    # Check if this is a path string assignment
                    if re.match(r'^["\']', val) and re.search(r'\.(?:jpg|jpeg|png|bmp|gif|tiff)["\']$', val, re.I):
                        missing_var_lines.append(
                            f'image_path = {val}  # rewritten: carry over from turn[0] ({var_name})')
                        seen_vars.add('image_path')
                        t2_defined.add('image_path')
                        new_vars_added = True
                        break

        # Recompute missing names after adding imports and vars
        missing_names = t2_used - t2_defined - PYTHON_BUILTINS
        # Also include names used by carried-over variables
        for vline in missing_var_lines:
            missing_names.update(get_used_names_ast(vline) - t2_defined - PYTHON_BUILTINS)

        if not new_vars_added:
            break

    return missing_import_lines, missing_var_lines


# ============================================================================
# Code Rewriting
# ============================================================================

def _fmt_num(n: float) -> str:
    """Format a number: int if whole, else float."""
    if n == int(n):
        return str(int(n))
    return str(n)


def rewrite_category_a(c2: str, t0: tuple, t2: tuple) -> str:
    """
    Category A: turn[2] loads from processed_path with RELATIVE coords.
    Rewrite to: load original image_path, use ABSOLUTE coords.

    absolute = turn[0]_offset + turn[2]_relative
    """
    y1_0, y2_0, x1_0, x2_0 = t0
    y1_2, y2_2, x1_2, x2_2 = t2

    abs_y1 = y1_0 + y1_2
    abs_y2 = y1_0 + y2_2
    abs_x1 = x1_0 + x1_2
    abs_x2 = x1_0 + x2_2

    new_code = c2

    # Detect whether turn[2] uses PIL or cv2 style
    uses_pil = 'Image.open' in new_code or '.crop(' in new_code or 'Image' in new_code
    uses_cv2 = 'cv2' in new_code

    # Determine the correct image loading expression
    if uses_pil and not uses_cv2:
        loader_expr = 'Image.open(image_path)  # rewritten: load original image'
    elif uses_cv2 and not uses_pil:
        loader_expr = 'cv2.imread(image_path)  # rewritten: load original image'
    elif uses_cv2 and uses_pil:
        # Mixed: if the processed_path load was via Image.open, keep PIL
        if 'Image.open' in new_code:
            loader_expr = 'Image.open(image_path)  # rewritten: load original image'
        else:
            loader_expr = 'cv2.imread(image_path)  # rewritten: load original image'
    else:
        loader_expr = 'cv2.imread(image_path)  # rewritten: load original image'

    # Step 1: Replace processed_path loading with image_path loading
    # Remove lines like: var = processed_path
    for pp_name in PROCESSED_PATH_NAMES:
        new_code = re.sub(
            r'^\s*\w+\s*=\s*' + re.escape(pp_name) + r'\s*$\n',
            '',
            new_code, flags=re.MULTILINE)
        # Replace imread(processed_path_var) with imread(image_path)
        new_code = re.sub(
            r'(?:cv2\.imread|Image\.open)\s*\(\s*' + re.escape(pp_name) + r'\s*\)',
            loader_expr,
            new_code)

    # Also handle: var = processed_path; var2 = imread(var)
    # Remove intermediate variable assignments that just alias processed_path
    for pp_name in PROCESSED_PATH_NAMES:
        new_code = re.sub(
            r'^\s*\w+\s*=\s*' + re.escape(pp_name) + r'\s*\n',
            '',
            new_code, flags=re.MULTILINE)

    # Step 1b: Replace any remaining references to removed path variables with image_path
    # (e.g., os.path.basename(cropped_image_path) → os.path.basename(image_path))
    for pp_name in PROCESSED_PATH_NAMES:
        new_code = re.sub(r'\b' + re.escape(pp_name) + r'\b', 'image_path', new_code)

    # Step 2: Replace crop coordinates with absolute ones
    # numpy slice: var[y1:y2, x1:x2]
    old_slice = f'{_fmt_num(y1_2)}:{_fmt_num(y2_2)}, {_fmt_num(x1_2)}:{_fmt_num(x2_2)}'
    new_slice = f'{_fmt_num(abs_y1)}:{_fmt_num(abs_y2)}, {_fmt_num(abs_x1)}:{_fmt_num(abs_x2)}'
    new_code = new_code.replace(old_slice, new_slice)

    # PIL crop: .crop((x1, y1, x2, y2))
    old_pil = f'({_fmt_num(x1_2)}, {_fmt_num(y1_2)}, {_fmt_num(x2_2)}, {_fmt_num(y2_2)})'
    new_pil = f'({_fmt_num(abs_x1)}, {_fmt_num(abs_y1)}, {_fmt_num(abs_x2)}, {_fmt_num(abs_y2)})'
    new_code = new_code.replace(old_pil, new_pil)

    # Also handle float variants (e.g. 39.05000000000001)
    old_pil_f = f'({x1_2}, {y1_2}, {x2_2}, {y2_2})'
    new_pil_f = f'({_fmt_num(abs_x1)}, {_fmt_num(abs_y1)}, {_fmt_num(abs_x2)}, {_fmt_num(abs_y2)})'
    new_code = new_code.replace(old_pil_f, new_pil_f)

    # var assign: x1, y1, x2, y2 = v1, v2, v3, v4
    old_assign = f'{_fmt_num(x1_2)}, {_fmt_num(y1_2)}, {_fmt_num(x2_2)}, {_fmt_num(y2_2)}'
    new_assign = f'{_fmt_num(abs_x1)}, {_fmt_num(abs_y1)}, {_fmt_num(abs_x2)}, {_fmt_num(abs_y2)}'
    new_code = new_code.replace(old_assign, new_assign)

    # Step 3: If the code uses `image` or `img` as a variable name (inherited from
    # turn[0]) but the image was loaded into a different variable name (e.g.,
    # `cropped_image = cv2.imread(image_path)`), add an alias so the inherited
    # references work. This is needed because turn[2] code may do `image.crop(...)`
    # while the load line assigns to `cropped_image`.
    for inherited_name in ('image', 'img'):
        # Check if this name is used as a variable (not just in strings/comments)
        # Simple check: appears as a standalone word followed by [ or . or as function arg
        if re.search(r'\b' + inherited_name + r'\s*[\[\.]', new_code):
            # Check if it's defined (either via import, assignment, or the loader line)
            # The loader line is: var = cv2.imread(image_path) or var = Image.open(image_path)
            # If `inherited_name` is not assigned anywhere, add alias after the load line
            has_assignment = bool(re.search(r'^\s*' + inherited_name + r'\s*=', new_code, re.MULTILINE))
            if not has_assignment:
                # Find the load variable name
                load_match = re.search(
                    r'(\w+)\s*=\s*(?:cv2\.imread|Image\.open)\s*\(\s*image_path',
                    new_code)
                if load_match and load_match.group(1) != inherited_name:
                    load_var = load_match.group(1)
                    # Add alias right after the load line
                    new_code = re.sub(
                        r'(' + re.escape(load_match.group(0)) + r'[^\n]*)',
                        r'\1\n' + inherited_name + ' = ' + load_var + '  # rewritten: alias for inherited variable',
                        new_code, count=1)

    return new_code


def rewrite_category_c(c2: str) -> str:
    """
    Category C: turn[2] uses inherited `image` var (original image from turn[0]).
    Coords are already absolute, but the code isn't standalone (no image loading).
    Prepend: image = cv2.imread(image_path)  (or Image.open for PIL code)
    Also add `img = image` alias if the code uses `img` (common inherited name).
    """
    uses_cv2 = 'cv2' in c2
    uses_pil = 'Image' in c2 or '.crop(' in c2

    if uses_cv2:
        loader = 'image = cv2.imread(image_path)  # rewritten: standalone image loading'
    elif uses_pil:
        loader = 'image = Image.open(image_path)  # rewritten: standalone image loading'
    else:
        loader = 'image = cv2.imread(image_path)  # rewritten: standalone image loading'

    prefix = loader

    # If the code uses `img` as a variable (inherited from turn[0]), add alias
    if re.search(r'\bimg\s*[\[\.]', c2):
        prefix += '\nimg = image  # rewritten: alias for inherited variable'

    return prefix + '\n\n' + c2


def _toposort_var_lines(var_lines: List[str]) -> List[str]:
    """
    Topologically sort variable assignment lines so that definitions
    appear before uses. E.g., image_path = "..." must come before
    filename = os.path.basename(image_path).

    Uses a simple heuristic: for each line, extract the variable name
    being defined and all names referenced in the value. A line must
    come after any line that defines a name it references.
    """
    if not var_lines:
        return var_lines

    # Parse each line: (var_names_defined, set_of_names_referenced, full_line)
    parsed = []
    for line in var_lines:
        # Match both simple assignment (var = ...) and tuple unpacking (a, b = ...)
        m = re.match(r'^\s*(\w+(?:\s*,\s*\w+)*)\s*=\s*(.+)$', line.strip())
        if not m:
            # Unparseable — keep as-is, no deps
            parsed.append((set(), set(), line))
            continue
        var_part = m.group(1)
        val = m.group(2)
        # Names being defined (support tuple unpacking: "a, b" → {a, b})
        var_names = set(v.strip() for v in var_part.split(','))
        # Remove string literals before extracting identifiers, so that
        # path strings like "/path/file.jpg" don't
        # produce false identifier dependencies (jpg, etc.)
        val_no_strings = re.sub(r'(["\']).*?\1', '', val)
        val_names = set(re.findall(r'\b([a-zA-Z_]\w*)\b', val_no_strings))
        # Remove the vars being defined and Python builtins/keywords
        val_names -= var_names
        val_names -= PYTHON_KEYWORDS
        val_names -= PYTHON_BUILTINS
        parsed.append((var_names, val_names, line))

    # Build dependency graph: line i depends on line j if j defines a name that i uses
    # Use iterative approach: repeatedly pick lines whose deps are all satisfied
    result = []
    defined = set()
    remaining = list(range(len(parsed)))

    for _ in range(len(parsed) + 1):
        progress = False
        for i in remaining[:]:
            var_names, val_names, line = parsed[i]
            # Check if all deps are satisfied (or defined elsewhere, e.g., in imports)
            unresolved = val_names - defined
            if not unresolved:
                result.append(line)
                defined.update(var_names)
                remaining.remove(i)
                progress = True
        if not progress:
            break

    # Any remaining lines (circular deps or unsatisfiable) — append as-is
    for i in remaining:
        result.append(parsed[i][2])

    return result


def add_missing_deps(c2: str, c0: str) -> str:
    """
    Analyze dependencies and prepend missing imports and variables to turn[2] code.
    Variables are topologically sorted so definitions precede uses.
    """
    missing_imports, missing_vars = compute_missing_deps(c0, c2)

    if not missing_imports and not missing_vars:
        return c2

    prefix_parts = []

    if missing_imports:
        # Deduplicate while preserving order
        seen = set()
        unique_imports = []
        for imp in missing_imports:
            if imp not in seen:
                seen.add(imp)
                unique_imports.append(imp)
        prefix_parts.append('\n'.join(unique_imports))

    if missing_vars:
        # Deduplicate
        seen = set()
        unique_vars = []
        for var_line in missing_vars:
            if var_line not in seen:
                seen.add(var_line)
                unique_vars.append(var_line)
        # Topologically sort: image_path = "..." before filename = os.path.basename(image_path)
        sorted_vars = _toposort_var_lines(unique_vars)
        prefix_parts.append('\n'.join(sorted_vars))

    prefix = '\n'.join(prefix_parts)
    return prefix + '\n\n' + c2


# ============================================================================
# Main Processing
# ============================================================================

def extract_code_block(text: str) -> Optional[str]:
    """Extract the first ```python ... ``` code block from text."""
    m = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    if m:
        return m.group(1)
    return None


def replace_code_block(text: str, new_code: str) -> str:
    """Replace the first ```python ... ``` code block in text with new_code."""
    def _repl(_match):
        return f'```python\n{new_code}```'
    return re.sub(
        r'```python\n.*?```',
        _repl,
        text,
        count=1,
        flags=re.DOTALL)


def process_record(record: dict) -> Tuple[Optional[dict], str]:
    """
    Process a single len=5 record.

    Returns:
      (rewritten_record, category) or (None, skip_reason)
    """
    resp = record.get('response', [])
    if isinstance(resp, str):
        resp = [resp]

    if len(resp) != 5:
        return None, 'not_len5'

    # Extract code from turn[0] and turn[2]
    c0_raw = extract_code_block(str(resp[0]))
    c2_raw = extract_code_block(str(resp[2]))

    if not c0_raw or not c2_raw:
        return None, 'no_code_block'

    # Extract coordinates
    t0 = extract_coords(c0_raw)
    t2 = extract_coords(c2_raw)

    if not t0 or not t2:
        return None, 'coord_extract_fail'

    # Classify turn[2]
    category = classify_turn2(c2_raw)

    if category == 'D':
        return None, 'category_D_skip'

    # Step 1: Rewrite based on category (coordinate + image loading)
    if category == 'A':
        new_c2 = rewrite_category_a(c2_raw, t0, t2)
    elif category == 'B':
        new_c2 = c2_raw
    elif category == 'C':
        new_c2 = rewrite_category_c(c2_raw)
    else:
        return None, 'unknown_category'

    # Step 1b: Global cleanup — replace any remaining references to removed
    # path variables (processed_path, cropped_image_path, etc.) with image_path.
    # This handles cases like os.path.basename(processed_path) that survive
    # the category-specific rewrite.
    for pp_name in PROCESSED_PATH_NAMES:
        new_c2 = re.sub(r'\b' + re.escape(pp_name) + r'\b', 'image_path', new_c2)

    # Step 2: Add missing imports and variables from turn[0]
    new_c2 = add_missing_deps(new_c2, c0_raw)

    # Replace code block in turn[2]
    new_resp2 = replace_code_block(str(resp[2]), new_c2)

    # Build new record
    new_record = dict(record)
    new_resp = list(resp)
    new_resp[2] = new_resp2
    new_record['response'] = new_resp
    new_record['_rewrite_category'] = category

    return new_record, category


def main():
    if not os.path.exists(INPUT_FILE):
        print(f'ERROR: Input file not found: {INPUT_FILE}')
        sys.exit(1)

    # Statistics
    stats = {
        'total': 0,
        'saved': 0,
        'len5': 0,
        'rewritten': 0,
        'category_A': 0,
        'category_B': 0,
        'category_C': 0,
        'len5_passthrough': 0,
        'not_len5': 0,
        'no_code_block': 0,
        'coord_extract_fail': 0,
        'category_D': 0,
        'skip_image_missing': 0,
    }

    print(f'Input:  {INPUT_FILE}')
    print(f'Output: {OUTPUT_FILE}')
    print(f'Reading...')

    with open(INPUT_FILE, 'r') as fin, open(OUTPUT_FILE, 'w') as fout:
        for line_num, line in enumerate(fin, 1):
            line = line.strip()
            if not line:
                continue

            stats['total'] += 1

            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                stats['no_code_block'] += 1
                continue

            # Filter out records whose image paths don't all exist on disk
            images = record.get('image', [])
            if isinstance(images, str):
                images = [images]
            if not images or not all(os.path.exists(p) for p in images):
                stats['skip_image_missing'] += 1
                continue

            resp = record.get('response', [])
            if isinstance(resp, str):
                resp = [resp]

            if len(resp) != 5:
                # Not a len=5 conversation: nothing to rewrite in turn[2]
                fout.write(json.dumps(record, ensure_ascii=False) + '\n')
                stats['not_len5'] += 1
                stats['saved'] += 1
            else:
                stats['len5'] += 1
                new_record, category = process_record(record)

                if new_record is not None:
                    fout.write(json.dumps(new_record, ensure_ascii=False) + '\n')
                    stats['rewritten'] += 1
                    stats['saved'] += 1
                    if category in ('A', 'B', 'C'):
                        stats[f'category_{category}'] += 1
                else:
                    # len=5 but could not be rewritten — skip
                    if category == 'no_code_block':
                        stats['no_code_block'] += 1
                    elif category == 'coord_extract_fail':
                        stats['coord_extract_fail'] += 1
                    elif category == 'category_D_skip':
                        stats['category_D'] += 1

            if line_num % 50000 == 0:
                print(f'  Processed {line_num} lines, '
                      f'rewritten {stats["rewritten"]} so far...')

    # Write stats
    print(f'\nDone! Writing stats to {STATS_FILE}')
    with open(STATS_FILE, 'w') as f:
        f.write('=== Rewrite Statistics ===\n\n')
        f.write(f'Total records:          {stats["total"]}\n')
        f.write(f'Saved (written out):    {stats["saved"]}\n')
        f.write(f'len=5 records:          {stats["len5"]}\n')
        f.write(f'  Rewritten:            {stats["rewritten"]}\n')
        f.write(f'    Category A (offset):  {stats["category_A"]}\n')
        f.write(f'    Category B (as-is):   {stats["category_B"]}\n')
        f.write(f'    Category C (prepend): {stats["category_C"]}\n')
        f.write(f'  len=5 skipped:\n')
        f.write(f'    No code block:        {stats["no_code_block"]}\n')
        f.write(f'    Coord extract fail:   {stats["coord_extract_fail"]}\n')
        f.write(f'    Category D:           {stats["category_D"]}\n')
        f.write(f'Not len=5 (saved as-is):{stats["not_len5"]}\n')
        f.write(f'\nFiltered out:\n')
        f.write(f'  Image missing:        {stats["skip_image_missing"]}\n')
        coverage = 100 * stats['rewritten'] / max(stats['len5'], 1)
        f.write(f'\nRewrite coverage: {stats["rewritten"]}/{stats["len5"]} '
                f'({coverage:.1f}% of len=5)\n')

    # Print summary
    print(f'\n=== Summary ===')
    print(f'Total records:          {stats["total"]}')
    print(f'Saved (written out):    {stats["saved"]}')
    print(f'len=5 records:          {stats["len5"]}')
    print(f'  Rewritten:            {stats["rewritten"]} '
          f'({100*stats["rewritten"]/max(stats["len5"],1):.1f}% of len=5)')
    print(f'    Category A (offset):  {stats["category_A"]}')
    print(f'    Category B (as-is):   {stats["category_B"]}')
    print(f'    Category C (prepend): {stats["category_C"]}')
    print(f'  len=5 skipped:           {stats["len5"] - stats["rewritten"]}')
    print(f'Not len=5 (saved as-is):{stats["not_len5"]}')
    print(f'Filtered out (image missing): {stats["skip_image_missing"]}')


if __name__ == '__main__':
    main()
