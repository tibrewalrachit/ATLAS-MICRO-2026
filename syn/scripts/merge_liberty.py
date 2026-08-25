#!/usr/bin/env python3
"""Merge several Liberty files into a single library.

yosys' dfflibmap/abc passes each take one -liberty file, but ASAP7 ships its
combinational, sequential and inverter/buffer cells as three separate
libraries.  This concatenates every top-level `cell (...)` block from the
inputs into the first file's library header so the whole cell set is visible
to technology mapping in one pass.

OpenSTA is not affected -- it accepts repeated read_liberty commands -- so the
merged file is only used by synthesis.
"""
import re
import sys


def split_library(text):
    """Return (preamble, [cell_blocks]) for one liberty file."""
    m = re.search(r'^\s*library\s*\(', text, re.M)
    if not m:
        raise ValueError("no library block found")

    # Locate the first top-level cell block; everything before it is preamble.
    cells = []
    i = 0
    first_cell = None
    depth = 0
    lib_start = text.index('{', m.start()) + 1
    i = lib_start
    depth = 1
    while i < len(text):
        c = text[i]
        if c == '{':
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0:
                break
        elif depth == 1:
            cm = re.match(r'cell\s*\(', text[i:])
            if cm:
                start = i
                # Walk to the matching close brace of this cell.
                j = text.index('{', i)
                d = 1
                j += 1
                while d:
                    if text[j] == '{':
                        d += 1
                    elif text[j] == '}':
                        d -= 1
                    j += 1
                cells.append(text[start:j])
                if first_cell is None:
                    first_cell = start
                i = j
                continue
        i += 1

    preamble = text[:first_cell] if first_cell is not None else text[:lib_start]
    return preamble, cells


def main(out_path, in_paths):
    preamble = None
    all_cells = []
    for p in in_paths:
        with open(p) as f:
            txt = f.read()
        pre, cells = split_library(txt)
        if preamble is None:
            preamble = pre
        all_cells.extend(cells)
        print("  %-24s %4d cells" % (p.rsplit('/', 1)[-1], len(cells)))

    with open(out_path, 'w') as f:
        f.write(preamble)
        for c in all_cells:
            f.write(c)
            f.write("\n")
        f.write("}\n")
    print("merged %d cells -> %s" % (len(all_cells), out_path))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2:])
