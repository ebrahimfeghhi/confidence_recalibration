#!/usr/bin/env python
"""One table formatter for every analysis in this project.

WHY IT EXISTS. Tables were being formatted ad hoc inside each script with hand-counted printf widths,
so every analysis had its own column sizes, its own decimal count, and alignment that broke the moment a
value got wider. This centralises it: pass rows, get a box-drawn table with numerics right-aligned on a
fixed number of decimals and a real minus sign.

    from fmt import table
    print(table(["interface","class","FREE","HYBRID","delta"],
                rows,                      # list of lists
                align="l l r r r",         # per column; default: left for str, right for numbers
                dec={"FREE":3,"HYBRID":3,"delta":3},
                signed=["delta"],          # force a leading + or -
                rules=[3]))                # horizontal rule AFTER these row indices
"""


def _fmt(v, dec=None, signed=False):
    if v is None:
        return ""
    if isinstance(v, float):
        s = f"{v:+.{dec if dec is not None else 3}f}" if signed else f"{v:.{dec if dec is not None else 3}f}"
        return s.replace("-", "−")          # real minus, so columns line up with '+'
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:+d}".replace("-", "−") if signed else str(v)
    return str(v)


def table(headers, rows, align=None, dec=None, signed=(), rules=(), title=None, style=None):
    """style="md" emits a GitHub-flavoured pipe table. That is the DEFAULT, because these tables are
    read in the Claude Code interface, which renders markdown -- box-drawing characters are not
    aligned there, they are just literal characters inside a rendered paragraph. style="box" keeps the
    old Unicode grid for terminal use. Override globally with FMT_STYLE=box."""
    import os as _os
    if (style or _os.environ.get("FMT_STYLE", "md")) == "md":
        return _md(headers, rows, align, dec, signed, title)
    return _box(headers, rows, align, dec, signed, rules, title)


def _md(headers, rows, align=None, dec=None, signed=(), title=None):
    dec = dec or {}
    signed = set(signed)
    ncol = len(headers)
    cells = [[_fmt(v, dec.get(headers[i]), headers[i] in signed) for i, v in enumerate(r)]
             for r in rows]
    if align:
        al = align.split()
    else:
        al = []
        for i in range(ncol):
            col = [r[i] for r in rows if r[i] is not None]
            al.append("r" if col and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                         for v in col) else "l")
    out = [f"**{title}**", ""] if title else []
    out.append("| " + " | ".join(headers) + " |")
    out.append("|" + "|".join("---:" if a == "r" else ":---" for a in al) + "|")
    for c in cells:
        out.append("| " + " | ".join(c) + " |")
    return "\n".join(out)


def _box(headers, rows, align=None, dec=None, signed=(), rules=(), title=None):
    dec = dec or {}
    signed = set(signed)
    ncol = len(headers)
    cells = [[_fmt(v, dec.get(headers[i]), headers[i] in signed) for i, v in enumerate(r)]
             for r in rows]

    if align:
        al = align.split()
    else:                                        # infer: numbers right, everything else left
        al = []
        for i in range(ncol):
            col = [r[i] for r in rows if r[i] is not None]
            al.append("r" if col and all(isinstance(v, (int, float)) and not isinstance(v, bool)
                                         for v in col) else "l")
    w = [max(len(headers[i]), *(len(c[i]) for c in cells)) if cells else len(headers[i])
         for i in range(ncol)]

    def line(l, m, r):
        return l + m.join("─" * (x + 2) for x in w) + r

    def row(vals):
        out = []
        for i, v in enumerate(vals):
            out.append(f" {v:>{w[i]}} " if al[i] == "r" else f" {v:<{w[i]}} ")
        return "│" + "│".join(out) + "│"

    lines = []
    if title:
        lines.append("  " + title)
    lines.append(line("┌", "┬", "┐"))
    lines.append(row(headers))
    lines.append(line("├", "┼", "┤"))
    for n, c in enumerate(cells):
        lines.append(row(c))
        if n in rules and n != len(cells) - 1:
            lines.append(line("├", "┼", "┤"))
    lines.append(line("└", "┴", "┘"))
    return "\n".join("  " + x for x in lines)


def kv(pairs, width=None, title=None):
    """A two-column summary block for scalars that do not belong in the main table."""
    width = width or max(len(k) for k, _ in pairs)
    out = ["  " + title] if title else []
    for k, v in pairs:
        out.append(f"  {k:<{width}}  {v}")
    return "\n".join(out)
