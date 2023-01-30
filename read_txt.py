""" Small utility to read a vector or matrix stored between a `# tag`
    and another `# anytag` (or EOF)
"""

import numpy as np

def read_txt (fname, tag) :
    start_row = None
    end_row = None
    with open(fname, 'r') as f :
        for ii, line in enumerate(f) :
            if line.rstrip() == f'# {tag}' :
                assert start_row is None
                start_row = ii
            if start_row is not None and line[0]=='#' :
                end_row = ii
    assert start_row is not None
    max_rows = end_row-start_row if end_row is not None else None
    return np.loadtxt(fname, skiprows=start_row, max_rows=max_rows)
