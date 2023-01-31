""" Small utility to read a vector or matrix stored between a `# tag`
    and another `# anytag` (or EOF).
    Can also read python objects if they can be parsed by eval().
"""

import itertools
import numpy as np

def read_txt (fname, tag, pyobj=False) :
    start_row = None
    end_row = None
    contents = ''
    with open(fname, 'r') as f :
        for ii, line in enumerate(f) :
            if line.rstrip() == f'# {tag}' :
                assert start_row is None
                start_row = ii
            elif start_row is not None and line[0]=='#' :
                end_row = ii
                break
            elif start_row is not None :
                contents += line
    assert start_row is not None
    if pyobj :
        return eval(contents)
    else :
        max_rows = end_row-start_row if end_row is not None else None
        with open(fname, 'r') as f :
            return np.loadtxt(itertools.islice(f, start_row, end_row))
