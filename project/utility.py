def get_lines(fp, k_line=5, verbose=False, preprocess=lambda x:x):
    '''
    Input
        fp: file name
        k_line: read first k lines, if k < 0, then read all file
        verbose:
        preprocess : an lambda function that preprocesses raw data

    Return:
        a list of preprocessed lines
    '''
    res = []
    for idx, line in enumerate(open(fp)):
        if k_line > 0 and idx >= k_line:
            break
        if verbose:
            print line

        res.append(preprocess(line))
    return res
