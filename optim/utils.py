def dict_sum(d1, d2):
    for kw in d2:
        d1[kw] = d2[kw]
    return d1

def print_dict(d, n_indent=0):
    for k in d:
        if isinstance(d[k], dict):
            print_dict(d[k], n_indent+1)
        else:
            print("{}- {} : {}".format(n_indent, k, d[k]))

