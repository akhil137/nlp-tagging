In [4]: %time corpus=[(reuters.raw(fileid),cat) for cat in reuters.categories() for fileid in reuters.fileids(cat)]
CPU times: user 3.19 s, sys: 1.90 s, total: 5.09 s
Wall time: 63.61 s

