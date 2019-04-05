import scipy.sparse as sp
import numpy as np

csr1 = sp.csr_matrix([[0, 0, 1, 0],
                      [0, 2, 0, 3],
                      [4, 0, 0, 0]])
csr2 = sp.csr_matrix([[0, 5, 6],
                      [0, 7, 0],
                      [0, 0, 0]])
csr3 = sp.csr_matrix([[0, 0, 0],
                      [8, 0, 0],
                      [0, 9, 10]])

row = csr1.shape[0]
offset_by_row = np.zeros([row], dtype=np.int64)
size = 0
column = 0
for csr in [csr1, csr2, csr3]:
    size += csr.size
    column += csr.shape[1]
    for i in range(row):
        offset_by_row[i] += csr.indptr[i]
    if not csr.shape[0] == row:
        raise Exception()
    print(csr.toarray())

# del *

result = {
    'data': np.zeros([size]),
    'indices': np.zeros([size], dtype=np.int64),
    'indptr': np.zeros([row + 1], dtype=np.int64)
}

width = 0
for csr in [csr1, csr2, csr3]:
    indptr = csr.indptr
    acc_row_size = 0
    for i in range(row):
        row_size = indptr[i + 1] - indptr[i]
        s1 = slice(indptr[i], indptr[i] + row_size)
        s2 = slice(offset_by_row[i], offset_by_row[i] + row_size)
        result['data'][s2] = csr.data[s1]
        result['indices'][s2] = csr.indices[s1] + width
        acc_row_size += row_size
        result['indptr'][i + 1] += acc_row_size
        offset_by_row[i] += row_size
    print(result)
    width += csr.shape[1]
print(result)

csr = sp.csr_matrix((result['data'], result['indices'], result['indptr']))
print(csr.toarray())
