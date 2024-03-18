# Tools

# Makespan

from numpy cimport ndarray
import numpy as np
cimport numpy as np
from cpython cimport bool
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef np.int64_t makespan(ndarray[np.int64_t, ndim=1] s, ndarray[np.int64_t, ndim=2] p):

    cdef Py_ssize_t i, j
    cdef list o = [(i, j) for i in range(len(p)) for j in range(len(p[0]))]
    
    
    cdef ndarray[np.int64_t, ndim=1] M = np.zeros(len(p), dtype=np.int64)
    cdef ndarray[np.int64_t, ndim=1] J = np.zeros(len(p[0]), dtype=np.int64)
    cdef Py_ssize_t oper
    
    for oper in s:
            
        if J[o[oper][1]] >= M[o[oper][0]]:
            M[o[oper][0]] = J[o[oper][1]] + p[o[oper]]
            J[o[oper][1]] = M[o[oper][0]]
        else:
            J[o[oper][1]] = M[o[oper][0]] + p[o[oper]]
            M[o[oper][0]] = J[o[oper][1]]
            
    return M.max()

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef np.int64_t makespan_with_decoding(ndarray[np.int64_t, ndim=1] W, ndarray[np.int64_t, ndim=2] p):
    
    
    
    cdef ndarray[np.int64_t, ndim=1] U = W.copy()
    cdef list Q = U.tolist()
    cdef list S = []
    cdef Py_ssize_t n_machines = p.shape[0]
    cdef Py_ssize_t n_jobs = p.shape[1]
    cdef ndarray[np.int64_t, ndim=1] M = np.zeros(n_machines, dtype=np.int64)
    cdef ndarray[np.int64_t, ndim=1] J = np.zeros(n_jobs, dtype=np.int64)
    cdef Py_ssize_t Machine
    cdef Py_ssize_t Job
    cdef list operations = [(Machine, Job) for Machine in range(n_machines) 
            for Job in range(n_jobs)]
    
    cdef list s
    cdef np.int64_t y 
    cdef list R
    cdef np.int64_t O
    
    cdef int machine
    cdef int job
    
    while len(Q) != 0:
        s = [max(M[operations[index][0]], J[operations[index][1]]) for index in Q]
        y = min(s)
        R = [element for index, element in enumerate(Q) if s[index] == y]
        O = R[0]
        Q.remove(O)
        S.append(O)
        machine = operations[O][0]
        job = operations[O][1] 
        
        if J[job] >= M[machine]:
            M[machine] = J[job] + p[machine, job]
            J[job] = M[machine]
        else:
            J[job] = M[machine] + p[machine, job]
            M[machine] = J[job]
    return M.max()

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef ndarray[np.int64_t, ndim=1] decoding(ndarray[np.int64_t, ndim=1] W, ndarray[np.int64_t, ndim=2] p):
    
    
    
    cdef ndarray[np.int64_t, ndim=1] U = W.copy()
    cdef list Q = U.tolist()
    cdef list S = []
    cdef Py_ssize_t n_machines = p.shape[0]
    cdef Py_ssize_t n_jobs = p.shape[1]
    cdef ndarray[np.int64_t, ndim=1] M = np.zeros(n_machines, dtype=np.int64)
    cdef ndarray[np.int64_t, ndim=1] J = np.zeros(n_jobs, dtype=np.int64)
    cdef Py_ssize_t Machine
    cdef Py_ssize_t Job
    cdef list operations = [(Machine, Job) for Machine in range(n_machines) 
            for Job in range(n_jobs)]
    
    cdef list s
    cdef np.int64_t y 
    cdef list R
    cdef np.int64_t O
    
    cdef int machine
    cdef int job
    
    while len(Q) != 0:
        s = [max(M[operations[index][0]], J[operations[index][1]]) for index in Q]
        y = min(s)
        R = [element for index, element in enumerate(Q) if s[index] == y]
        O = R[0]
        Q.remove(O)
        S.append(O)
        machine = operations[O][0]
        job = operations[O][1] 
        
        if J[job] >= M[machine]:
            M[machine] = J[job] + p[machine, job]
            J[job] = M[machine]
        else:
            J[job] = M[machine] + p[machine, job]
            M[machine] = J[job]
			
    cdef ndarray[np.int64_t, ndim=1] SOL = np.array(S, dtype=np.int64)
    return SOL

@cython.wraparound(False)
@cython.boundscheck(False)
cpdef ndarray[np.int64_t, ndim=1] LS(ndarray[np.int64_t, ndim=1] s, ndarray[np.int64_t, ndim=2] p):
    
    cdef int n_machines = p.shape[0]
    cdef int n_jobs = p.shape[1]
    cdef Py_ssize_t machine
    cdef Py_ssize_t job
    cdef list operations = [(machine, job) for machine in range(n_machines) 
                for job in range(n_jobs)]
    
    cdef ndarray[np.int64_t, ndim=1] W = s.copy() 
    cdef np.int64_t Make = makespan(W, p) 
    cdef Py_ssize_t r= (n_machines* n_jobs)-1 
    cdef bool improvement = True
    
    
    cdef int i
    cdef int j
    cdef Py_ssize_t r2
    cdef Py_ssize_t r3
    cdef Py_ssize_t index
    cdef bool redundancy
    cdef Py_ssize_t num_operations = n_machines* n_jobs
    
    cdef ndarray[np.int64_t, ndim=1] WW 
    cdef np.int64_t make_WW
    
    
    while r > 0:
        if improvement == False:
            r -= 1
        improvement = False
        #aux = W[r]
        i = operations[W[r]][0]
        j = operations[W[r]][1]
        index = operations.index((i, j)) # era i, j
    
        for r2 in range(num_operations):
            redundancy = False
            if r2 > 0 and i != operations[W[(r2-1)]][1] and j !=  operations[W[(r2-1)]][0]:
                redundancy = True
            if redundancy == False:
                WW = W.copy()
                WW[index], WW[r2] = WW[r2], WW[index]
                make_WW = makespan_with_decoding(WW, p)
                if make_WW < Make:
                    improvement = True
                    Make = make_WW
                    r3 = r2
                    
        if improvement == True:
            W[index], W[r3] = W[r3], W[index]
            r = (n_machines* n_jobs)-1
            
    return W
