# Test insertion heuristic

import numpy as np
from copy import copy
from ConstrutiveHeuristics import LS
from collections import defaultdict

def LB(p, oper=None): 
    P = np.copy(p)
    if oper:
        P[oper]= 0
    return max(max(P.sum(axis=0)), max(P.sum(axis=1)))

# Função objetivo, recebe como argumento uma sequência de processamento das operações.
def makespan(s, p):
    operations = tuple((i, j) for i in range(len(p)) for j in range(len(p[0])))
    M, J = np.zeros(len(p), dtype='int'), np.zeros(len(p[0]), dtype='int')
    for oper in s:
        
        machine, job = operations[oper]
        time = p[machine, job]
        M[machine] = max(M[machine], J[job]) + time
        J[job] = M[machine]
        
    return M.max()

def makespan_with_decoding(W, p):
    U = W.copy()
    U = list(U)
    S = []
    n_machines, n_jobs = p.shape
    M,J = np.zeros(p.shape[0], dtype='int'),np.zeros(p.shape[1], dtype='int')
    operations = [(machine, job) for machine in range(n_machines) 
            for job in range(n_jobs)]
    
    while len(U) != 0:
        s = [max(M[operations[index][0]], J[operations[index][1]]) for index in U]
        y = min(s)
        R = [element for index, element in enumerate(U) if s[index] == y]
        O = R[0]
        U.remove(O)
        S.append(O)
        machine, job = operations[O]
        time = p[machine, job]
        
        M[machine] = max(M[machine], J[job]) + time
        J[job] = M[machine]

    return M.max()
# p = np.loadtxt('GP03-01.txt', skiprows=3, dtype='int')

def BICH_MIH(p, alpha=0.0, alphaGRASP=1.0,  ls=False):
        
    s = [] # Inicia a solução vazia
        
    
    n_machines, n_jobs = p.shape # Coleta a quantidade máquinas e de trabalhos através dos dados.
    
    # Gera uma lista com todas as operaçções.
    operations = [(machine, job) for machine in range(n_machines) 
            for job in range(n_jobs)]
    
    # Alocandol memória para os vetores que vão guardar os tempos acumulados das máquinas
    # e dos trabalhos.
    M,J = np.zeros(p.shape[0], dtype='int'),np.zeros(p.shape[1], dtype='int')
    
    
    
    # Criando uma cópia da matriz de tempos para ser utilizada nas iterações do algoritmo.
    iter_p = np.copy(p)
    
    while len(s) < n_machines*n_jobs: # Enquanto a solução não estiver completa,
    # ou seja, não estiverem alocados todos os trabalhos.    
        
        # variável min_index_m vai receber o índice da máquina liberada mais cedo.
        menor = float('inf')
        for index, m in enumerate(M):
            if m < menor and iter_p[index].sum() != 0:
                menor = m
                min_index_m = index
        
        
        menor = float('inf')
    
        CL = {}
        
        for index, (machine, job) in enumerate(operations):# Percorrendo cada operação.
            if index not in s and machine == min_index_m and p[machine, job] != 0:# Se a operação ainda não foi alocada e se a máquina que será utilizada for a máquina liberada mais cedo.
                aux = copy(s)
                aux.append(index)
                make_aux = makespan(aux, p)
                ociosidade = J[job] - M[machine]  if J[job] > M[machine] else 0
                # Vou aalocar o próximo trabalho que obtiver a menor soma de alpha*ociosidade + (1-alpha)*LB,
                # onde esse alpha é o grau de importânica que o a regra construtiva dá para ociosidade, um alpha =1 siginifica que a regra olha apenas para a ociosidade
                # um alpha = 0 significa que a regra olha apenas para o LB.
                CL[index] = (1-alpha)*(make_aux + LB(iter_p, (machine, job))) + (alpha)*ociosidade
    
                    
        RCL = [index for index, value in CL.items() if value <= min(CL.values())/alphaGRASP]
    
        candidate = np.random.choice(RCL)
        
        s.append(candidate)
        
        Best_oper = operations[candidate]
        
        iter_p[Best_oper] = 0
        # Atualizo os tempos acumulados dos trabalhos e das máquinas.
        Best_machine, Best_job = Best_oper
        time = p[Best_oper]
        
        M[Best_machine] = max(M[Best_machine], J[Best_job]) + time
        J[Best_job] = M[Best_machine]    
            
    # Trasnformo s em np.array                
    s = np.array(s)     
    if ls:
        s = LS(s, p)
    return s

def insert_heuristic(U, p):
    U = U.tolist().copy()
    s = [U[0], U[1]]
    
    U.pop(0)
    U.pop(0)
    
    while U:
        
        aux = s.copy()
        
        menor_make = float('inf')
        #for element in U:
        element = U[0]
        for index,_ in enumerate(s):
    
            aux.insert(index, element)
            make_aux = makespan(aux, p)
            aux.pop(index)
            
            if make_aux < menor_make:
                menor_make = make_aux
                menor_index = index
                best_element = element
        
        # aux.append(element)
        # make_aux = makespan(aux, p)
        # aux.pop(-1)
        
        # if make_aux < menor_make:
        #     menor_make = make_aux
        #     menor_index = index
        #     best_element = element        
                    
        s.insert(menor_index, best_element)
        U.remove(best_element)
    
    return np.array(s)

#s = sorted(range(p.size), key=lambda x: p[o[x]], reverse=True)

# Phi = np.linspace(0.1, 0.9, 10)
# prob = np.full(10, 1/10)
# A = defaultdict(list)
# q = {}
# ##sol = makespan(s, p)
# p = np.loadtxt('GP10-01.txt', skiprows=3, dtype='int')
# o = tuple((i, j) for i in range(p.shape[0]) for j in range(p.shape[1]))

# BestSol = sorted(range(p.size), key=lambda x: p[o[x]], reverse=True)
# BestMakespan = makespan_with_decoding(BestSol, p)

# for k in range(p.size*10):
#     alpha = 0.0
#     alphaGRASP = np.random.choice(Phi, p=prob)
    
#     sol = BICH_MIH(p, alpha=alpha, alphaGRASP=alphaGRASP, ls=True)
    
#     make_sol = makespan_with_decoding(sol, p)
    
#     A[alphaGRASP].append(make_sol)
    
#     if make_sol < BestMakespan:
#         BestMakespan = make_sol
#         BestSol = sol 
#         print(BestMakespan)
        
#     q = {i: (BestMakespan/(sum(A[i])/len(A[i])))**10 for i in Phi if A[i]}
    
#     if len(q.keys()) == len(Phi):
#         prob = np.array([q[i]/sum(q.values()) for i in Phi])

# BestMakespan = makespan_with_decoding(BestSol, p)
# print(BestMakespan)




       