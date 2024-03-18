from copy import copy
from numpy cimport ndarray
import numpy as np
cimport numpy as np
from cpython cimport bool
cimport cython
from Tools import makespan
from Tools import LS
from Tools import makespan_with_decoding

# Função que calcula o Lower Bound com alguma operação com um valor zero.
# Utilizado para o BICH.
def LB(p, oper=None): 
    P = np.copy(p)
    if oper:
        P[oper]= 0
    return max(max(P.sum(axis=0)), max(P.sum(axis=1)))


# Gráfico da função Gantt, utilizar para construir as figuras para o artigo.
def gantt(s, p):
    
    import numpy as np
    import matplotlib as mpl
    mpl.use('pgf')
    
    def figsize(scale):
        fig_width_pt = 468.0                          # Get this from LaTeX using \the\textwidth
        inches_per_pt = 1.0/72.27                       # Convert pt to inch
        golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)
        fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
        fig_height = fig_width*golden_mean              # height in inches
        fig_size = [fig_width,fig_height]
        return fig_size
    
    pgf_with_latex = {                      # setup matplotlib to use latex for output
        "pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
        "text.usetex": True,                # use LaTeX to write all text
        "font.family": "serif",
        "font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
        "font.sans-serif": [],
        "font.monospace": [],
        "axes.labelsize": 10,               # LaTeX default is 10pt font.
        "font.size": 10,
        "legend.fontsize": 10,               # Make the legend/label fonts a little smaller
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": figsize(0.95),     # default fig size of 0.9 textwidth
        "pgf.preamble": "\n".join([
            r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
            r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
            ])
        }
    mpl.rcParams.update(pgf_with_latex)
    
    import matplotlib.pyplot as plt
    
    # I make my own newfig and savefig functions
    def newfig(width):
        plt.clf()
        fig = plt.figure(figsize=figsize(width))
        ax = fig.add_subplot(111)
        return fig, ax
    
    def savefig(filename):
        plt.savefig('{}.pgf'.format(filename), bbox_inches='tight')
        plt.savefig('{}.pdf'.format(filename), bbox_inches='tight')
    
    
    # Simple plot
    fig, ax  = newfig(0.95)
    
    colors = ["yellow", "black", "purple", "red", "orange", "blue", "green", "brown", "pink", "violet", "magenta", "#4682B4", "#7FFF00", "#F4A460", "#D8BFD8"]
    n = 0
    
    pos_bars = list(range(10,(len(p))*15,15))

    ax.set_xlabel('Process Times (u.t)')



    ax.set_yticks(list(range(15,(len(p)+1)*15,15)))
    ax.set_yticklabels(['M$_' + str(i+1) + '$' for i in range(len(p))])

    o = tuple((i, j) for i in range(len(p)) for j in range(len(p[0])))
    M, J = np.zeros(len(p)), np.zeros(len(p[0]))
    for oper in s:
        if J[o[oper][1]] > M[o[oper][0]]:

            ax.broken_barh([(J[o[oper][1]], p[o[oper]] )], (pos_bars[o[oper][0]],10), facecolors = colors[o[oper][1] +n])
            M[o[oper][0]] = J[o[oper][1]] +  p[o[oper]]
            J[o[oper][1]] = M[o[oper][0]]
        else:
            if J[o[oper][1]] == 0:
                ax.broken_barh([(M[o[oper][0]], p[o[oper]] )], (pos_bars[o[oper][0]],10), facecolors = colors[o[oper][1] +n], label = 'J$_ ' + str(o[oper][1] + 1) + '$')
            else:
                ax.broken_barh([(M[o[oper][0]], p[o[oper]] )], (pos_bars[o[oper][0]],10), facecolors = colors[o[oper][1] +n])
            
            M[o[oper][0]] += p[o[oper]]
            J[o[oper][1]] = M[o[oper][0]]
        
    ax.set_title('Gantt Chart - Open Shop Problem (Makespan: %i)'% M.max())

    plt.legend(loc=0)
    plt.show()
    return savefig('gantt')





def BICH(p, ls=False):

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
        for index, (machine, job) in enumerate(operations): # Percorrendo cada operação.
            if index not in s and machine == min_index_m and p[machine, job] != 0: # Se a operação ainda não foi alocada e se a máquina que será utilizada for a máquina liberada mais cedo.
                aux = copy(s)
                aux.append(index)
                make_aux = makespan(np.array(aux), p)
                # Calculo o makespan da solução atual e somo com o LB retirando a operação em questão.
                if make_aux + LB(iter_p, (machine, job)) < menor: 
                    menor = make_aux + LB(iter_p, (machine, job))
                    Best_oper = machine, job
                    Best_index = index

        # Aloco a operação que resultar no menor valor
        s.append(Best_index)
        # Transformo seu tempo em 0 para ser levado em consideração nas próximas alteraçãoes.
        iter_p[Best_oper] = 0
        # Atualizo os tempos acumulados dos trabalhos e das máquinas.
        if J[Best_oper[1]] >= M[Best_oper[0]]:
            M[Best_oper[0]] = J[Best_oper[1]] + p[Best_oper]
            J[Best_oper[1]] = M[Best_oper[0]]
        else:
            J[Best_oper[1]] = M[Best_oper[0]] + p[Best_oper]
            M[Best_oper[0]] = J[Best_oper[1]]
      
                
    # Trasnformo s em np.array
    s = np.array(s) 
    # Retorno a solução encontrada
    if ls:
        s = LS(s, p)
    return s       
      

def MIH(p, ls=False):
    
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
        
        for index, (machine, job) in enumerate(operations): # Percorrendo cada operação
            if index not in s and machine == min_index_m and p[machine, job] != 0: # Se a operação ainda não foi alocada e se a máquina que será utilizada for a máquina liberada mais cedo.
                # Calculando a ociosidade, se o tempo acumulado do trabalho for maior do que o da máquina, significa que a máquina
                # terá que espeerar até ele ser concluido para ser alocado nela, ou seja, essa operação gerará uma ociosidade.
                # se o tempo acumulado do job for menor do que o dá máquina então ele já terminou de ser processado em outra máquina, ou seja,
                # Seu processamento na nova máquina não gerará ociosidade.
                ociosidade = J[job] - M[machine]  if J[job] >M[machine] else 0
                if  ociosidade < menor: 
    
                    menor =  ociosidade
                    Best_oper = machine, job
                    Best_index = index
                    
        # Aloco o trabalho que gerar a menor ociosidade na máquina que for liberada mais cedo.
        s.append(Best_index)
        # Transformo seu tempo em 0 para ser levado em consideração nas próximas alteraçãoes.
        iter_p[Best_oper] = 0
        # Atualizo os tempos acumulados dos trabalhos e das máquinas.
        if J[Best_oper[1]] >= M[Best_oper[0]]:
            M[Best_oper[0]] = J[Best_oper[1]] + p[Best_oper]
            J[Best_oper[1]] = M[Best_oper[0]]
        else:
            J[Best_oper[1]] = M[Best_oper[0]] + p[Best_oper]
            M[Best_oper[0]] = J[Best_oper[1]]
    
            
            
    
     # Trasnformo s em np.array
    s = np.array(s) 
    # Retorno a solução encontrada
    if ls:
        s = LS(s, p)
    return s   

def BICH_MIH(p, alpha=0.0, ls=False):
    
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

    
        
        for index, (machine, job) in enumerate(operations):# Percorrendo cada operação.
            if index not in s and machine == min_index_m and p[machine, job] != 0:# Se a operação ainda não foi alocada e se a máquina que será utilizada for a máquina liberada mais cedo.
                aux = copy(s)
                aux.append(index)
                make_aux = makespan(np.array(aux), p)
                ociosidade = J[job] - M[machine]  if J[job] >M[machine] else 0
                # Vou aalocar o próximo trabalho que obtiver a menor soma de alpha*ociosidade + (1-alpha)*LB,
                # onde esse alpha é o grau de importânica que o a regra construtiva dá para ociosidade, um alpha =1 siginifica que a regra olha apenas para a ociosidade
                # um alpha = 0 significa que a regra olha apenas para o LB.
                if (1-alpha)*(make_aux + LB(iter_p, (machine, job))) + (alpha)*ociosidade < menor: 
    
                    menor = (1-alpha)*(make_aux + LB(iter_p, (machine, job))) + (alpha)*ociosidade
                    Best_oper = machine, job
                    Best_index = index
                    

        # Aloco o trabalho que gerar a menor ociosidade + LB para um certo alpha na máquina que for liberada mais cedo.
        s.append(Best_index)
        # Transformo seu tempo em 0 para ser levado em consideração nas próximas alteraçãoes.
        iter_p[Best_oper] = 0
        # Atualizo os tempos acumulados dos trabalhos e das máquinas.
        if J[Best_oper[1]] >= M[Best_oper[0]]:
            M[Best_oper[0]] = J[Best_oper[1]] + p[Best_oper]
            J[Best_oper[1]] = M[Best_oper[0]]
        else:
            J[Best_oper[1]] = M[Best_oper[0]] + p[Best_oper]
            M[Best_oper[0]] = J[Best_oper[1]]
    
            
            
    
    # Trasnformo s em np.array                
    s = np.array(s) 
    # Retorno a solução encontrada
    if ls:
        s = LS(s, p)
    return s

@cython.boundscheck(False)
@cython.wraparound(False)
def IR1(ndarray p):
    
    cdef int n_machines = p.shape[0]
    cdef int n_jobs = p.shape[1]

    
    cdef list operations = [(machine, job) for machine in range(n_machines) 
                for job in range(n_jobs)]    
    
    cdef int i
    cdef int j
    cdef Py_ssize_t r
    cdef Py_ssize_t rr
    cdef Py_ssize_t r3
    cdef bool redundancy
    cdef Py_ssize_t num_operations = n_machines* n_jobs
    
    cdef list WW 
    cdef np.int64_t make_WW
    
    cdef list E = sorted(range(num_operations), key=lambda x: p.ravel()[x], reverse=True)
    
    cdef list W = [E[0]]
    
    
    for r in range(1, num_operations):
        
        Make = float('inf')
        
        i = operations[E[r]][1]
        j = operations[E[r]][0]
        
        
        for rr in range(r):
            redundancy = False
            if rr > 0 and i != operations[W[(rr-1)]][1] and j !=  operations[W[(rr-1)]][0]:
                redundancy = True
            if redundancy == False:
                
                WW = W.copy()
                WW.insert(rr, E[r])
                make_WW = makespan(np.array(WW), p)
                
                if make_WW < Make:
                    Make = make_WW
                    r3 = rr
    
        W.insert(r3, E[r])
    
        
    return np.array(W)

@cython.boundscheck(False)
@cython.wraparound(False)
def IR2(ndarray p):

    cdef int n_machines = p.shape[0]
    cdef int n_jobs = p.shape[1]

    
    cdef list operations = [(machine, job) for machine in range(n_machines) 
                for job in range(n_jobs)]    
    
    cdef int i
    cdef int j
    cdef Py_ssize_t r
    cdef Py_ssize_t rr
    cdef Py_ssize_t r3
    cdef bool redundancy
    cdef Py_ssize_t num_operations = n_machines* n_jobs
    
    cdef list WW 
    cdef np.int64_t make_WW
    
    cdef list E = sorted(range(num_operations), key=lambda x: p.ravel()[x], reverse=True)
    
    cdef list W = [E[0]]
    
    
    for r in range(1, num_operations):
        
        Make = float('inf')
        
        i = operations[E[r]][1]
        j = operations[E[r]][0]
        
        
        for rr in range(r):
            redundancy = False
            if rr > 0 and i != operations[W[(rr-1)]][1] and j !=  operations[W[(rr-1)]][0]:
                redundancy = True
            if redundancy == False:
                
                WW = W.copy()
                WW.insert(rr, E[r])
                make_WW = makespan_with_decoding(np.array(WW), p)
                
                if make_WW < Make:
                    Make = make_WW
                    r3 = rr
    
        W.insert(r3, E[r])
        
    return np.array(W)

@cython.boundscheck(False)
@cython.wraparound(False)
def IR3(ndarray p):
    
    
    cdef int n_machines = p.shape[0]
    cdef int n_jobs = p.shape[1]

    
    cdef list operations = [(machine, job) for machine in range(n_machines) 
                for job in range(n_jobs)]    
    
    cdef int i
    cdef int j
    cdef Py_ssize_t r
    cdef Py_ssize_t r2
    cdef Py_ssize_t r3
    cdef bool redundancy
    cdef Py_ssize_t num_operations = n_machines* n_jobs
    
    cdef ndarray WW 
    cdef np.int64_t make_WW
    cdef Py_ssize_t index
    
    cdef ndarray W = IR2(p)
    
    cdef np.int64_t Make = makespan(W, p)
    
    r= (n_machines* n_jobs)-1
    
    cdef bool improvement = True
    
    while r > 0:
        if improvement == False:
            r -= 1
        improvement = False
        i = operations[W[r]][1]
        j = operations[W[r]][0]
        index = operations.index((j, i))
    
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

@cython.boundscheck(False)
@cython.wraparound(False)
def IR4(ndarray p):

    cdef int k = 5
    
    cdef int n_machines = p.shape[0]
    cdef int n_jobs = p.shape[1]
    
        
    cdef list operations = [(machine, job) for machine in range(n_machines) 
                    for job in range(n_jobs)]
        
    cdef list E = sorted(range(n_machines*n_jobs), key=lambda x: p.ravel()[x], reverse=True)
        
    cdef list W = [E[0]]
    
    
    cdef int i
    cdef int j
    cdef int b
    cdef int c
    cdef Py_ssize_t r
    cdef Py_ssize_t r2
    cdef Py_ssize_t p_index
    cdef Py_ssize_t p_new
    cdef Py_ssize_t r3
    cdef Py_ssize_t r4
    cdef bool redundancy
    cdef Py_ssize_t num_operations = n_machines* n_jobs
    
    cdef list WW 
    cdef list XXX
    cdef np.int64_t make_XX
    cdef np.int64_t make_WW
    
    
    for r in range(1, num_operations):
        
        Make = float('inf')
        
        i = operations[E[r]][1]
        j = operations[E[r]][0]
        
        for r2 in range(r):
            redundancy= False
            
            if r2 > 0 and i != operations[W[(r2-1)]][1] and j !=  operations[W[(r2-1)]][0]:
                    redundancy = True
                    
            if redundancy == False:
                WW = W.copy()
                WW.insert(r2, E[r])
                make_WW = makespan_with_decoding(np.array(WW), p)
                    
                if make_WW < Make:
                    Make = make_WW
                    p_index = r2
                    
        W.insert(p_index, E[r])
        
        for r3 in range(max(1, p_index-k), min(r, p_index+k)):
            
            Make = float('inf')
            
            W.remove(E[r3])
            
            b = operations[E[r3]][1]
            c = operations[E[r3]][0]
            
            for r4 in range(r):
                redundancy = False
                if r4 > 0 and b != operations[W[(r4-1)]][1] and c !=  operations[W[(r4-1)]][0]:
                    redundancy = True
                    
                if redundancy == False:
                    XX = W.copy()
                    XX.insert(r4, E[r3])
                    make_XX = makespan_with_decoding(np.array(XX), p)
                    if make_XX < Make:
                        Make = make_XX
                        p_new = r4
                        
                        
            W.insert(p_new, E[r3])
    return np.array(W)

#p = np.loadtxt('GP03-01.txt', skiprows=3, dtype='int')

#alphas = np.linspace(0.0, 1.0, num=10)  


#best_makespan = min(makespan(decoding(BICH_ID(p, alpha=alpha, ls=True), p)) for alpha in alphas)

#best_makespan = makespan_with_decoding(BICH_MIH(p, alpha=0.0, ls=True), p)

#print(best_makespan)
    




'''
instâncias = []# Deve receber todos os nomes das instânicas no formato .txt  for i in [3,4,5,6,7,8,9,10]:
for i in [3,4,5,6,7,8,9,10]:
    for j in range(1,11):
        if i<10 and j<10:
            instâncias.append('GP0' + str(i) + '-0' + str(j) + '.txt')
        elif i >= 10 and j < 10:
            instâncias.append('GP' + str(i) + '-0' + str(j) + '.txt')
        elif i < 10 and j >= 10:
             instâncias.append('GP0' + str(i) + '-' + str(j) + '.txt')
        else:
            instâncias.append('GP' + str(i) + '-' + str(j) + '.txt')

alphas = np.linspace(0.0, 1.0, num=10)  

results = []            
for instance in instâncias:
    print(instance)
    p = np.loadtxt(instance, skiprows=3, dtype='int')
    # Coleto o makespan que gerar o melhor resultado.
    #results.append(min(makespan(decoding(BICH_ID(p, alpha=alpha, ls=True), p)) for alpha in alphas))
    results.append(makespan(decoding(ID(p, ls=True), p)))
results =  np.array(results)

np.savetxt('MIH-LS-RuleWithDecoding.txt',results, fmt='%d', newline='\r\n')

'''

        



