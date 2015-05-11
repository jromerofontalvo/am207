
# coding: utf-8

# In[58]:

# author: Jhonathan Romero Fontalvo (v.alpha,, 2015)

import time
import numpy as np
import scipy
from scipy.optimize import minimize
import warnings
import commutators
import scipy.sparse
import scipy.optimize
import scipy.sparse.linalg
from itertools import combinations
from sys import argv
import itertools
import commutators #(Ryan Babbush)


# ## Symbolic Bravyi-Kitaev (BK) and Jordan-Wigner (JW) transformations.
# 
# Next cell contents the information required to obtain the BK representation of systems up to 8 qubits.

# In[59]:

# Parity, Update and Flip sets for 8 qubits                                                                                                        
P8={7:[6,5,3],6:[5,3],5:[4,3],4:[3],3:[2,1],2:[1],1:[0],0:[]}
R8={7:[6,5,3],6:[5,3],5:[4,3],4:[3],3:[2,1],2:[1],1:[0],0:[]}
U8={7:[],6:[7],5:[7],4:[5,7],3:[7],2:[3,7],1:[3,7],0:[1,3,7]}
F8={7:[6,5,3],6:[],5:[4],4:[],3:[2,1],2:[],1:[0],0:[]}
# Creating remaining set
for n in range(0,8):
    for x in F8[n]:
        if x in R8[n]:
            R8[n].remove(x)

# Parity, Update and Flip sets for 6 qubits                                                                                                        
P6={5:[4,3],4:[3],3:[2,1],2:[1],1:[0],0:[]}
R6={5:[4,3],4:[3],3:[2,1],2:[1],1:[0],0:[]}
U6={5:[],4:[5],3:[],2:[3],1:[3],0:[1,3]}
F6={5:[4],4:[],3:[2,1],2:[],1:[0],0:[]}
# Creating remaining set
for n in range(0,6):
    for x in F6[n]:
        if x in R6[n]:
            R6[n].remove(x)

# Parity, Update and Flip sets for 4 qubits                                                                                                        
P4={3:[2,1],2:[1],1:[0],0:[]}
R4={3:[2,1],2:[1],1:[0],0:[]}
U4={3:[],2:[3],1:[3],0:[1,3]}
F4={3:[2,1],2:[],1:[0],0:[]}
# Creating remaining set
for n in range(0,4):
    for x in F4[n]:
        if x in R4[n]:
            R4[n].remove(x)
print R4
print P4


# Next cell comprises several functions:
# 
# 1) chunckIt: just breaks a vector in chuncks of a given dimension. For example [a,b,c,d] in chuncks of 2: [a,b] [c,d].
# 
# 2) Class term: Is a class to write down products of Pauli matrices. Every term has a coefficient (complex) and a string (a dictionary of letters I, X, Y, P assigned to positions for the qubits). Terms can be reduced (multiply the Pauli matrices for a given qubit) and there are 4 routines for printing the terms for different purposes.
# 
# 3) returnBKform: take an object of class term and the corresponding Parity, Update, Flip and Reminder sets used in the BK transformation to return the BK form of the term (It returns a list of terms).
# 
# 4) getHamiltonian: take a list of coefficients and terms written in second quantization and transform it into the corresponding terms in BK form. The routine compares all the terms (brute force comparison) to reduce the Hamiltonian to a minimal algebraic represetation. The format for the second quantized terms in explained below (same format that Ryan's commutator routine.)

# In[68]:

# create chucks
def chunkIt(seq, num):
  avg = len(seq) / float(num)
  out = []
  last = 0.0

  while last < len(seq):
    out.append(seq[int(last):int(last + avg)])
    last += avg

  return out

# Class term: defines a product of Pauli matrices
class term:
    def __init__(self,nqubits,alpha=0.0):
        string={}
        for n in range(0,nqubits):
            string[n]=['I']
        self.c = 1.0
        self.pl = string    
        self.matrix=[]
        self.l=nqubits
        self.alpha=alpha
    def reduceTerm(self):
        dic1={'ZZ':'I','YY':'I','XX':'I','II':'I','XY':'Z','XZ':'Y','YX':'Z','YZ':'X','ZX':'Y',
              'ZY':'X','IX':'X','IY':'Y','IZ':'Z','ZI':'Z','YI':'Y','XI':'X'}
        dic2={'ZZ':1.0,'YY':1.0,'XX':1.0,'II':1.0,'XY':1.0j,'XZ':-1.0j,'YX':-1.0j,'YZ':1.0j,'ZX':1.0j,
              'ZY':-1.0j,'IX':1.0,'IY':1.0,'IZ':1.0,'ZI':1.0,'YI':1.0,'XI':1.0}
        factor=1.0
        for x in self.pl:
            cosita=self.pl[x]
            length=len(cosita)
            while length>1:
                pair=''.join(cosita[:2])
                del cosita[:2]
                cosita.insert(0,dic1[pair])
                factor=factor*dic2[pair]
                length=len(cosita)
            self.pl[x]=cosita
        self.c=factor*self.c
    def printTerm(self):
        vprint=[]
        for x in self.pl:
            vprint += self.pl[x]
            vprint += str(x)
        vprint=''.join(vprint)
        return self.c,vprint
    def printTerm2(self):
        vprint=[]
        for x in self.pl:
            vprint += self.pl[x]
        return self.c,vprint
    def printTerm3(self):
        vprint=[]
        counter=0
        n_swaps=0
        n_x=0; n_y=0
        maxi=0
        mini=self.l
        for x in self.pl:
            if self.pl[x] == ['Y']:
                n_y += 1
            if self.pl[x] == ['X']:
                n_x += 1
            if self.pl[x] != ['I']:
                counter=counter+1
                vprint += self.pl[x]
                if x>maxi:
                    maxi=x
                if x<mini:
                    mini=x
                vprint += str(x)
            elif self.pl[x] == ['I']:
                vprint += [' ']
        for n in range(mini,maxi+1):
            if self.pl[n] == ['I']:
                n_swaps += 2
        vprint=''.join(vprint)
        n_cnots=2*(maxi-mini)-n_swaps
        sqcount=n_x*2+n_y*2+2*n_cnots+6*n_swaps
        czcount=1*(n_cnots)+3*n_swaps
        return sqcount+2*czcount
    def printTerm4(self):
        vprint=[]
        counter=0
        for x in self.pl:
            if self.pl[x] != ['I']:
                counter=counter+1
                vprint += '\sigma^'
                vprint += self.pl[x]
                vprint += '_'
                vprint += str(x)
        vprint=''.join(vprint)
        return self.c,vprint,8-counter
    def getMatrix(self):
        occ = scipy.sparse.csr_matrix([[0], [1]], dtype=float)
        vir = scipy.sparse.csr_matrix([[1], [0]], dtype=float)
        I = scipy.sparse.csr_matrix([[1, 0], [0, 1]], dtype=complex)
        X = scipy.sparse.csr_matrix([[0, 1], [1, 0]], dtype=complex)
        Y = scipy.sparse.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
        Z = scipy.sparse.csr_matrix([[1, 0], [0, -1]], dtype=complex)
        SWAP=scipy.sparse.csr_matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex) # S
        H = (1/np.sqrt(2))*scipy.sparse.csr_matrix([[1, 1], [1, -1]], dtype=complex) # H
        CNOT=scipy.sparse.csr_matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex) # C
        Rz=scipy.sparse.linalg.expm(-1j*(self.alpha/2)*Z)
        RX2=scipy.sparse.linalg.expm(-(np.pi/4)*1j*X) # A
        RY2=scipy.sparse.linalg.expm(-(np.pi/4)*1j*Y) # B
        R_X2=scipy.sparse.linalg.expm((np.pi/4)*1j*X) # a
        R_Y2=scipy.sparse.linalg.expm((np.pi/4)*1j*Y) # b
        CZ=scipy.sparse.csr_matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex) # E        
        # Dictionary of gates
        pauli={'I':I,'X':X,'Y':Y,'Z':Z,'H':H,'A':RX2,'B':RY2,'a':R_X2,'b':R_Y2,'C':CNOT,'1':occ,'0':vir,'S':SWAP,
               'E':CZ,'Rz':Rz}
        counter=-1
        m=1
        for t in self.pl:
            counter += 1
            m=scipy.sparse.kron(pauli[self.pl[t][0]],m,'csr')
        self.matrix=self.c*m
        
# input: operator of the form [i,j,-k,-l] that represents a^{/dag}_i a^{/dag}_j a_k a_l. Any length.
# output: the representation of the operator in the BK form
def returnBKform(operator,P,U,F,R):
    # P, U and F must have the same number of elements = nqubits
    l=len(operator)
    size=2*l
    nqubits=len(P)
    BKform=[]
    counter=-1
    for a in operator:
        counter += 1
        s1=term(nqubits); s2=term(nqubits)
        j=abs(a)-1
        # case j is even
        if j%2==0:
            s1.pl[j] += 'X'
            #print U[j], P[j], R[j]
            for k in U[j]:
                s1.pl[k] += 'X'
                #print k
            for k in P[j]:
                #print k
                s1.pl[k] += 'Z'
            s2.pl[j] += 'Y'
            for k in U[j]:
                s2.pl[k] += 'X'
                #print k
            for k in P[j]:
                s2.pl[k] += 'Z'
                #print k
            # annihilator
            if a<0:
                s1.c=0.5
                s2.c=0.5j
            # creator
            elif a>0:
                s1.c=0.5
                s2.c=-0.5j              
            BKform += [s1]
            BKform += [s2]
            s1.reduceTerm()
            s2.reduceTerm()
        # case j is odd
        elif j%2==1:
            s1.pl[j] += 'X'
            #print U[j], P[j], R[j]
            for k in U[j]:
                s1.pl[k] += 'X'
                #print k
            for k in P[j]:
                s1.pl[k] += 'Z'
                #print k
            s2.pl[j] += 'Y'
            for k in U[j]:
                s2.pl[k] += 'X'
                #print k
            for k in R[j]:
                s2.pl[k] += 'Z'
                #print k
            # annihilator
            if a<0:
                s1.c=0.5
                s2.c=0.5j
            # creator
            elif a>0:
                s1.c=0.5
                s2.c=-0.5j 
            s1.reduceTerm()
            s2.reduceTerm()
            BKform += [s1]
            BKform += [s2]
        #print s1.printTerm()
        #print s2.printTerm()
        del s1; del s2
    products=[]
    iterables = chunkIt(range(size), size/2)
    counter = 0
    for element in itertools.product(*iterables):
        counter += 1
        s=term(nqubits)
        for t in element:
            for k in range(0,nqubits):
                s.pl[k] += BKform[t].pl[k]
            s.c=s.c*BKform[t].c
        s.reduceTerm()
        products += [s]
        del s
    # finding terms that are equivalent
    products2=simplifyTerms(products)
    return products2

def simplifyTerms(products):
    # finding terms that are equivalent
    error=10**(-12)
    ss=[]
    products2=[]
    indexes=range(0,len(products))
    counter=0
    removed=[]
    for p in indexes:
        if (p in removed)==False:
            s=[]
            s += [p]; removed += [p]; counter +=1
            value,string=products[p].printTerm()
            for q in indexes:
                if (q in removed)==False:
                    value2,string2=products[q].printTerm()
                    if string2==string:
                        s += [q]; removed += [q]; counter +=1
            ss += [s]
        if counter == len(products):
            break
    # calculating sums and eliminated repeated terms
    for x in ss:
        value=0.0
        for y in x:
            value=value+products[y].c
        if abs(value)>error:
            products[x[0]].c=value
            products2.append(products[x[0]])
    del products
    return products2

def getHamiltonian(coefficients, terms,P,U,F,R):
     #= commutators.GetHamiltonianTerms(molecule, basis, add_conjugates=True)
    products=[]
    conteo=0
    for c,t in zip(coefficients,terms):    
        #print c,t
        cosa = returnBKform(t,P,U,F,R)
        conteo=conteo+len(cosa)
        for element in cosa:
            element.c=c*element.c
            #print element.printTerm()
        products.extend(cosa)
    # finding terms that are equivalent
    products2=simplifyTerms(products)
    return products2  


# The same functions defined above but for the JW transformation.

# In[69]:

# input: operator of the form [i,j,-k,-l] that represents a^{/dag}_i a^{/dag}_j a_k a_l. Any length.
# output: the representation of the operator in the BK form
def returnJWform(operator,nqubits):
    # P, U and F must have the same number of elements = nqubits
    l=len(operator)
    size=2*l
    BKform=[]
    counter=-1
    for a in operator:
        counter += 1
        s1=term(nqubits) 
        s2=term(nqubits)
        j=abs(a)-1
        # case j is even
        s1.pl[j] += 'X'
        #print U[j], P[j], R[j]
        for k in range(j,nqubits):
            s1.pl[k] += 'Z'
        s2.pl[j] += 'Y'
        for k in range(j,nqubits):
            s1.pl[k] += 'Z'
        # annihilator
        if a<0:
            s1.c=0.5
            s2.c=0.5j
        # creator
        elif a>0:
            s1.c=0.5
            s2.c=-0.5j 
        s1.reduceTerm()
        s2.reduceTerm()
        BKform += [s1]
        BKform += [s2]
        #print s1.printTerm()
        #print s2.printTerm()
        #del s1; del s2
    products=[]
    iterables = chunkIt(range(size), size/2)
    counter = 0
    for element in itertools.product(*iterables):
        counter += 1
        s=term(nqubits)
        for t in element:
            for k in range(0,nqubits):
                s.pl[k] += BKform[t].pl[k]
            s.c=s.c*BKform[t].c
        s.reduceTerm()
        products += [s]
        del s
    # finding terms that are equivalent
    products2=simplifyTerms(products)
    return products2    

def getHamiltonianJW(coefficients, terms,nqubits):
    error=10**(-12)
    #coefficients, terms = commutators.GetHamiltonianTerms(molecule, basis, add_conjugates=True)
    products=[]
    conteo=0
    for c,t in zip(coefficients,terms):    
        #print c,t
        cosa = returnJWform(t,nqubits)
        conteo=conteo+len(cosa)
        for element in cosa:
            element.c=c*element.c
            #print element.printTerm()
        products.extend(cosa)
    # finding terms that are equivalent
    products2=simplifyTerms(products)
    return products2


# In[3]:

def reduceHamiltonian(hamiltonian):
    check_term={}
    result=True
    new_nqubits=0
    for n in range(0,hamiltonian[0].l):
        counter=0
        for x in hamiltonian:
            if x.pl[n]==['I'] or x.pl[n]==['Z']:
                counter+=1
        if counter==len(hamiltonian):
            check_term[n]=False
        else:
            check_term[n]=True
        result=result*check_term[n]
        new_nqubits += check_term[n]
    products=[]
    if result==0:
        for x in hamiltonian:
            y=term(int(new_nqubits))
            counter=-1
            y.c=x.c
            for n in range(0,len(check_term)):
                if check_term[n]==True:
                    counter += 1
                    y.pl[counter]=x.pl[n]
            products.append(y)
        products2=simplifyTerms(products)
        return products2
    else:
        return hamiltonian


# #### a) Getting the Hamiltonian for LiH in BK representation (STO-6G basis). 
# Note: commutators.py and the folder from_Jarrod/ are required. We are using one of the printing routines.

# ### Example c: getting the representation of the Unitary Coupled Cluster operators (JW and BK representations)
# This operators are excitation operators minus their conjugates. For example: $a_i^{\dagger}a_j^{\dagger}a_ka_l-a_l^{\dagger}a_k^{\dagger}j_ka_i$ (double excitation). The double excitation is represented as: [i,j,-k,-l].

# In[18]:

def BKsum2(example1,P,U,F,R):
    example2=example1[::-1]
    example2 = [x * -1 for x in example2]
    expression=getHamiltonian([1.0,-1.0],[example1,example2],P,U,F,R)
    return expression

# In[23]:

def BKdiffJW(example1,nqubits):
    example2=example1[::-1]
    example2 = [x * -1 for x in example2]
    expression=getHamiltonianJW([1.0,-1.0],[example1,example2],nqubits)
    return expression

# #### Getting all the single and double operators for a system with 8 molecular orbitals in 8 qubits.

# In[26]:

def createExcitations(nocc,total,N):
    occ=range(1,nocc+1)
    vir=range(nocc+1,total+1)
    operators=[]
    for n in range(1,N+1):
        for cosa1 in itertools.combinations(occ,n):
            for cosa2 in itertools.combinations(vir,n):
                cosita=[]
                cosita.extend(cosa2[::-1])
                cosita.extend([x * -1 for x in cosa1[::-1]])
                operators.append(cosita)
    return operators


# #### Here we also count the number of SWAPS and CNOTs required to simulate the first term in the sum representing every excitation operator (excitation operators are sums of at least 2 products of Pauli matrices in the BK representation :P).

# In[28]:

def BKdiff(example1,P,U,F,R):
    example2=example1[::-1]
    example2 = [x * -1 for x in example2]
    expression=getHamiltonian([1.0,-1.0],[example1,example2],P,U,F,R)
    return expression

# ## From symbolic to matrix representation (testing circuits).
# 
# The purpose of this part is to write circuits. We introduce a dictionary of gates we are going to work with.

# In[29]:

# Input: a list of characters representing gates
# Output: the matrix representation of the gates
def getMatrix(term):
    occ = scipy.sparse.csr_matrix([[0], [1]], dtype=float)
    vir = scipy.sparse.csr_matrix([[1], [0]], dtype=float)
    I = scipy.sparse.csr_matrix([[1, 0], [0, 1]], dtype=complex)
    X = scipy.sparse.csr_matrix([[0, 1], [1, 0]], dtype=complex)
    Y = scipy.sparse.csr_matrix([[0, -1j], [1j, 0]], dtype=complex)
    Z = scipy.sparse.csr_matrix([[1, 0], [0, -1]], dtype=complex)
    SWAP=scipy.sparse.csr_matrix([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex) # S
    H = (1/np.sqrt(2))*scipy.sparse.csr_matrix([[1, 1], [1, -1]], dtype=complex) # H
    CNOT=scipy.sparse.csr_matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex) # C
    RX2=scipy.sparse.linalg.expm(-(np.pi/4)*1j*X) # A
    RY2=scipy.sparse.linalg.expm(-(np.pi/4)*1j*Y) # B
    R_X2=scipy.sparse.linalg.expm((np.pi/4)*1j*X) # a
    R_Y2=scipy.sparse.linalg.expm((np.pi/4)*1j*Y) # b
    CZ=scipy.sparse.csr_matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, -1]], dtype=complex) # E
    
    # Dictionary of gates
    gates={'I':I,'X':X,'Y':Y,'Z':Z,'H':H,'A':RX2,'B':RY2,'a':R_X2,'b':R_Y2,'C':CNOT,'1':occ,'0':vir,'S':SWAP,'E':CZ}
    # gates
    counter=-1
    m=1
    for t in term:
        counter += 1
        m=scipy.sparse.kron(gates[t],m,'csr')
    return m

# get rotations of single qubit operations
def exponentiate(term,alpha):
    cosa=getMatrix(term)
    expop=scipy.sparse.linalg.expm(-1j*(alpha/2)*cosa)
    return expop

# Phase shift gate (CZ(\phi)). This gate is special because is essential for Superconducting Qubit architectures. 
def CZ(phi):
    m=scipy.sparse.csr_matrix([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, np.exp(1j*phi)]], dtype=complex)
    return m

# In[149]:

# input: object of class term representing the operators
# alpha: parameter to be optimized with VQE
# this implements the original circuit for the exponentiation based on CNOTs and SWAPS
def buildExpCircuit(alpha,t):
    # original circuit
    t.getMatrix()
    circuit=scipy.sparse.linalg.expm(-1j*(alpha/2)*t.matrix)
    l=len(t.pl)
    connectivity=np.zeros(l-1)
    maxi=0
    mini=l
    for x in range(0,l-1):
        if (t.pl[x] != ['I']) & (t.pl[x+1] != ['I']) :
            connectivity[x]=1
    for x in range(0,l):            
        if (t.pl[x] != ['I']):
            if x>maxi:
                maxi=x
            if x<mini:
                mini=x
    ## Circuit= part5*part4*part3*part2*part1
    # CNOTs and SWAPs
    part2=1
    part4=1
    for p in range(mini,maxi):
        segment=term(l-1)
        if connectivity[p]==1 or p==(mini):
            segment.pl[p]=['C']
            segment.getMatrix()
            part4=segment.matrix*part4
            part2=part2*segment.matrix
        if connectivity[p]==0 and p!=(mini):
            segment.pl[p]=['S']
            segment.getMatrix()
            part4=segment.matrix*part4
            part2=part2*segment.matrix
    # rotation in the middle
    segment1=term(mini)
    segment2=term(l-mini-1)    
    segment1.getMatrix()
    segment2.getMatrix()
    part3=scipy.sparse.kron(segment2.matrix,scipy.sparse.kron(exponentiate(['Z'],alpha),segment1.matrix))
    # passing to another basis
    segment=term(l)
    for x in t.pl:
        if t.pl[x]==['X']:
            segment.pl[x]=['B']
        if t.pl[x]==['Y']:
            segment.pl[x]=['a']
    segment.getMatrix()
    part5=segment.matrix
    part1=part5.getH()
    #part1=segment.matrix
    #part5=part1.getH()
    unitary=part5*part4*part3*part2*part1
    if np.allclose(unitary.todense(),circuit.todense(),rtol=0.0, atol=1e-07)==True:
        print 'the same'
    else:
        print 'they are different'
    return unitary

# In[154]:

# input: object of class term representing the operators
# alpha: parameter to be optimized with VQE
# this implements the original circuit for the exponentiation based in terms of CZs only, 
# this is for superconducting qubits
def buildExpCircuit2(alpha,t):
    # original circuit
    t.getMatrix()
    circuit=scipy.sparse.linalg.expm(-1j*(alpha/2)*t.matrix)
    l=len(t.pl)
    connectivity=np.zeros(l-1)
    maxi=0
    mini=l
    for x in range(0,l-1):
        if (t.pl[x] != ['I']) & (t.pl[x+1] != ['I']) :
            connectivity[x]=1
    for x in range(0,l):            
        if (t.pl[x] != ['I']):
            if x>maxi:
                maxi=x
            if x<mini:
                mini=x
    ## Circuit= part5*part4*part3*part2*part1
    # CNOTs and SWAPs
    part2=1
    part4=1
    for p in range(mini,maxi):
        sCZ=term(l-1); sCZ.pl[p]=['E']; sCZ.getMatrix()
        if connectivity[p]==1 or p==(mini):
            syh=term(l); syh.pl[p]=['B']; syh.getMatrix()
            CNOT=syh.matrix*sCZ.matrix*syh.matrix.getH()
            part4=CNOT*part4
            part2=part2*CNOT
        if connectivity[p]==0 and p!=(mini):
            syh=term(l); syh.pl[p]=['B']; syh.getMatrix()
            CNOT=syh.matrix*sCZ.matrix*syh.matrix.getH()
            #syh2=term(l); syh2.pl[p+1]=['B']; syh2.getMatrix()
            #CNOTi=syh2.matrix*sCZ.matrix*syh2.matrix.getH()
            syh=term(l); syh.pl[p]=['B']; syh.getMatrix()
            syh2=term(l); syh2.pl[p]=['b']; syh2.pl[p+1]=['B']; syh2.getMatrix()
            SWAP=syh.matrix*sCZ.matrix*syh2.matrix*sCZ.matrix*syh2.matrix.getH()*sCZ.matrix*syh.matrix.getH()    
            #SWAP=CNOT*CNOTi*CNOT
            part4=SWAP*part4
            part2=part2*SWAP
    # rotation in the middle
    segment1=term(mini)
    segment2=term(l-mini-1)    
    segment1.getMatrix()
    segment2.getMatrix()
    part3=scipy.sparse.kron(segment2.matrix,scipy.sparse.kron(exponentiate(['Z'],alpha),segment1.matrix))
    # passing to another basis
    segment=term(l)
    for x in t.pl:
        if t.pl[x]==['X']:
            segment.pl[x]=['B']
        if t.pl[x]==['Y']:
            segment.pl[x]=['a']
    segment.getMatrix()
    part5=segment.matrix
    part1=part5.getH()
    #part1=segment.matrix
    #part5=part1.getH()
    unitary=part5*part4*part3*part2*part1
    if np.allclose(unitary.todense(),circuit.todense(),rtol=0.0, atol=1e-06)==True:
        print 'the same'
    else:
        print 'they are different'
    return unitary

# input: object of class term representing the operators
# alpha: parameter to be optimized with VQE
def buildExpCircuit3(alpha,t):
    # original circuit
    #t.getMatrix()
    #circuit=scipy.sparse.linalg.expm(-1j*(alpha/2)*t.matrix)
    l=len(t.pl)
    connectivity=np.zeros(l-1)
    maxi=0
    mini=l
    # calculates connectivity
    for x in range(0,l-1):
        if (t.pl[x] != ['I']) & (t.pl[x+1] != ['I']) :
            connectivity[x]=1
    for x in range(0,l):            
        if (t.pl[x] != ['I']):
            if x>maxi:
                maxi=x
            if x<mini:
                mini=x
    ## Circuit= part5*part4*part3*part2*part1
    # CNOTs and SWAPs
    cnotbackseg=[]
    for p in range(mini,maxi):
        segment=term(l-1)
        if connectivity[p]==1 or p==(mini):
            segment.pl[p]=['C']
            #segment.getMatrix()
            cnotbackseg.append(segment)
        if connectivity[p]==0 and p!=(mini):
            segment.pl[p]=['S']
            #segment.getMatrix()
            cnotbackseg.append(segment)

    # rotation in the middle
    rseg=term(l,alpha=alpha)
    rseg.pl[mini]=['Rz']
    rseg.getMatrix()
    # passing to another basis
    changebbackseg=term(l)
    for x in t.pl:
        if t.pl[x]==['X']:
            changebbackseg.pl[x]=['B']
        if t.pl[x]==['Y']:
            changebbackseg.pl[x]=['a']
    #changebbackseg.getMatrix()
    result=[changebbackseg,cnotbackseg,rseg]
    #unitary=changebback*cnotbackseg*rseg*cnotbackseg.getH()*changebbackseg.getH()
    return result

# input: object of class term representing the operators
# alpha: parameter to be optimized with VQE
def getExpCircuit4(alpha,t):
    l=len(t.pl)
    # print 'This is the size of the term:',l
    connectivity=np.zeros(l-1)
    maxi=0
    mini=l
    for x in range(0,l-1):
        if (t.pl[x] != ['I']) & (t.pl[x+1] != ['I']) :
            connectivity[x]=1
    for x in range(0,l):            
        if (t.pl[x] != ['I']):
            if x>maxi:
                maxi=x
            if x<mini:
                mini=x
    ## Circuit= part5*part4*part3*part2*part1
    # CNOTs and SWAPs
    
    cnotforwardseg=[]

    for p in range(mini,maxi):

        s1=term(l)
        s1.pl[p]=['B']
        
        s2=term(l)
        s2.pl[p]=['b']
        s2.pl[p+1]=['B']
        
        s3=term(l-1)
        s3.pl[p]=['E']
        
        s4=term(l)
        s4.pl[p]=['B']
        s4.pl[p+1]=['b']
        
        s5=term(l)
        s5.pl[p]=['b']
        # always starts with a single rotation        
        if p==mini:
            cnotforwardseg.append(s1)
        # adding CNOT
        if connectivity[p]==1 or p==(mini):
            cnotforwardseg.append(s3)
        # adding SWAP        
        if connectivity[p]==0 and p!=(mini):
            cnotforwardseg.append(s3)
            cnotforwardseg.append(s2)
            cnotforwardseg.append(s3)
            cnotforwardseg.append(s4)
            cnotforwardseg.append(s3)
        # connecting or closing sequence
        if p<maxi-1:
            cnotforwardseg.append(s2)
        else:
            cnotforwardseg.append(s5)

    # rotation in the middle
    rseg=term(l,alpha=alpha)
    rseg.pl[mini]=['Rz']
    rseg.getMatrix()
    # passing to another basis
    changebbackseg=term(l)
    for x in t.pl:
        if t.pl[x]==['X']:
            changebbackseg.pl[x]=['B']
        if t.pl[x]==['Y']:
            changebbackseg.pl[x]=['a']
    #changebbackseg.getMatrix()
    result=[changebbackseg,cnotforwardseg,rseg]
    #unitary=changebback*cnotbackseg*rseg*cnotbackseg.getH()*changebbackseg.getH()
    return result

