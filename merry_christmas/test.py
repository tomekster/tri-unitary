import sys
import os
import numpy as np
import scipy.sparse as sparse
from numpy import linalg as LA
import json
import warnings
warnings.filterwarnings("ignore")
import time
from scipy.optimize import linear_sum_assignment

from SpinLibraryCherynePython3 import *

#### Required functions
    
def Utostate(U,legswap):
    #takes as inpu 2^L x 2^L matrix
    #reshapes it into 4^L x 1 vector
    #shuffles legs according to vec
    L=len(legswap)
    legs=[int(2) for i in legswap]
    n=np.log(np.shape(U)[0])/np.log(2)
    psi=np.reshape(U,legs)
    #psi=np.reshape(U/np.sqrt(2**n),legs)
    psi=np.transpose(psi,legswap)
    psi=np.reshape(psi,(2**L,1))
    psi=psi/linalg.norm(psi,2)
    #print(np.round(linalg.norm(psi,2),2))
    return psi

def xcutop_to_xcutstate(L,x):
    lin=np.arange(0,L)
    lout=np.arange(L,2*L)
    l1=np.concatenate((lin[0:x],lout[0:x]))
    l2=np.concatenate((lin[x:],lout[x:]))
    return np.concatenate((l1,l2)).tolist() #we want list as output
    
def SW_generator_firstorder(L,hlist):
        dim=2**L
        A=np.zeros((L,dim,dim))
        for i in range(dim):
            sigma=de2bi(i,L)
            Zsigma=1-2*sigma
            ZZsigma=Zsigma[1:]*Zsigma[:-1] #obc
            Esigma=-.25*ZZsigma.sum(axis=0)+.5*np.dot(Zsigma,hlist)
            for n in range(L):
                sigma_n=sigma.copy()
                sigma_n[n]=np.mod(sigma_n[n]+1,2)
                j=bin2int(sigma_n)
                Zsigma_n=1-2*sigma_n
                ZZsigma_n=Zsigma_n[1:]*Zsigma_n[:-1]
                Esigma_n=-.25*ZZsigma_n.sum(axis=0)+.5*np.dot(Zsigma_n,hlist)
                A[n,i,j]=1/(Esigma-Esigma_n)
        Atot=np.sum(A,axis=0)
        return Atot
    
def get_opweight_n1n2_local(tau,L):
   ##### Explicit construction 
    I,X,Y,Z=paulixyz();
    S = [I,X,Y,Z]/np.sqrt(2)
    labels = ['I",''X', 'Y', 'Z']

    ## 0 site ops
    #P0_fullsum=np.abs(np.trace(np.eye(2**L)/(np.sqrt(2))**(L)@(tau.conj().T)))**2

    ## 1 site ops
    P1=np.zeros(num_ops(1,L))
    count=0
    for i in range(1,L+1):
        for a in range(1,3+1):
            op=(mkron(np.eye(2**(i-1)),S[a],np.eye(2**(L-i)))/(np.sqrt(2)**(L-1)))
            P1[count]=np.abs(np.trace(op@(tau.conj().T)))**2
            count=count+1

    ## 2 site ops
    P2=np.zeros(3**2*L)

    count=0
    for i in range(1,L):
        j=np.mod(i+1,L+1)
        for a in range(1,3+1):
            for b in range(1,3+1):
                op=(mkron(np.eye(2**(i-1)),S[a],np.eye(2**(j-i-1)),S[b],np.eye(2**(L-j)))/(np.sqrt(2)**(L-2)))
                P2[count]=np.abs(np.trace(op@(tau.conj().T)))**2
                count=count+1
    i=L
    j=1
    for a in range(1,3+1):
            for b in range(1,3+1):
                op=(mkron(np.eye(2**(j-1)),S[a],np.eye(2**(i-j-1)),S[b],np.eye(2**(L-i)))/(np.sqrt(2)**(L-2)))
                P2[count]=np.abs(np.trace(op@(tau.conj().T)))**2

    return P1,P2
    

#### cechkpoint class

class Checkpoint:
    ### initialize
    def __init__(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path
        if not os.path.exists(checkpoint_path):
            os.mkdir(checkpoint_path)

        version_file = 'version'
        self.version_path = os.path.join(checkpoint_path, version_file)

        if not os.path.exists(self.version_path):
            self.save_version(0)

    def get_version(self):
        with open(self.version_path, 'r') as f:
            line = f.readline()
            return int(line[0])

    def save_version(self, version):
        with open(self.version_path, 'w') as f:
            f.write(str(version))

    def save(self, state):
        new_version = 1 - self.get_version()
        for key in State.fields:
            filepath = os.path.join(checkpoint_path, '{}_{}.npy'.format(new_version, key))
            np.save(filepath, state[key])
        self.save_version(new_version)

    def load(self):
        current_version = self.get_version()
        try:
            fields_dict = {}
            for f in State.fields:
                print('Loading: ', f)
                fields_dict[f] = np.load(os.path.join(self.checkpoint_path, '{}_{}.npy'.format(current_version, f)))
        except Exception as e:
            print(e)
            return None
        return State(fields_dict)

#### state class

class State:
    fields = ['nmin','SUmidcut_SW','SUmidcut_ED','SUmidcut_SWED','P1','P2']
    scalars = ['nmin']

    def __init__(self, fields_dict=None):

        if fields_dict:
            for scalar in State.scalars:
                if type(fields_dict[scalar]) is not int:
                    fields_dict[scalar] = int(fields_dict[scalar])
        self.fields_dict = fields_dict 
        

    def __getitem__(self, key):
        return self.fields_dict[key]

    def __setitem__(self, key, val):
        self.fields_dict[key] = val

    def reset(self, L, chunk_size):

        self.fields_dict = {'nmin': 0,\
                            'SUmidcut_SW': np.zeros((chunk_size,len(gvec))),\
                            'SUmidcut_ED': np.zeros((chunk_size,len(gvec))),\
                            'SUmidcut_SWED': np.zeros((chunk_size,len(gvec))),\
                            'P1': np.zeros((chunk_size,len(gvec),L,3*L)),\
                            'P2': np.zeros((chunk_size,len(gvec),L,3**2*L))}
        return self

    
### reading in parameters
def parse_params_json(params_json):
    params = json.loads(params_json)

    L = params["L"]

    chunk_id = params['chunk_id']
    chunk_id = str(chunk_id).zfill(4)
    chunk_size = params['chunk_size']
    directory = params['directory']
    return L, chunk_id, chunk_size, directory

############################################################

# Takes as input 1 parameters: L
params_json = sys.argv[1]
L, chunk_id, chunk_size, directory = parse_params_json(params_json)
#directory='.'
#L=8
#chunk_id=3
#chunk_size=10

## Stuff that needs to be done wherever in code we are (whether it's starting from scratch or loading data)
dim=2**L
s0_list, x, y, z = gen_s0sxsysz(L)
W=1.
J=1.
gvec=np.concatenate((np.arange(0,0.3,0.01),np.arange(0.3,0.5,0.05),np.arange(0.5,1,0.1)))
x_cutop=L//2

## Stuff for specific run
run_identifier = 'L%d_chunk_id_%s'%(L, chunk_id)
checkpoint_path = os.path.join(directory, run_identifier)

# Initialize state 
checkpoint = Checkpoint(checkpoint_path)
state=checkpoint.load()
if not state: # If failed to read state for the file - reset state
    print("Failed to load checkpoint, reseting state")
    state = State().reset(L, chunk_size)

nmin = state['nmin']
SUmidcut_SW = state['SUmidcut_SW']
SUmidcut_ED = state['SUmidcut_ED']
SUmidcut_SWED = state['SUmidcut_SWED']
P1 = state['P1']
P2 = state['P2']


# Here we have init default state OR we have state from checkpoint
print(nmin)

times=[]

for runs in range(nmin,chunk_size):
    ## Hdiag
    hlist=np.random.uniform(-W,W,L)
    Hdiag= -.25*J*(gen_nn_int(z,[],'obc')) + .5*gen_onsite_field(z,hlist);
    ## Hkick
    Hkick= .5*gen_onsite_field(x,np.ones(L));

    for g_ind,g in enumerate(gvec):
        ## SW 
        Atot=SW_generator_firstorder(L,hlist) #SW generator to O(g)
        Omega=expm(-g*Atot) #SW unitary
            ## Operator space entanglement of U=Omega=e^{-Atot}=SW unitary
        psi=Utostate(Omega,xcutop_to_xcutstate(L,x_cutop)).flatten()
        SUmidcut_SW[runs,g_ind]=EntanglementEntropy(psi,2*x_cutop,dim=2)

        ## ED
        H=Hdiag+g*Hkick
        evals_g,evecs_g = LA.eigh(H.todense()) #get exact eigenstates
            ## Operator space entanglement of U = sorted eigenvectors (incr. energy)
        psi=Utostate(np.asarray(evecs_g),xcutop_to_xcutstate(L,x_cutop)).flatten()
        SUmidcut_ED[runs,g_ind]=EntanglementEntropy(psi,2*x_cutop,dim=2)

        
        ## Overlap between ED and SW generator
        Mab=np.abs(np.conj(Omega).T@evecs_g)
        Mab=np.asarray(Mab)

        ## Hungarian algorithm to determine a ->b asssignment
        row_ind, col_ind = linear_sum_assignment(-Mab)
        U=np.asarray(evecs_g[:,col_ind.tolist()]) #reshuffle exact eigenstatess according to largest overlap with SW        
        
        ## Operator space entanglement of (reshuffled) U 
        psi=Utostate(U,xcutop_to_xcutstate(L,x_cutop)).flatten()
        SUmidcut_SWED[runs,g_ind]=EntanglementEntropy(psi,2*x_cutop,dim=2)
        
        ## Operator weight
        for i in range(L):
            tau=np.conj(U).T@z[i]@U
            #a=time.time()
            P1_temp,P2_temp=get_opweight_n1n2_local(tau,L)
            #b=time.time()
            #times.append(b-a)
            #print([runs,g,i,b-a])
            P1[runs,g_ind,i,:]=P1_temp
            P2[runs,g_ind,i,:]=P2_temp
            
        print([runs,g])
        
    state = State({'SUmidcut_SW': SUmidcut_SW,'SUmidcut_ED': SUmidcut_ED, 'SUmidcut_SWED': SUmidcut_SWED, 'P1':P1, 'P2':P2, 'nmin': np.array([runs])})
    checkpoint.save(state)
    
# SAVE FINAL RESULT
state['SUmidcut_SW'] = SUmidcut_SW
state['SUmidcut_ED'] = SUmidcut_ED
state['SUmidcut_SWED'] = SUmidcut_SWED
state['P1'] = P1
state['P2'] = P2
state['nmin']: np.array([runs+1])
checkpoint.save(state)