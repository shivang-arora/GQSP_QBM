
import numpy as np

import torch
from torch.nn.functional import conv1d, pad
from torch.fft import fft
from torchaudio.transforms import FFTConvolve
import time
import numpy as np
import matplotlib.pyplot as plt
import pennylane as qml
from helper import *
from scipy.stats import unitary_group


def GQSP(U,poly: torch.Tensor,plot=False,QEVT=False) -> np.ndarray:
    #global dev
    '''
    Function that returns output state as a result of simulating pennylane circuit implementing GQSP on unitary U and polymonial coefficients (poly).

    Arguments:
    U : Unitary martrix as numpy array
    poly (torch.Tensor)  : Coefficients for the polynomial in increasing order
    plot (bool)(default=False): Draws the corresponding GQSP circuit
    QEVT (bool) : Set True for QEVT circuit with qubitization operators.
    Returns :
    
    The first column of the transformed unitary.
    '''
    
    
    dim=int(np.log2(len(U[0])))
    
    target_wires=range(1,dim+1)
    
    #Promote to a controlled unitary with control as qubit 0

    A=qml.ControlledQubitUnitary(U, control_wires=0 , wires=target_wires)

    #print('Polynomial Coefficients: ', poly)

    dev=qml.device('default.qubit',wires=len(A.wires))
    @qml.qnode(device=dev)
    
    def Circuit(poly,A):
        '''
        Function that simulates the pennylane circuit for QSP.
        Args:
        poly
        A : controlled unitary matrix with control as qubit 0.
            
        Returns:
        qml.state()
        '''
    
        # Generate the complimentary polynomial Q based on P, that satusfies the axioms for GQSP.
        
        S=np.array([torch.detach(P_polynomial(poly)),torch.detach(Q_polynomial(poly))],dtype=np.complex128)

        #Get angles for the GQSP sequence based on https://github.com/Danimhn/GQSP-Code
        
        theta,phi,lamb=Compute_parameters(S,len(poly)-1)
       
        
        ## Generated angles for QEVT are just phase shifted angles phi of the GQSP sequence
        if QEVT==True:
           
        


            
            # Add pi to each element except the last 
            modified_elements = tuple(x + np.pi for x in phi[:-1])

        
            phi =modified_elements + phi[-1:]
            
                
                
        qml.QubitUnitary(Rotation(theta[0],phi[0],lamb),wires=0)
        
        for i in range(1,len(theta)):
            
            qml.PauliX(wires=0)
            qml.QubitUnitary(A.matrix(),wires=(A.wires),id='A')
            qml.PauliX(wires=0)
            
            
            if QEVT==True:
                # for QEVT, the Qubitized unitary is obtained by adding reflection operators. 
                
                diag=np.ones(len(A.matrix()[0]))
                
                diag[0:2*(dim-1)]=-1
                
                qml.DiagonalQubitUnitary(diag,wires=range(0,dim+1))
               
            qml.QubitUnitary(Rotation(theta[i],phi[i],0),wires=0)

        
        
        return qml.state()
        
    

    
    
    
    
    
    
    state=Circuit(poly,A)
    
    if QEVT==True:
        
        result=state.reshape(2,2,-1)[0][0]
        #result=state
    else:
        #pass
        #result=state
        result=state.reshape(2,-1)[0]
    
    if plot==True:
        qml.draw_mpl(Circuit)(poly,A)
    
    
    return result

# Exact calculation
