
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
import itertools





from pyqsp.poly import PolyOneOverX
from pyqsp.poly import PolyCosineTX
import scipy
from pyqsp.poly import PolyTaylorSeries   # built-in since v0.1.4




class GQEVT:
    
    def __init__(self, n=30, pre_fac=-1.0, method='GQSP'):
        

        def generate_coeff_exp(n=30,beta=-1.0):
            '''
    
            Generate coefficients in the Chebychev basis for the function 1/x.
            '''
           # e^{β x} on  x∈[-1,1]
            samples = 2*n         # slight oversampling avoids aliasing
    
    # build the Chebyshev expansion
            poly = PolyTaylorSeries()
            c = poly.taylor_series(
                    func          = lambda x: np.exp(beta * x),  # or np.exp(1j*t*x) for e^{i t x}
                    degree        = n,
                    chebyshev_basis = True,   # <- ask for Chebyshev—not monomial—coeffs
                    cheb_samples  = samples,  # # of interpolation nodes
                    return_scale  = False)    # only need the coefficients
            
          # c[0], c[1], … , c[n]  (d
            return c.coef,1

        self.n=n
        self.pre_fac=pre_fac

        self.poly,self.scale=generate_coeff_exp(n,pre_fac)
        #self.poly=torch.from_numpy(self.poly)
        self.method = method
              
        self._compute_angles(method)
        polynomial=torch.from_numpy(self.poly).clone()
    

        ft = fft(polynomial)
        
            # Normalize P
        P_norms = ft.abs()
        self.p_scale=torch.max(P_norms)
        self.scale=self.scale/self.p_scale

    
    
    
    def generate_coeff(kappa=3,epsilon=0.1):
        '''
    
        Generate coefficients in the Chebychev basis for the function 1/x.
        '''
        
        poly_generator = PolyOneOverX(verbose=False)
        #poly_generator = PolyCosineTX(verbose=False)
      
    
        coefficients,scale = poly_generator.generate(kappa=kappa, epsilon=epsilon, return_scale=True,chebyshev_basis=True,return_coef=True)
        return coefficients,scale

    
    def _BlockEncode(self,A,n):
    
        encoding=qml.BlockEncode(A, wires=range(n))
        
        return encoding
    
    def _compute_angles(self, method):
        if method == 'gradient':
            p_poly = P_polynomial(self.poly)
            q_poly, _ = Q_polynomial(self.poly)

            S = torch.stack([
                p_poly.to(torch.float64),
                q_poly.to(torch.float64)
            ])

            self.angles = Compute(S, len(self.poly) - 1)
        else:
            #poly_np=P_polynomial(torch.tensor(self.poly))
            angles = qml.poly_to_angles(self.poly, "GQSP")
            
            self.angles= [tuple(torch.from_numpy(angles[0])),tuple(torch.from_numpy(angles[1])),torch.tensor(angles[2][0])]
            
    def circuit(self,angles,direct='mat'):
            
      
       if direct=='full':
        
            theta, phi, lamb = angles

            modified_phi = tuple(p + torch.pi for p in phi[:-1]) + (phi[-1],)

            qml.QubitUnitary(TorchRotation(theta[0], modified_phi[0], lamb), wires=0)

            for i in range(1, len(theta)):
                qml.PauliX(wires=0)
                qml.QubitUnitary(self.A.matrix(), wires=self.A.wires, id='Block')
                qml.PauliX(wires=0)

                diag = torch.ones(2 ** (self.dim + 1), dtype=torch.complex128)
                diag[:2 * (self.dim - 1)] *= -1
                qml.DiagonalQubitUnitary(diag, wires=range(self.dim + 1), id='Refl')

                qml.QubitUnitary(TorchRotation(theta[i], modified_phi[i], torch.tensor(0.0)), wires=0)

       else:
          
           qml.QubitUnitary(self.block_block_inv,wires=range(self.dim+1))

    def measure_qnode(self,direct='mat'):
        self.dev = qml.device("default.qubit", wires=len(self.A.wires))
        
      
        @qml.qnode(self.dev, interface="torch")
        def measurement_circuit(angles):
            self.circuit(angles,direct)
            return qml.state()   

        return measurement_circuit(self.angles)
        
        

    
    def get_circuit(self):
        return self.circuit(self.angles)

    def plot_circuit(self):
        qml.draw_mpl(self.circuit)(self.angles)

    def build(self,H):
        
        self.H=H
        dim=int(np.log2(len(H[0])))
        U=self._BlockEncode(H,dim+1)
        BlockU=qml.matrix(U)
        
        self.U = BlockU
        
        inv=scipy.linalg.expm(-self.H)


        self.dim = int(torch.log2(torch.tensor(len(BlockU), dtype=torch.float64)).item())
        self.target_wires = list(range(1, self.dim + 1))

        self.A = qml.ControlledQubitUnitary(self.U, control_wires=0, wires=self.target_wires)
        self.block_block_inv=self._BlockEncode(inv,dim+2)
        self.norm = self.block_block_inv.hyperparameters['norm']
        self.block_block_inv =qml.matrix( self.block_block_inv)
        #self.block_block_inv=qml.matrix(self._BlockEncode(qml.matrix(block_inv),dim+2))
