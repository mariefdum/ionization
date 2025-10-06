r"""
Python module for diagonalizing transmon at symmetry point n_g = 0 and offset phase of zero
"""
################ Importing ################
import numpy as np
from numpy import pi as pi

import scipy as sp

import qutip as qt

import warnings

import copy


class HarmonicTransmon:
    def __init__(self, Transmon_params):
        """
        Initialize the HarmonicTransmon class.

        Parameters:
        EJ : list of floats
            Josephson energies for each harmonic term.
        EC : float
            Charging energy.
        ng : float
            Offset charge.
        ncut : int
            Charge basis cutoff.
        truncated_dim : int, optional
            Dimension after truncation.
        """
        self.T_params = copy.deepcopy(Transmon_params)
        self.EJ = self.T_params['E_J']
        self.EC = self.T_params['E_C']
        self.ng = self.T_params['n_g']
        self.ncut = self.T_params['max_charge']
        self.dimension = self.hilbertdim()
        self.n_harmonics = len(self.EJ)
        self.truncated_dim = self.T_params['trunc_charge']
        self.truncated_operators()#Generated truncated and diagonalized operators
        
    def hilbertdim(self):
        """
        Returns the dimension of the Hilbert space.
        """
        return 2 * self.ncut + 1

    def cos_m_phi_operator(self, m):
        """
        Returns the operator cos(m*phi) in the charge basis.

        Parameters:
        m : int
            Harmonic index.
        """
        cos_op = np.zeros((self.dimension, self.dimension))
        for n in range(self.dimension - m):
            cos_op[n, n + m] = 0.5
            cos_op[n + m, n] = 0.5
        for n in range(m):
            cos_op[n,self.dimension-m+n] = 0.5
            cos_op[self.dimension-m+n,n] = 0.5
            
        return cos_op
        
    def hamiltonian(self):
        """
        Returns the Hamiltonian in the charge basis.
        """
        self.n_operator=np.diag(np.linspace(-self.ncut, self.ncut, self.dimension))
        hamiltonian_mat = 4 * self.EC * (self.n_operator - np.eye(self.hilbertdim())*self.ng) ** 2
        
        for m in range(1, self.n_harmonics + 1):
            hamiltonian_mat += -self.EJ[m-1] * self.cos_m_phi_operator(m)
        return hamiltonian_mat
    
    def truncated_operators(self):
        """
        Returns the truncated Hamiltonian in the eigen basis.
        """
        self.eigvals, self.eigvecs = sp.linalg.eigh(self.hamiltonian())
        # if (self.ng)%0.5==0 :
        #    self.eigvecs = self.Sort()
            
        self.Hdiagonal=self.basis_change(self.eigvecs,self.hamiltonian())[0:self.truncated_dim,0:self.truncated_dim]
        self.n_diagonal=self.basis_change(self.eigvecs,self.n_operator)[0:self.truncated_dim,0:self.truncated_dim]
        self.n_diagonal=1/2*(self.n_diagonal+np.conj(np.transpose(self.n_diagonal)))

    def basis_change(self,newbasis,operator):
        """
        Returns the operator in the new basis.
        """
        return np.dot(np.conj(np.transpose(newbasis)),np.dot(operator,newbasis))

    def Sort(self):
        
        Eigvecs_new = []
        if self.ng%1==0:
            for q in range(2*self.ncut+1):
                if q%2==0:
                    new_eigv_t_e=np.hstack([np.flip(self.eigvecs[self.ncut+1:,q]),self.eigvecs[self.ncut:,q]])
                    #new_eigv_t_e[self.ncut] = self.eigvecs[self.ncut,q]
                    norm = sp.linalg.norm(new_eigv_t_e)
                    Eigvecs_new.append(new_eigv_t_e/norm)
                elif q%2==1:
                    new_eigv_t_o=np.hstack([-np.flip(self.eigvecs[self.ncut+1:,q]),self.eigvecs[self.ncut:,q]])
                    #new_eigv_t_o[self.ncut] = 0
                    norm = sp.linalg.norm(new_eigv_t_o)
                    Eigvecs_new.append(new_eigv_t_o/norm)
            return np.array(Eigvecs_new).T
        else:
            for q in range(2*self.ncut+1):
                if q%2==0:
                    new_eigv_t_e=np.hstack([np.real(self.eigvecs[:self.ncut+1,q]),np.flip(np.real(self.eigvecs[1:self.ncut+1,q]))])
                    norm = sp.linalg.norm(new_eigv_t_e)
                    Eigvecs_new.append(new_eigv_t_e/norm)
                elif q%2==1:
                    new_eigv_t_o=np.hstack([np.real(self.eigvecs[:self.ncut+1,q]),-np.flip(np.real(self.eigvecs[1:self.ncut+1,q]))])
                    norm = sp.linalg.norm(new_eigv_t_o)
                    Eigvecs_new.append(new_eigv_t_o/norm)
            return np.array(Eigvecs_new).T

class CoupledHarmonicTransmonResonator:
    def __init__(self, System_params):
        self.System_params = copy.deepcopy(System_params)

        self.hmon = HarmonicTransmon(self.System_params)
        self.fr_b = self.System_params['omega_r']
        self.g = self.System_params['g']
        self.r_levels = self.System_params['max_fock']

    def hamiltonian(self):
        dim_hmon = self.hmon.truncated_dim
        dim_res = self.r_levels

        H_t = np.kron(self.hmon.Hdiagonal, np.eye(dim_res))
        H_r = np.diag([r * self.fr_b for r in range(dim_res)])
        H_r = np.kron(np.eye(dim_hmon), H_r)
        a = np.diag(np.sqrt(np.arange(1, dim_res)), 1)
        a_dagger = np.diag(np.sqrt(np.arange(1, dim_res)), -1)
        V = self.g * np.kron(self.hmon.n_diagonal-np.eye(dim_hmon)*self.hmon.ng, (a + a_dagger))
        return H_t + H_r + V

    def diagonalize(self):
        H = self.hamiltonian()
        evals, evecs = sp.linalg.eigh(H)
        self.evals = evals - evals[0]  # Shift energies to have the ground state at 0
        self.evecs = evecs
    
    def get_label(self,eigvecs,Initial_basis):
        ''' 
        Computes max overlap between two set of states
        '''
        labels=[]
        for element in Initial_basis:
            prod = np.abs(np.dot(np.conj(eigvecs).transpose(2,1,0), element))**2
            labels.append(np.argmax(prod))
        return labels
    
    def sort_eigenenergies(self, q_photons=15, r_photons=1,max_photons=-1):
        self.diagonalize()
        bare_state_0_Photon = [np.kron(np.eye(self.hmon.truncated_dim)[:, t], np.eye(self.r_levels)[:, 0]).reshape(-1,1) for t in range(q_photons+1)]
        bare_state_1_Photon = [np.kron(np.eye(self.hmon.truncated_dim)[:, t], np.eye(self.r_levels)[:, 1]).reshape(-1,1) for t in range(q_photons+1)]
        indices_0_Photon=self.get_label(self.evecs.reshape(self.hmon.truncated_dim*self.r_levels,self.hmon.truncated_dim*self.r_levels,1),bare_state_0_Photon)
        indices_1_Photon=self.get_label(self.evecs.reshape(self.hmon.truncated_dim*self.r_levels,self.hmon.truncated_dim*self.r_levels,1),bare_state_1_Photon)
        self.dressed_qubit=np.diff(self.evals[indices_0_Photon])
        self.dressed_resonator=self.evals[indices_1_Photon]-self.evals[indices_0_Photon]

