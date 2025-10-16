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


class Transmon:
    
    def __init__(self, Transmon_params):
        self.T_params = copy.deepcopy(Transmon_params)
        
        self.E_C = self.T_params['E_C']

        #Number of charge states used to diagonalize the transmon
        if self.T_params.get('max_charge') == None:
            self.max_charge = 200
        elif type(self.T_params['max_charge']) != int :
            raise TypeError('Max number of charge states used to diagonalize the transmon should be an integer')
        else: 
            self.max_charge = self.T_params['max_charge']

        #Truncating and writing operators in eigenbasis
        if self.T_params.get('n_trunc') == None:
            self.n_trunc = 31
        else: 
            self.n_trunc = self.T_params['n_trunc']
        
        #Building the usual charge, cos and sin operators
        self.n_t = qt.Qobj(np.diag(np.arange(2*self.max_charge+1)-self.max_charge))
                                
        self.cos_t = 0.5*(qt.Qobj((np.diag(np.ones(2*self.max_charge),1)+np.diag(np.ones(2*self.max_charge),-1)))
                       +qt.basis(2*self.max_charge+1,2*self.max_charge)*qt.basis(2*self.max_charge+1,0).dag()
                       +qt.basis(2*self.max_charge+1,0)*qt.basis(2*self.max_charge+1,2*self.max_charge).dag())
        self.sin_t = 0.5*1j*(qt.Qobj((np.diag(np.ones(2*self.max_charge), 1) - np.diag(np.ones(2*self.max_charge), -1)))
                             +qt.basis(2*self.max_charge+1,2*self.max_charge)*qt.basis(2*self.max_charge+1,0).dag()
                             -qt.basis(2*self.max_charge+1,0)*qt.basis(2*self.max_charge+1,2*self.max_charge).dag())
        
        #Building Hamiltonian and getting Eigenstates and eigenvalues
        if self.T_params.get('n_g') != None :  
            self.n_g = self.T_params['n_g']
        else:
            self.n_g = 0

        if self.T_params.get('phi_ext') != None :  
            self.phi_ext = self.T_params['phi_ext']
        else:
            self.phi_ext = 0

        # no higher harmonics
        if self.T_params.get('E_J') != None:  
            self.E_J_a_m = [self.T_params.get('E_J')]
            self.E_J_b_m = [0 for _ in range(len(self.E_J_a_m))]
        else:
            self.E_J_a_m = self.T_params['E_J_a_m']
            if self.T_params.get('E_J_b_m') != None:  
                self.E_J_b_m = [0 for _ in range(len(self.E_J_a_m))]
            else:
                self.E_J_b_m = self.T_params['E_J_b_m']


        
        self.H_t = 4*self.E_C*(self.n_t-qt.qeye(2*self.max_charge+1)*self.n_g)**2
        for index_E_J_m in range(len(self.E_J_a_m)):
            m = index_E_J_m+1
            self.H_t -= (self.E_J_a_m[index_E_J_m]+self.E_J_b_m[index_E_J_m]*np.cos(m*self.phi_ext))*self.cos_m_t(m)
            self.H_t -= (self.E_J_b_m[index_E_J_m]*np.sin(m*self.phi_ext))*self.sin_m_t(m)
        
        self.E, self.Eigvecs = (self.H_t).eigenstates()

        if (self.n_g%0.5) == 0:
            self.Eigvecs = self.Sort_higher_harmonics()

        # SQUID=False
        # # Higher harmonics with a SQUID
        # if self.T_params.get('phi_ext') != None and self.T_params.get('E_J_b_m') != None :
        #     if np.abs(self.T_params.get('E_J_b_m')[0]) != 0 :
        #         SQUID=True
        #         self.E_J_a_m = self.T_params['E_J_a_m']
        #         self.E_J_b_m = self.T_params['E_J_b_m']
        #         self.phi_ext = self.T_params['phi_ext']
    
        #         self.H_t = 4*self.E_C*(self.n_t-qt.qeye(2*self.max_charge+1)*self.n_g)**2
        #         for index_E_J_m in range(len(self.E_J_a_m)):
        #             m=index_E_J_m+1
        #             self.H_t-=(self.E_J_a_m[index_E_J_m]+self.E_J_b_m[index_E_J_m]*np.cos(m*self.phi_ext))*self.cos_m_t(m)
        #             self.H_t-=(self.E_J_b_m[index_E_J_m]*np.sin(m*self.phi_ext))*self.sin_m_t(m)
                
        #         self.E, self.Eigvecs = (self.H_t).eigenstates()

        #         if (self.n_g%0.5)==0:
        #             self.Eigvecs = self.Sort_higher_harmonics()

        # # One junction
        # if SQUID==False:

        #     # Higher harmonics, no flux dependance 
        #     if self.T_params.get('E_J_a_m') != None:
        #         self.E_J_a_m = self.T_params['E_J_a_m']
        #         self.H_t = 4*self.E_C*(self.n_t-qt.qeye(2*self.max_charge+1)*self.n_g)**2
        #         for index_E_J_m in range(len(self.E_J_a_m)):
        #             m=index_E_J_m+1
        #             self.H_t-=self.E_J_a_m[index_E_J_m]*self.cos_m_t(m)
        #         self.E, self.Eigvecs = (self.H_t).eigenstates()
        #         if (self.n_g%0.5)==0:
        #             self.Eigvecs = self.Sort_higher_harmonics()

        #     # n_g is 0 or 0.5
        #     elif (self.n_g%0.5)==0:
        #         self.E_J = self.T_params['E_J']
        #         self.H_t = 4*self.E_C*(self.n_t-qt.qeye(2*self.max_charge+1)*self.n_g)**2-self.E_J*self.cos_t
        #         self.E, self.Eigvecs = (self.H_t).eigenstates()
        #         self.Eigvecs = self.Sort_higher_harmonics()
            
        #     else:
        #         self.E_J = self.T_params['E_J']
        #         self.H_t = 4*self.E_C*(self.n_t-qt.qeye(2*self.max_charge+1)*self.n_g)**2-self.E_J*self.cos_t
        #         self.E, self.Eigvecs = (self.H_t).eigenstates()
    
        self.omega_q = self.E[1]-self.E[0]
        
        self.I_t=qt.qeye(self.n_trunc)

        self.H_t_eigbasis = qt.Qobj(np.diag((self.E)[0:self.n_trunc],0))
        self.n_t_eigbasis = self.transform(self.n_t)
        self.cos_t_eigbasis = self.transform(self.cos_t)
        self.sin_t_eigbasis = self.transform(self.sin_t)


    ##################################################################################      
    def Odd(self):
        #Getting eigenvectors whose wavefunction is odd under n -> - n
        Kinetic_odd = 4*self.E_C*(np.diag(np.arange(1,self.max_charge+1))-self.n_g*np.eye(self.max_charge))**2
        Potential_odd = -self.E_J/2*(np.diag(np.ones(self.max_charge-1),1)+ np.diag(np.ones(self.max_charge-1),-1))
    
        [E_odd, Eigvecs_half_odd] = sp.linalg.eigh(Kinetic_odd + Potential_odd)
    
        #Creating full eigenvector of size 2*max_charge + 1. 
        Eigvecs_odd = np.zeros([2*self.max_charge+1, self.max_charge])
    
        for j in range(self.max_charge):
            Eigvecs_odd[self.max_charge+1: 2*self.max_charge+1, j] = Eigvecs_half_odd[:,j]
            Eigvecs_odd[0:self.max_charge,j] = -np.flip(Eigvecs_odd[self.max_charge+1: 2*self.max_charge+1, j])
        
        return E_odd, Eigvecs_odd
    
    ##################################################################################   
    def Even(self):
        Kinetic_even = 4*self.E_C*(np.diag(np.arange(0,self.max_charge+1))-self.n_g*np.eye(self.max_charge+1))**2
        Potential_even = -self.E_J/2*(np.diag(np.ones(self.max_charge),1)+ np.diag(np.ones(self.max_charge),-1))
        
        #Without using symmetry, eigenvalue equation is -E \psi_0 -E_J/2(psi_1+psi_-1). Employing symmetry
        #means making following substitution to Hamiltonian
        Kinetic_even[0,1] += -self.E_J/2
   
        [E_even, Eigvecs_half_even] = sp.linalg.eig(Kinetic_even + Potential_even)

        #We are digaonlizing a non-Hermitian matrix, we might get non-real eigenvalues
        if np.max(np.imag(E_even)) != 0:
            warnings.warn("Imaginary part of the energy was found and takes a value " + str(np.max(np.imag(E_even))))
    
        #We must sort the eigenvalues and eigenvectors ourself, since scipy doesn't do it for us for a non-Hermitian Hamiltonian
        order_list = np.argsort(np.real(E_even))
        
        E_even = np.real(E_even)
        
        temp_Eigvecs = copy.deepcopy(Eigvecs_half_even)
        
        for j in range(self.max_charge+1):
            Eigvecs_half_even[:,j] = temp_Eigvecs[:,order_list[j]]
        
        #Creating full eigenvector of size 2*max_charge + 1. 
        Eigvecs_even = np.zeros([2*self.max_charge+1, self.max_charge+1])
    
        for j in range(self.max_charge+1):
            Eigvecs_even[self.max_charge: 2*self.max_charge+1, j] = Eigvecs_half_even[:,j]
            Eigvecs_even[0:self.max_charge,j] = np.flip(Eigvecs_half_even[1:, j])
        
        return E_even, Eigvecs_even
        
    ################################################################################## 
    def Sort(self):
        E_even, Eigvecs_even = self.Even()
        E_odd, Eigvecs_odd = self.Odd()
        
        E = np.sort(np.concatenate((E_even, E_odd)))
        #E -= E[0] #Making sure lowest energy state is zero
        
        Eigvecs = np.zeros([2*self.max_charge +1, 2*self.max_charge+1])
    
        for j in range(self.max_charge):
            Eigvecs[:,2*j] = Eigvecs_even[:,j]
            Eigvecs[:,2*j+1] = Eigvecs_odd[:,j]
    
        Eigvecs[:,-1] = Eigvecs_even[:,-1]
    
        #Normalizing
        for j in range(2*self.max_charge+1):
            Eigvecs[:,j] /= sp.linalg.norm(Eigvecs[:,j])

        Eigvecs_qobj=[]
        for j in range(len(E)):
            Eigvecs_qobj.append(qt.Qobj(Eigvecs[:,j]))
            
        return E, Eigvecs_qobj

    ################################################################################## 
    def Sort_higher_harmonics(self):

        Eigvecs_qobj=[]
        if self.n_g%1==0:
            for q in range(2*self.max_charge+1):
                if q%2==0:
                    if (np.argmax(self.Eigvecs[q]))>self.max_charge:
                        new_eigv_t_e=qt.Qobj(np.vstack([np.flip(self.Eigvecs[q][self.max_charge+1:]),self.Eigvecs[q][self.max_charge:]])).unit()
                    else:
                        new_eigv_t_e=qt.Qobj(np.vstack([self.Eigvecs[q][:self.max_charge+1],np.flip(self.Eigvecs[q][:self.max_charge])])).unit()
                    Eigvecs_qobj.append(qt.Qobj(new_eigv_t_e))
                
                elif q%2==1:
                    if (np.argmax(self.Eigvecs[q]))>self.max_charge:
                        new_eigv_t_o=qt.Qobj(np.vstack([-np.flip(self.Eigvecs[q][self.max_charge+1:]),self.Eigvecs[q][self.max_charge:]])).unit()
                    else:
                        new_eigv_t_o=qt.Qobj(np.vstack([self.Eigvecs[q][:self.max_charge+1],-np.flip(self.Eigvecs[q][:self.max_charge])])).unit()
                    Eigvecs_qobj.append(qt.Qobj(new_eigv_t_o))
    
            return Eigvecs_qobj

        else:
            for q in range(2*self.max_charge+1):
                if q%2==0:
                    if (np.argmax(self.Eigvecs[q]))>self.max_charge:
                        new_eigv_t_e=qt.Qobj(np.vstack([0,np.flip(self.Eigvecs[q][self.max_charge+1:]),self.Eigvecs[q][self.max_charge+1:]])).unit()
                    else:
                        new_eigv_t_e=qt.Qobj(np.vstack([self.Eigvecs[q][:self.max_charge+1],np.flip(self.Eigvecs[q][1:self.max_charge+1])])).unit()
                    Eigvecs_qobj.append(qt.Qobj(new_eigv_t_e))
                
                elif q%2==1:
                    if (np.argmax(self.Eigvecs[q]))>self.max_charge:
                        new_eigv_t_o=qt.Qobj(np.vstack([0,-np.flip(self.Eigvecs[q][self.max_charge+1:]),self.Eigvecs[q][self.max_charge+1:]])).unit()
                    else:
                        new_eigv_t_o=qt.Qobj(np.vstack([self.Eigvecs[q][:self.max_charge+1],-np.flip(self.Eigvecs[q][1:self.max_charge+1])])).unit()
                    Eigvecs_qobj.append(qt.Qobj(new_eigv_t_o))
    
            return Eigvecs_qobj


            return Eigvecs_qobj
    ################################################################################## 
    # function to write the operators from charge basis in transmon eigen basis
    def transform(self,ket): 
        return qt.Qobj((ket).transform(self.Eigvecs)[0:self.n_trunc,0:self.n_trunc])

    ##################################################################################
    def cos_m_t(self,m):
        cos_m_t=qt.Qobj(np.diag(np.ones(2*self.max_charge+1-m),m)+np.diag(np.ones(2*self.max_charge+1-m),-m))/2
        for m_index in range(m):
            cos_m_t+=0.5*qt.basis(2*self.max_charge+1,2*self.max_charge+1-m+m_index)*qt.basis(2*self.max_charge+1,m_index).dag()
            cos_m_t+=0.5*qt.basis(2*self.max_charge+1,m_index)*qt.basis(2*self.max_charge+1,2*self.max_charge+1-m+m_index).dag()
        return cos_m_t

    ##################################################################################
    def sin_m_t(self,m):
        sin_m_t=qt.Qobj(1j*(np.diag(np.ones(2*self.max_charge+1-m),m)-np.diag(np.ones(2*self.max_charge+1-m),-m)))/2
        for m_index in range(m):
            sin_m_t+=0.5*1j*qt.basis(2*self.max_charge+1,2*self.max_charge+1-m+m_index)*qt.basis(2*self.max_charge+1,m_index).dag()
            sin_m_t+=-0.5*1j*qt.basis(2*self.max_charge+1,m_index)*qt.basis(2*self.max_charge+1,2*self.max_charge+1-m+m_index).dag()
        return sin_m_t
    

class L_shunted_Transmon:

    def __init__(self, Transmon_params):
        self.T_params = copy.deepcopy(Transmon_params)
        
        self.E_C = self.T_params['E_C']
        self.E_J = self.T_params['E_J']
        self.E_L = self.T_params['E_L']
        self.n_g =0 
        if self.T_params.get('phi_ext') == None:
            self.phi_ext = 0
        else: 
            self.phi_ext = self.T_params['phi_ext']
        
        # Number of phase states used to diagonalize the transmon
        if self.T_params.get('phi_size') == None:
            self.phi_size = 2001
        elif type(self.T_params['phi_size']) != int :
            raise TypeError('Number of phase states used to diagonalize the transmon should be an integer')
        else: 
            self.phi_size = self.T_params['phi_size']

        # maximal phase
        if self.T_params.get('phi_max') == None:
            self.phi_max = 20
        else: 
            self.phi_max = self.T_params['phi_max']

        self.phi_vec=np.linspace(-self.phi_max,self.phi_max,self.phi_size)
        self.delta_phi=2*self.phi_max/(self.phi_size-1)

        self.I_op=qt.qeye(self.phi_size)
        
        self.phi_op=qt.qdiags(self.phi_vec,0)
        
        self.cosphi_op=qt.qdiags(np.cos(self.phi_vec),0)
        self.cos2phi_op=qt.qdiags(np.cos(2*self.phi_vec),0)
        self.cos3phi_op=qt.qdiags(np.cos(3*self.phi_vec),0)
        
        self.sinphi_op=qt.qdiags(np.sin(self.phi_vec),0)
        self.sin2phi_op=qt.qdiags(np.sin(2*self.phi_vec),0)
        self.sin3phi_op=qt.qdiags(np.sin(3*self.phi_vec),0)
   
        self.n2_op=-1*(qt.qdiags(np.ones(self.phi_size-1),1)+qt.qdiags(-2*np.ones(self.phi_size),0)+qt.qdiags(np.ones(self.phi_size-1),-1))/self.delta_phi**2 # second order derivative
        self.n_op=1j*(qt.qdiags(1/2*np.ones(self.phi_size-1),1)+qt.qdiags(-1/2*np.ones(self.phi_size-1),-1))/self.delta_phi # second order derivative
        
        self.H_t=4*self.E_C*self.n2_op-self.E_J*self.cosphi_op+self.E_L/2*(self.phi_op+self.I_op*self.phi_ext)**2
 
        self.E, self.Eigvecs = (self.H_t).eigenstates()
    
        self.omega_q = self.E[1]-self.E[0]

        if self.T_params.get('n_trunc') == None:
            self.n_trunc = 21
        else: 
            self.n_trunc = self.T_params['n_trunc']
        
        self.I_t=qt.qeye(self.n_trunc)

        self.H_t_eigbasis = qt.Qobj(np.diag((self.E)[0:self.n_trunc],0))
        self.n2_t_eigbasis = self.transform(self.n2_op)
        self.n_t_eigbasis = self.transform(self.n_op)
        self.cos_t_eigbasis = self.transform(self.cosphi_op)
        self.sin_t_eigbasis = self.transform(self.sinphi_op)


    def transform(self,ket): 
        return qt.Qobj((ket).transform(self.Eigvecs)[0:self.n_trunc,0:self.n_trunc])


################################################################################## 

class Cavity:
    
    def __init__(self, Cavity_params):
        self.C_params = copy.deepcopy(Cavity_params)
        self.omega_r = self.C_params['omega_r']

        #Number of charge states used to diagonalize the transmon
        if self.C_params.get('max_fock') == None:
            self.max_fock = 50
        elif type(self.C_params['max_fock']) != int :
            raise TypeError('Max number of fock states used to diagonalize the resonator should be an integer')
        else: 
            self.max_fock = self.C_params['max_fock']
            
        self.at=qt.create(self.max_fock)
        self.a=qt.destroy(self.max_fock)
        self.I_r=qt.qeye(self.max_fock)
        self.H_r=self.omega_r*self.at*self.a

################################################################################## 

class Transmon_Cavity:

    def __init__(self, System_params):
        self.S_params = copy.deepcopy(System_params)
        self.g = self.S_params['g']
        
        if self.S_params.get('type') != None:
            if self.S_params['type']=='L_shunt':
                self.Transmon=L_shunted_Transmon(self.S_params)
            else:
                print('Invalid type, using transmon type. For L-shunted transmon, please use L_shunt')
                self.Transmon=Transmon(self.S_params)
        else:
            self.Transmon=Transmon(self.S_params)
        
        self.Cavity=Cavity(self.S_params)

        if self.S_params.get('fast_diag') == True:
            self.H_int=self.g*qt.tensor(self.Transmon.n_t_eigbasis-self.Transmon.n_g*self.Transmon.I_t,(self.Cavity.a+self.Cavity.at))
        else:
            self.H_int=-1j*self.g*qt.tensor(self.Transmon.n_t_eigbasis-self.Transmon.n_g*self.Transmon.I_t,(self.Cavity.a-self.Cavity.at))

        
        self.H_coupl=qt.tensor(self.Transmon.H_t_eigbasis,self.Cavity.I_r)+qt.tensor(self.Transmon.I_t,self.Cavity.H_r)+self.H_int
        self.eigs_coupl, self.eigv_coupl=self.H_coupl.eigenstates()
        self.eigs_coupl=np.real(self.eigs_coupl)-np.min(np.real(self.eigs_coupl))

        self.H_bare=qt.tensor(self.Transmon.H_t_eigbasis,self.Cavity.I_r)+qt.tensor(self.Transmon.I_t,self.Cavity.H_r)
        self.eigs_bare, self.eigv_bare=self.H_bare.eigenstates()
        self.eigs_bare=np.real(self.eigs_bare)-np.min(np.real(self.eigs_bare))


    def Schrieffer_wolff(max_SW=11):

        n_trunc=self.Transmon.max_charge
        eigs_t=self.Transmon.eigs_t
        omega_r=self.Cavity.omega_r
        g=self.g
        n_t=self.Transmon.n_t
        H_int=self.H_int
        eigv_bare=self.eigv_bare
        
        S_t=np.zeros((n_trunc,n_trunc),dtype=np.complex_)
        for i in range(max_SW):
            for j in range(max_SW):
                #if (i_t, j_t) != (9, 10) and (i_t, j_t) != (10, 9):
                S_t[i,j]=g*n_t[i,j]/(eigs_t[j]-eigs_t[i]-omega_r)

        S_t=qt.Qobj(S_t)
        St_t=S_t.dag()
        
        S=-1j*(qt.tensor(S_t,at)+qt.tensor(St_t,a))
        A=np.zeros((n_trunc,n_trunc),dtype=np.complex_)
        op_A=g*(n_t*S_t+St_t*n_t)/2
        for i in range(max_SW):
            for j in range(max_SW):
                if i!=j:
                    A[i,j]=op_A[i,j]/(eigs_t[i]-eigs_t[j])
        A=qt.Qobj(A)
        
        B=np.zeros((n_trunc,n_trunc),dtype=np.complex_)
        op_B=g*(qt.commutator(n_t,S_t-St_t))/2
        for i in range(max_SW):
            for j in range(max_SW):
                if i!=j:
                    B[i,j]=op_B[i,j]/(eigs_t[i]-eigs_t[j])
        B=qt.Qobj(B)
        
        C=np.zeros((n_trunc,n_trunc),dtype=np.complex_)
        op_C=g*(qt.commutator(n_t,St_t))/2
        for i in range(max_SW):
            for j in range(max_SW):
                C[i,j]=op_C[i,j]/(eigs_t[i]-eigs_t[j]-2*omega_r)
        C=qt.Qobj(C)
        
        D=np.zeros((n_trunc,n_trunc),dtype=np.complex_)
        op_D=-g*(qt.commutator(n_t,S_t))/2
        for i in range(max_SW):
            for j in range(max_SW):
                D[i,j]=op_D[i,j]/(eigs_t[i]-eigs_t[j]+2*omega_r)
        D=qt.Qobj(D)
        
        
        self.T=qt.tensor(A,I_r)+qt.tensor(B,at*a)+qt.tensor(C,a*a)+qt.tensor(D,at*at)
        self.expT_m=(-T).expm()
    
        self.eigv_SW=self.expS_m*self.expT_m*eigv_bare
    

    