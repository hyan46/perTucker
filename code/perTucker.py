import tensorly as tl
from scipy.linalg import orth
import numpy as np
from tensorly.decomposition import partial_tucker

'''
Personalized Tucker Algorithm
'''
class perTucker():
    def __init__(self, dim_global, dims_local, dims_orthogonal=None, tol=1e-9,max_itr=100,orthogonal=True):
        '''
        Parameters_newnewnew
        ----------
        dim_global : list, [a,b,c]
            dimension of the global core tensor. All clients have the same global core dimensions
        dims_local : list of lists, [[a1,b1,c1],[a2,b2,c2],...]
            dimension of the local core tensor. Each client can have its own dimensions
        dims_orthogonal : list, [a,b,c]
            dimensions that local factor matrices are orthogonal to global factor matrices, start with 0
            if None, all dimensions are orthogonal
        '''
        self.dim_global=dim_global
        self.dims_local=dims_local
        if dims_orthogonal is None:
            self.dims_orthogonal=np.arange(len(dim_global))
        else:
            self.dims_orthogonal=np.array(dims_orthogonal)
        self.tol=tol 
        self.max_itr=max_itr
        self.loss=[]
        self.loss_subspace=[]
        self.loss_subspace_global=[]
        self.loss_subspace_local_client=[]
        self.loss_subspace_local_dim=[]
        
        self.orthogonal=orthogonal
        
    def fit(self,Y,U_global_true=False,U_local_true=False,init_method='random', rho=0):
        '''
        Parameters
        ----------
        Y : tensor, or list of tensors
            If Y is a tensor, its shape is [C,I1,...,IK,n]. C is the number of clients.
                n is the number of samples in each client.
            If Y is a list of tensors, C clients have C tensors. Each tensor has dimension [I1,...,IK,n_i].
                The last dimension is the number of samples in each client, 1 should be used when there is only 1 sample.
        U_global_true : list of factor matrices,
            true global factor matrices to calculate the subspace error
        U_local_true : list of list of factor matrices,
            true local factor matrices to calculate the subspace error
        init_method : initialization method. Choose from 'svd' and 'random'. Default is 'random'
        rho: coefficients of the proximal term. Default is 0 meaning no proximal term
        
        Returns
        -------
        Core_global: list of ndarray, [C1,C2,...]
            each client have different core tensor
        Factors_global: list of ndarray, [U1,U2,U3]
            all the cilents share the same global factor matrices
        Cores_local: list of ndarray, [C1,C2,...]
            each client have different core tensor, dimensions may be different
        Factors_local: list of lists, [[U1,U2,U3],[U1,U2,U3],...]
            each client have his own local factor matrices

        Methods
        -------
        self.loss: reconstruction loss path
        self.loss_subspace: global factor subspace loss path
        self.loss_subspace_local_client: average local factors subspace loss path for different clients
        self.loss_subspace_local_dim: average local factors subspace loss path for different dimensions
        '''
        # ----------- check conditions ---------- #
        n_clients = len(Y)
        data_dim = Y[0].shape[:-1]
        for i in range(1,n_clients):
            if Y[i].shape[:-1]!= data_dim:
                raise ValueError("The data dimension is not uniform")
        
        # ---------- initialization ---------- #
        C_global,U_global=self.initialize_global(Y, method=init_method)
        C_local,U_local=self.initialize_local(Y, C_global, U_global, method=init_method)
        
        # ---------- BCD iterations ---------- #
        for itr in range(self.max_itr):
            for j in range(len(self.dim_global)):
            ## ----- Global factor ----- ##
                Y_res_global=[]
                for i in range(n_clients):
                    Y_res_global.append(Y[i]-tl.tenalg.multi_mode_dot(C_local[i], U_local[i]))
            
                matrix_temp=0
                for i in range(len(Y)):
                    W=tl.unfold(tl.tenalg.multi_mode_dot(Y_res_global[i], U_global,skip=j,transpose=True),mode=j)
                    matrix_temp+=W.dot(W.T)
                eigen=np.linalg.eig(matrix_temp + 2*rho*U_global[j].dot(U_global[j].T))
                U_global[j]=np.real(eigen[1][:,np.argsort(eigen[0])[::-1][:self.dim_global[j]]])
            
            #self.loss.append(self.recons_loss(Y,C_global,U_global,C_local,U_local))
            
            ## ----- Local factor ----- ##
                for i in range(n_clients):
                    Y_res_local=(Y[i]-tl.tenalg.multi_mode_dot(C_global[i], U_global))
                    if self.orthogonal==False:
                        C_local[i],U_local[i] = partial_tucker(Y_res_local, modes=[0,1,2] , rank = self.dims_local[i])
                    else:
                        W=tl.unfold(tl.tenalg.multi_mode_dot(Y_res_local, U_local[i],skip=j,transpose=True),mode=j)
                        if j in self.dims_orthogonal: 
                            U_local[i][j]=self.Local_update(U_local[i][j],W,U_global[j],rho)
                        else:
    #                            W=tl.unfold(tl.tenalg.multi_mode_dot(C_local[i], U_local[i],skip=j),mode=j).dot(tl.unfold(Y_res_local,mode=j).T)
                            U_local[i][j]=self.Local_update(U_local[i][j],W,np.zeros(U_global[j].shape),rho)
                            
            
            ## ----- Global and local core ----- ##
                for i in range(n_clients):
                    C_global[i]=tl.tenalg.multi_mode_dot(Y_res_global[i], U_global, transpose=True)
                    Y_res_local=(Y[i]-tl.tenalg.multi_mode_dot(C_global[i], U_global))
                    C_local[i]=tl.tenalg.multi_mode_dot(Y_res_local, U_local[i], transpose=True)
            #self.loss.append(self.recons_loss(Y,C_global,U_global,C_local,U_local))
            
            
            ## ----- Loss information ----- ##
            self.loss.append(self.recons_loss(Y,C_global,U_global,C_local,U_local))
            if U_global_true is not False:
                self.loss_subspace.append(self.subspace_loss(U_global_true,U_global))
            if U_local_true is not False:
                temp_local=[]
                for i in range(len(Y)):
                    temp_local.append(self.subspace_loss(U_local_true[i], U_local[i]))
                self.loss_subspace_local_client.append(temp_local)
                
                dim_loss=[]
                for j in range(len(self.dim_global)-1):
                    loss_temp=0
                    for i in range(len(Y)):
                        loss_temp+=self.subspace_loss([U_local_true[i][j]], [U_local[i][j]])
                    dim_loss.append(loss_temp)
                self.loss_subspace_local_dim.append(dim_loss)
     
        return C_global, U_global, C_local, U_local
        
        
    def initialize_global(self, data, method='random'):
        n_clients=len(data)
        C_global=[]
        if method=='svd':
            data_svd=data[0]
            for i in range(1,n_clients):
                data_svd=np.concatenate((data_svd,data[i]),axis=-1)
            C_global, U_global = partial_tucker(data_svd, range(len(data[0].shape)-1), self.dim_global)
        elif method=='random':
            U_global=[]
            for i in range(len(self.dim_global)):
                temp=tl.tensor(np.random.rand(data[0].shape[i], self.dim_global[i]))
                U_global.append(orth(temp))
            for i in range(n_clients):
                C_global.append(tl.tenalg.multi_mode_dot(data[i], U_global,transpose=True))
        return C_global, U_global
            
            
    def initialize_local(self, data, C_global, U_global, method='random'):
        n_clients=len(data)
        C_local=[]
        U_local=[]
        if method == 'svd':
            for i in range(n_clients):
                res_temp=data[i]-tl.tenalg.multi_mode_dot(C_global[i],U_global)
                core_temp, factors_temp = partial_tucker(res_temp, range(len(data[0].shape)-1), self.dims_local[i])
                C_local.append(core_temp)
                U_local.append(factors_temp)
        elif method=='random':
            for i in range(n_clients):
                res_temp=data[i]-tl.tenalg.multi_mode_dot(C_global[i],U_global)
                U_temp=[]
                for j in range(len(self.dim_global)):
                    temp=tl.tensor(np.random.rand(data[0].shape[j], self.dims_local[i][j]))
                    U_temp.append(orth(temp))
                U_local.append(U_temp)
                C_local.append(tl.tenalg.multi_mode_dot(res_temp, U_temp,transpose=True))
        return C_local, U_local
    
    def Local_update(self,V,W,U,rho):
        temp=U@U.T
        S_p=(np.eye(temp.shape[0])-temp)@W@W.T@(np.eye(temp.shape[0])-temp) + 2*rho*V@V.T
        eigen=np.linalg.eig(S_p)
        return np.real(eigen[1][:,np.argsort(eigen[0])[::-1][:V.shape[1]]])
        
    
    def recons_single(self,Y,C_global,U_global,C_local,U_local):
        Y_global=tl.tenalg.multi_mode_dot(C_global, U_global)
        Y_local=tl.tenalg.multi_mode_dot(C_local, U_local)
        return np.sum((Y-Y_global-Y_local)**2)
    
    def recons_loss(self,Y,C_global,U_global,C_local,U_local):
        loss=0
        for i in range(len(Y)):
            loss+=self.recons_single(Y[i],C_global[i],U_global,C_local[i],U_local[i])
        return loss
    
    def subspace_loss(self,U_global_true,U_global):
        loss_s=[]
        for i in range(len(U_global_true)):
            r=len(U_global_true[i][0])
            pu = orth(U_global_true[i])@orth(U_global_true[i]).T
            pv = U_global[i]@U_global[i].T
            loss_s.append(r-np.trace(pu@pv))
        return np.mean(loss_s)