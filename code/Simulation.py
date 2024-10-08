import tensorly as tl
from scipy.linalg import orth
import numpy as np

'''
Simulation Data Generation
'''
class SimulationExperiments():
    def __init__(self, image_height=20, image_width=20, n_channels=None, weight_value=1,is_random=True, global_rank = 2, ratio_lowerbound = 1, ratio_upperbound = 1, image_per_client = 1, nclients_pershape = 10, is_global_orthogonal_basis = True, global_scale = 10, noise = False):
        '''
        Parameters
        ----------
        image dimension: image_height * image_width
        weight value: height of the shape
        global_rank: rank of the background
        shape difference: ratio_lowerbound and ratio_upperbound
        image_per_client: how many images per client
        nclient_pershape: how many clients per shape
        global_scale: impact of the global background component
        '''
        n_channel=None
        self.image_height=image_height
        self.image_width=image_width  
        self.n_channels = n_channel
        self.weight_value=weight_value
        self.global_rank = global_rank
        self.ratio_lowerbound = ratio_lowerbound
        self.ratio_upperbound = ratio_upperbound
        U_true = np.random.randn(image_width,global_rank)
        V_true = np.random.randn(global_rank,image_height)

        if is_global_orthogonal_basis:
            U_true = orth(U_true)
            V_true = orth(V_true.T).T

        self.U_true = U_true
        self.V_true = V_true
        self.image_per_client = image_per_client
        self.nclients_pershape = nclients_pershape
        self.nclients = 3 #* self.nclients_pershape
        self.global_scale = global_scale
        self.noise = noise
        
    def gen_local(self, region="swiss", ratio=1, n_channels=None, weight_value=1):
        """Generates an image for regression testing
        # TODO Test n_channels
        Parameters
        ----------
        region : {'swiss', 'rectangle'}
        image_height : int, optional
        image_width : int, optional
        weight_value : float, optional
        n_channels : int or None, optional
        ratio      : float, optional (control the ratio of x and y axis of the shape)
            if not None, the resulting image will have a third dimension
        Returns
        -------
        ndarray
            array of shape ``(image_height, image_width)``
            or ``(image_height, image_width, n_channels)``
            array for which all values are zero except the region specified
        """
        image_height = self.image_height
        image_width = self.image_width
        weight = np.zeros((image_height, image_width), dtype=float)
        weight_value=self.weight_value

        if region == "swiss":
            slim_width = int((image_width // 2) - (image_width // (10*ratio) + 1))
            large_width = int((image_width // 2) - (image_width // (3./ratio) + 1))
            slim_height = int((image_height // 2) - (image_height // (10*ratio) + 1))
            large_height = int((image_height // 2) - (image_height // (3./ratio) + 1))
#            print(large_height)
            weight[large_height:-large_height, slim_width:-slim_width] = weight_value
            weight[slim_height:-slim_height, large_width:-large_width] = weight_value

        elif region == "rectangle":
            large_height = int((image_height // 2) - (image_height // (4*ratio)))
            large_width = int((image_width // 2) - (image_width // (4./ratio)))
            weight[large_height:-large_height, large_width:-large_width] = weight_value

        elif region == "circle":
            radius_x = int(image_width // (3*ratio))
            radius_y = int(image_width // (3./ratio))
            cy = image_width // 2
            cx = image_height // 2
            y, x = np.ogrid[-radius_y:radius_y, -radius_x:radius_x]
            index = x**2/radius_x**2 + y**2/radius_y**2 <= 1
            weight[cy - radius_y : cy + radius_y, cx - radius_x : cx + radius_x][index] = 1

        if n_channels is not None and weight.ndim == 2:
            weight = np.concatenate([weight[..., None]] * n_channels, axis=-1)

        return tl.tensor(weight)


    def gen_global(self, C):
        return tl.tensor(self.U_true@C@self.V_true)


    def gen_images(self):
        img_ht = self.image_height
        img_width = self.image_width
        rank = self.global_rank
        
        patterns = ['swiss', 'circle','rectangle']
        X = np.zeros((self.nclients,img_ht,img_width,self.image_per_client))
        X_local = np.zeros((self.nclients,img_ht,img_width,self.image_per_client))
        X_global = np.zeros((self.nclients,img_ht,img_width,self.image_per_client))
#        iterations = np.arange(self.nclients_pershape)
        
        for i, pattern in enumerate(patterns):
            for j in range(self.image_per_client):
                # Generate the original image
                X_local[i,:,:,j] = self.gen_local(region=pattern, ratio = np.random.uniform(self.ratio_lowerbound,self.ratio_upperbound))
                C = np.random.randn(rank,rank)
                X_global[i,:,:,j] = self.gen_global(C) * self.global_scale
                X_local[i,:,:,j] = tl.tensor(X_local[i,:,:,j])
                X[i,:,:,j] = X_local[i,:,:,j] + X_global[i,:,:,j]
                if self.noise:
                    X[i,:,:,j]=X[i,:,:,j]+2*np.random.normal(0,1,X[i,:,:,j].shape)
        
        return X, X_local, X_global