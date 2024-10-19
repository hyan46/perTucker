import matplotlib.pyplot as plt
import tensorly as tl
import numpy as np
import pandas as pd
import seaborn as sns
from perTucker import perTucker
from Simulation import SimulationExperiments

'''
Reproduce the Result
'''
def divide_along_axis(a,b,axis=None):
    if axis is None:
        return a/b
    c = a.copy()
    for i, x in enumerate(c.swapaxes(0,axis)):
        x /= b[i]
    return c

def main():
    np.random.seed(7777)
    
    # generate true label
    label_true=[0,1,2]*50
    label_true.sort()

    # record accuracy
    accuracy=[]

    # perform 100 replications
    for itr in range(100):
        # generate simulation data, X is the observed data, X_global and X_local are true global and local components
        simulation_data=SimulationExperiments(image_height=50, image_width=50, global_rank=5,
                                          ratio_lowerbound=0.7, ratio_upperbound=1.3,
                                          image_per_client=50 ,nclients_pershape=100,
                                          weight_value=5,global_scale=50)
        X, X_local, X_global = simulation_data.gen_images()

        # perform perTucker decomposition
        test=perTucker([5,5],[[5,5]]*len(X),dims_orthogonal=[0,1],max_itr=30)
        C_global, U_global, C_local, U_local = test.fit(X, rho=10)

        # generate test images and concatenate images
        X_test, _, _ = simulation_data.gen_images()
        X_test=np.transpose(X_test,(0,3,1,2)).reshape(3*simulation_data.image_per_client,50,50)

        ### Use perTucker to classify figures ###
        C_projection=np.zeros((X_test.shape[0],3,test.dims_local[0][0],test.dims_local[0][1]))
        for i in range(3):
            for j in range(X_test.shape[0]):
                C_projection[j,i]=tl.tenalg.multi_mode_dot(X_test[j], U_local[i],transpose=True)
        C_L2_norm=np.sum(C_projection**2, axis=(2,3))
        accuracy.append(np.mean(np.argmax(C_L2_norm,axis=1)==label_true))
    
        print(f'Finished replication {itr}')

    # Check the test statistics in the 100th replication
    standard = C_L2_norm.reshape(3,50,3)
    standard_median=np.median(standard,axis=(1,2))

    # Devide the test statistics of L2 norm by the median statistics within each category for better clarification
    temp=divide_along_axis(standard, standard_median,0)


    # Plot the boxplot
    sns.set(rc={"figure.dpi":800, 'savefig.dpi':800})
    data_box=pd.DataFrame({"Statistics": temp.reshape(150,3).T.reshape(-1,),
                       "Pattern": (["Swiss"]*50+["Oval"]*50+["Rectangle"]*50)*3,
                       "Label": ["Swiss"]*150+["Oval"]*150+["Rectangle"]*150})
    flierprops = dict(marker='D', markerfacecolor='black', markeredgecolor='black',markersize=4, linestyle='none')
    graph = sns.boxplot(x = data_box['Pattern'],
            y = data_box['Statistics'],
            hue = data_box['Label'], orient='v',flierprops=flierprops)
    plt.legend(title='Test statistics for')
    graph.set(xlabel='True patterns', ylabel='Relative norm of local core')
    plt.savefig('results/Class_box.png')
    print("Done! Figure saved to results/Class_box.png")

if __name__ == "__main__":
    main()
