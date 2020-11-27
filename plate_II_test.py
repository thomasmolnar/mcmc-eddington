import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plate_deflections_eclipse = pd.read_csv('./data/plate_deflections_eclipse.csv')
plate_deflections_comparisons = pd.read_csv('./data/plate_deflections_comparison.csv')
star_positions = pd.read_csv('./data/star_positions.csv')

# Return in single data format [[plate 1 data], [plate 2 data], ...]
def get_plate_data(df, n_plate):
    df = df.loc[df['Plate']==n_plate].copy()
    Dx_corr = df['Dx_corr'].values[0]
    Dy_corr = df['Dy_corr'].values[0]
    
    plate_data = []
    for i in np.arange(1,8):
        star_Dx = df['Dx_star{}'.format(i)].values[0] - Dx_corr
        star_Dy = df['Dy_star{}'.format(i)].values[0] - Dy_corr
        plate_data.append([star_Dx, star_Dy])
    
    return plate_data

def get_all_plate_data(df):
    all_plate_data = []
    for i in np.arange(1,8):
        plate_data = get_plate_data(df, i)
        all_plate_data.append(plate_data)
    
    return np.array(all_plate_data)

ecl_deflections = get_all_plate_data(plate_deflections_eclipse)
comp_deflections = get_all_plate_data(plate_deflections_comparisons)

use_cols = ['RA_plate', 'Decl_plate', 'Ex', 'Ey']
star_pos = np.array([star_positions[use_cols][ind-1:ind].values[0] for ind in np.arange(1,8)])

import algorithms.metropolis_hastings as mh
import algorithms.multi_chain as mc

plate2_data_eclipse = ecl_deflections[2]
plate2_data_comparison = comp_deflections[2]

test = mh.metropolis_hastings(init_theta, plate2_data_eclipse, star_pos,
                             width = 0.1, n_steps = 100000,
                             method_kwargs = method_kwargs)

#3D Scatter diagram showing equilibrium distribution 
    fig1 = plt.figure()    
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.scatter(a, b, alpha)
    ax1.set_title('('+r"$b$"+')',style='italic')
    ax1.set_xlabel(r"$a$",style='italic')
    ax1.set_ylabel(r"$b$",style='italic')
    ax1.set_zlabel(r"$\alpha$" +'(arcsec)',style='italic')
    fig1.tight_layout()
    fig1.savefig('(b).png')
    
    
    fig2 = plt.figure()
    ax2 = fig2.add_subplot(111, projection='3d')
    ax2.scatter(og_a, og_b, og_alpha)
    ax2.set_title('('+r"$a$"+')',style='italic')
    ax2.set_xlabel(r"$a$",style='italic')
    ax2.set_ylabel(r"$b$",style='italic')
    ax2.set_zlabel(r"$\alpha$" +' (arcsec)',style='italic')
    fig2.tight_layout()
    fig2.savefig('(a).png')
    
    """#Histogram for frequencies of Alpha (Marginalisation)
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    ax.hist(alpha, bins = 50)
    ax.set_title('Histogram of '+r"$\alpha$"+' values for plate ' + str(plate_number))
    ax.set_xlabel(r"$\alpha$"+' value (arcseconds)')
    ax.set_ylabel('Frequencies')
    print('Mean Alpha for Plate '+str(plate_number)+' = '+str(chain[4])+' Sigma = '+str(np.sqrt((chain[5]))))
    
    #Diagram showing change in Alpha over iterations
    count = []
    for i in range(len(alpha)):
        count.append(i)
    fig3 = plt.figure()
    ax = fig3.add_subplot(111)
    ax.scatter(count, alpha)
    ax.set_title('Trace of '+r"$\alpha$"+' values for plate ' + str(plate_number))
    ax.set_xlim([0,max(count)])
    ax.set_xlabel('Iteration')
    ax.set_ylabel(r"$\alpha$"+' value (arcseconds)')
    
    
    #Diagram showing change in a over iterations
    count = []
    for i in range(len(a)):
        count.append(i)
    fig4 = plt.figure()
    ax = fig4.add_subplot(111)
    ax.scatter(count, a)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('a value')"""
    
    plt.show() 

def Corner(Chains):
    array1 = np.zeros((len(Chains[1]),7))
    array1[:,0] = Chains[1]
    array1[:,1] = Chains[2]
    array1[:,2] = Chains[3]
    array1[:,3] = Chains[11]
    array1[:,4] = Chains[12]
    array1[:,5] = Chains[13]
    array1[:,6] = Chains[14]
    fig1 = corner.corner(array1,labels=[r"$\alpha$"+' (acrsec)', r"$a$", r"$b$", r"$c$"+' (acrsec)', r"$d$", r"$e$", r"$f$"+' (arcsec)'],show_titles=True, title_kwargs={"fontsize": 12})
    fig1.tight_layout()
    fig1.savefig('corner.png')
    plt.show()