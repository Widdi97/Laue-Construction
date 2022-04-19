import numpy as np
import laueSpots as ls
try: # make plot with sliders interactive
    import IPython
    shell = IPython.get_ipython()
    shell.enable_matplotlib(gui='qt')
except:
    pass  

a = 5e-10 # lattice constant

#%% fcc

a1 = 0.5 * a * (np.eye(3)[1] + np.eye(3)[2])
a2 = 0.5 * a * (np.eye(3)[0] + np.eye(3)[2])
a3 = 0.5 * a * (np.eye(3)[0] + np.eye(3)[1])

# silicon basis
r_base = a * np.array([[0, 0, 0], [0.25, 0.25, 0.25]])  # vectors of the basis atoms
f_base = [1, 1]  # atomic form factors

#%% sc

# a1 = a * np.array([1,0,0])
# a2 = a * np.array([0,1,0])
# a3 = a * np.array([0,0,1])

# r_base = a * np.array([[0, 0, 0]])  # vectors of the basis atoms
# f_base = [1]  # atomic form factors

# %% bcc with monoatomic basis

# a1 = 0.5 * a * ( - np.eye(3)[0] + np.eye(3)[1] + np.eye(3)[2])
# a2 = 0.5 * a * ( + np.eye(3)[0] - np.eye(3)[1] + np.eye(3)[2])
# a3 = 0.5 * a * ( + np.eye(3)[0] + np.eye(3)[1] - np.eye(3)[2])

# r_base = a * np.array([[0, 0, 0]]
#                   )  # vectors of the basis atoms
# f_base = [1]  # atomic form factors


# %% experimental parameters: geometry

l = 0.05  # [m] distance between crystal and detector / film at x = y = 0
b_vec = np.array([0.1, 0.1])
pixels_vec = [500, 500]

n_laue = 4
lambda_min = 12.34 / 35 * 1e-10 # [m] Duane-Hunt-law for 35 kV
lambda_max = 5e-10  # [m] max wavelength


#%% run script
A = np.array([a1, a2, a3])

# lat = ls.Lattice(A, r_base, f_base, 45/180*np.pi, -35.264389682754655/180*np.pi, 0) # lattice in (1 1 1) direction
lat = ls.Lattice(A, r_base, f_base, 0,0,0) # lattice in (1 0 0) direction
det = ls.Detector(l, b_vec, pixels_vec, 1.06/180*np.pi, -1.5/180*np.pi, 0.0)
ld = ls.LaueDiffraction(lat, det)

ld.plot3d(n_laue,lambda_min,lambda_max,pltSpots3d=True)
ld.plot_img(n_laue,lambda_min,lambda_max)
    
    
    
    
