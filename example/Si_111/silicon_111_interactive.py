import numpy as np
import laueSpots as ls
import laueInteractive as li
try: # make plot with sliders interactive
    import IPython
    shell = IPython.get_ipython()
    shell.enable_matplotlib(gui='qt')
except:
    pass  

#%% define silicon lattice

a_Si = lambda T: (5.4304 + 1.8138e-5 * T + 1.542e-9 * T**2) * 1e-10 # https://aip.scitation.org/doi/pdf/10.1063/1.1663432
a = a_Si(300)

# lattice vectors
a1 = 0.5 * a * (np.eye(3)[1] + np.eye(3)[2])
a2 = 0.5 * a * (np.eye(3)[0] + np.eye(3)[2])
a3 = 0.5 * a * (np.eye(3)[0] + np.eye(3)[1])

# basis
r_base = a * np.array([[0, 0, 0], [-0.25, 0.25, -0.25]])  # vectors of the basis atoms
f_base = [1, 1]  # atomic form factors

A = np.array([a1, a2, a3])



#%% experimental parameters: geometry

l = 0.043  # [m] distance between crystal and detector at x = y = 0
b_vec = np.array([0.05, 0.05]) # lengths of the detector screen
pixels_vec = [500, 500] # amount of pixels


#%% parameters for the Laue image
n_laue = 4
lambda_min = 12.34 / 35 * 1e-10 # [m] Duane-Hunt-law for 35 kV
lambda_max = 5e-10  # [m] max wavelength


#%% run script

lat = ls.Lattice(A, r_base, f_base, 45/180*np.pi, -35.264389682754655/180*np.pi, 163/180*np.pi, 0.48/180*np.pi,2.23/180*np.pi)
det = ls.Detector(l, b_vec, pixels_vec, 1.06/180*np.pi, -1.5/180*np.pi, 0.0)
ld = ls.LaueDiffraction(lat, det)


fname = "Si_111.npy"

#%% interactive plot
svals = [[-10,10,0.48],[-10,10,2.23],[-180,180,163],[0.02,0.06,0.043],[-0.01,0.01,-0.00028],[-0.01,0.01,0.00041],[-10,10,1.06],[-10,10,-1.5]] # [min, max, initial]
sl_names = ["$\\delta$","$\\epsilon$","$\\gamma$","$l$","$\\Delta x$","$\\Delta y$","$\\alpha_\\mathrm{screen}$","$\\beta_\\mathrm{screen}$"]
linter = li.LaueDiffractionInteractive(lat,det,svals,sl_names,n_laue,lambda_min,lambda_max,fname)

#%% 3D measurement geometry
ld.plot3d(n_laue,lambda_min,lambda_max,pltSpots3d=True)

#%% calculated laue spots with measurement
detectorOffset = [-0.00028,0.00041]
ld.plot_img(n_laue,lambda_min,lambda_max,fname,detectorOffset)
    
    
    
    
    
    
    
    
    
