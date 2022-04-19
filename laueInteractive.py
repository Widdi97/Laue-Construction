import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from matplotlib import gridspec
from laueSpots import Lattice, Detector, LaueDiffraction

"""
This script is a mess.
It is specialized for one use-case.
Use at your own risk!
"""

class LaueDiffractionInteractive:
    def __init__(self,lattice,detector,svals,sl_names,n_laue,lambda_min,lambda_max,fname=""):
        #starting values for the sliders
        self.svals = svals
        self.sl_names = sl_names
        
        #lattice parameters
        self.A = lattice.A
        self.r_base = lattice.r_base
        self.f_base = lattice.f_base
        
        # detector parameters
        self.b_vec = detector.b_vec
        self.pixels_vec = detector.pixels_vec
        
        self.n_laue = n_laue
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.data = np.load(fname)
        self.run()
    
    def run(self):
        self.fig = plt.figure(figsize=(12,15))
        spec = gridspec.GridSpec(ncols=1, nrows=2,height_ratios=[4, 1])
        gs_sliders = gridspec.GridSpecFromSubplotSpec(len(self.sl_names), 1, subplot_spec=spec[1],hspace=0.3) #,height_ratios=[1,2]
        self.ax_spots = self.fig.add_subplot(spec[0])
        
        #position sliders
        sls = [self.fig.add_subplot(gs_sliders[k], facecolor='lightgoldenrodyellow') for k in range(len(self.sl_names))]
        self.sliders = [Slider(sls[k], self.sl_names[k], *self.svals[k]) for k in range(len(self.sl_names))]
        self.ax_spots.set_xlim(0, 500)
        self.ax_spots.set_ylim(0, 500)
        
        for sl in self.sliders:
            sl.on_changed(self.sliderChange)
        
        #initial plot
        self.sliderChange(None)
        self.fig.tight_layout()
    
    def sliderChange(self,val):
        latice = Lattice(self.A, self.r_base, self.f_base, 45/180*np.pi, -35.264389682754655/180*np.pi, self.sliders[2].val/180*np.pi,
                         self.sliders[0].val/180*np.pi,self.sliders[1].val/180*np.pi)
        detector = Detector(self.sliders[3].val, self.b_vec, self.pixels_vec,  self.sliders[6].val/180*np.pi, self.sliders[7].val/180*np.pi , 0.0)
        ld = LaueDiffraction(latice, detector)
        spots = ld.calc_laue_spots(self.n_laue,self.lambda_min,self.lambda_max)
        self.ax_spots.cla()
        self.ax_spots.pcolormesh(self.data, cmap="OrRd", shading='flat',vmin=-0, vmax=400)
        alphas = []
        for hkl_vec, v_screen, v_screen_px, sf_sq  in spots:
            alphas.append( (sf_sq*( 1 / np.linalg.norm(hkl_vec))**2)**0.3 )
        alphas = np.array(alphas)
        alphas = alphas / alphas.max()
        for idx, spot in enumerate(spots):
            hkl_vec, v_screen, v_screen_px, sf_sq = spot
            self.ax_spots.scatter([v_screen_px[0]+self.sliders[4].val/detector.b_vec[0]*detector.pixels_vec[0]],
                                  [v_screen_px[1]+self.sliders[5].val/detector.b_vec[1]*detector.pixels_vec[1]],
                                  marker="o", facecolors='none', edgecolors='k',s=250,
                                     alpha=alphas[idx] )
            self.ax_spots.text(v_screen_px[0]+self.sliders[4].val/detector.b_vec[0]*detector.pixels_vec[0],
                                v_screen_px[1]+self.sliders[5].val/detector.b_vec[1]*detector.pixels_vec[1]+8,
                                str(hkl_vec), fontsize=16,alpha=alphas[idx])
        self.ax_spots.set_xlim(0, 500)
        self.ax_spots.set_ylim(0, 500)
    
    
