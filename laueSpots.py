import numpy as np
import matplotlib.pyplot as plt

def rot_mat_3d(alpha, beta, gamma):
    mx = [[1,              0,            0],  # R^3 rotational matrices
          [0,              np.cos(alpha),  -np.sin(alpha)],
          [0,              np.sin(alpha),  np.cos(alpha)]]
    my = [[np.cos(beta),  0,            np.sin(beta)],
          [0,              1,            0],
          [-np.sin(beta), 0,            np.cos(beta)]]
    mz = [[np.cos(gamma),    -np.sin(gamma), 0],
          [np.sin(gamma),    np.cos(gamma),  0],
          [0,              0,            1]]
    m = np.dot(mz, np.dot(my, mx))
    return m

class Lattice:
    def __init__(self, A, r_base=[0, 0, 0], f_base=[1], alpha=0, beta=0, gamma=0, delta=0, epsilon=0,rho=0):
        self.A = A
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.r_base = r_base
        self.f_base = f_base
        
        # rotate lattice vectors
        m_rot_xyz = rot_mat_3d(alpha, beta, gamma) # used for rough placement
        m_rot_od = rot_mat_3d(delta, epsilon, rho) # overdefines the rotations but helps placing the crystal accurately (sliders in the interactive plot).
        self.m_rot = np.dot(m_rot_od,m_rot_xyz)
        self.A_rot = np.dot(self.m_rot, A)
        
        #rotate basis vectors
        self.r_base_rot = [np.dot(self.m_rot, r_j) for r_j in self.r_base]

        # calculate reciprocal lattice vectors
        self.B = 2 * np.pi * np.linalg.inv(A).T  # reciprocal lattice vectors
        self.B_rot = 2 * np.pi * np.linalg.inv(self.A_rot).T

class Detector:
    def __init__(self, l, b_vec, pixels_vec, phi=0, theta=0, chi=0):
        # distance from the sample
        self.l = l
        self.screen_origin = np.array([0, 0, l])  # lab system

        # widths of the detector in x and y dir
        self.b_vec = b_vec
        self.pixels_vec = pixels_vec

        # V[0] and V[1]: vectors of pixel x and y directions
        # V[2]: detector normal vector
        self.V = np.eye(3)
        self.m_rot = rot_mat_3d(phi, theta, chi)
        self.V_rot = np.dot(self.m_rot, self.V)

        # pixel step vectors
        self.dpx_x = self.V_rot[0] * self.b_vec[0] / self.pixels_vec[0]
        self.dpx_y = self.V_rot[1] * self.b_vec[1] / self.pixels_vec[1]

        # corner of pixel (0,0)
        self.corner_origin = 0.5 * \
            ((1 - self.pixels_vec[0]) * self.dpx_x +
             (1 - self.pixels_vec[1]) * self.dpx_y)

        # pixels in the LAB system
        ctr = self.screen_origin + self.corner_origin
        self.pixels = [[ctr + px_x * self.dpx_x + px_y * self.dpx_y for px_y in range(
            self.pixels_vec[1])] for px_x in range(self.pixels_vec[0])]


class LaueDiffraction:
    def __init__(self, lattice, detector):
        self.lattice = lattice
        self.detector = detector
        
    def calc_laue_spots(self,n_laue,lambda_min,lambda_max):
        """
        Main method of the script. Use this method, when only the Laue spots should be calculated.
        """
        k_max = 2 * np.pi / lambda_min
        k_min = 2 * np.pi / lambda_max
        res = []
        for h in range(-n_laue, n_laue+1):
                for k in range(-n_laue, n_laue+1):
                    for l in range(-n_laue, n_laue+1):
                        try:
                            k_plt = self.calculate_laue_spot_k_vec([h, k, l])
                            abs_k = (k_plt[0]**2 + k_plt[1]**2 + k_plt[2]**2)**0.5
                            abs_hkl = (h**2 + k**2 + l**2)**0.5
                            if k_plt[2] > 0 and abs_k < k_max and abs_k > k_min and abs_hkl < n_laue:
                                sf = self.structure_factor(k_plt)
                                v_screen = self.trace_onto_screen(k_plt)
                                v_screen_px = self.pixelCoordinates(v_screen)
                                vs_bc = v_screen_px + 0.5  # used for boolean check
                                if 0 < vs_bc[0] and vs_bc[0] < self.detector.pixels_vec[0] and 0 < vs_bc[1] and vs_bc[1] < self.detector.pixels_vec[1]:
                                    res.append([[h,k,l],v_screen,v_screen_px,np.absolute(sf)**2])
                        except:
                            print("Error with parameters h,k,l =", h, k, l)
        return res
            
    def plot3d(self,n_laue,lambda_min,lambda_max,pltSpots3d=False):
        """
        usefull to check the crystal and detector lineup
        """
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()
        ax.view_init(elev=30, azim=0)
        
        # # plot all pixels. Only uncomment with low amount of pixels. Otherwise the script crashes.
        # for px_x in range(self.detector.pixels_vec[0]):
        #     for px_y in range(self.detector.pixels_vec[1]):
        #         vec = self.detector.pixels[px_x][px_y]
        #         if px_x == 0 and px_y == 0:
        #             ax.scatter([vec[0]], [vec[1]], [vec[2]],
        #                         depthshade=True, marker="o", color="k")
        #         else:
        #             ax.scatter([vec[0]], [vec[1]], [vec[2]],
        #                         depthshade=True, marker="x", color="grey")

        # detector borders
        p1 = self.detector.pixels[0][0]
        p2 = self.detector.pixels[0][-1]
        p3 = self.detector.pixels[-1][-1]
        p4 = self.detector.pixels[-1][0]
        p5 = self.detector.pixels[0][0]
        ax.plot(*[[p1[j],p2[j],p3[j],p4[j],p5[j]] for j in range(3)],color="k")

        # setting nice axis limits
        lim = self.detector.l
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_zlim(-lim, lim)

        # plot crystal with correct orientation
        self.generate_plt_objects()
        for k in range(len(self.indexPairs)):
            i1, i2 = self.indexPairs[k][0], self.indexPairs[k][1]
            c1 = np.dot(self.lattice.m_rot, self.corners[i1])
            c2 = np.dot(self.lattice.m_rot, self.corners[i2])
            ax.plot([c1[0], c2[0]], [c1[1], c2[1]],
                    [c1[2], c2[2]], color="k")
        
        #plot crystal unit cell axis cross
        ax_cross_colors = ["r","g","b"]
        for j in range(3):
            vec = self.lattice.m_rot.T[j] * lim / 6
            ax.plot([0, vec[0]], [0, vec[1]], [0, vec[2]],color=ax_cross_colors[j])
        
        # plot incoming beam
        ax.plot([0, 0], [0, 0], [-lim, 0], color="fuchsia")
        ax.plot([-lim/20, 0], [0, 0], [-lim/15, 0], color="fuchsia")
        ax.plot([lim/20, 0], [0, 0], [-lim/15, 0], color="fuchsia")

        # # plot detector normal vector
        # n_det = self.detector.V_rot[2] * lim * 0.1
        # ax.plot([0, n_det[0]], [0, n_det[1]], [self.detector.l,
        #         self.detector.l + n_det[2]], color="lime")
        
        # # plot a laue reflex k'-vector (debugging of ray-tracing)
        # k_plt = self.calculate_laue_spot_k_vec([1, 2, 3])
        # k_plt = k_plt / np.linalg.norm(k_plt) * lim
        # ax.plot([0, k_plt[0]], [0, k_plt[1]], [0, k_plt[2]], color="r")
        
        # # plot screen projection of k'
        # v_screen = self.trace_onto_screen(k_plt)
        # ax.scatter([v_screen[0]], [v_screen[1]], [v_screen[2]],
        #            depthshade=True, marker="o", color="r")
        
        # # represent with pixel vectors
        # v_screen_px = self.pixelCoordinates(v_screen)
        
        if pltSpots3d:
            spots = self.calc_laue_spots(n_laue,lambda_min,lambda_max)
            for hkl_vec, v_screen, v_screen_px, sf_sq in spots:
                ax.scatter([v_screen[0]], [v_screen[1]], [v_screen[2]], marker="o",color="k")
        fig.tight_layout()
        plt.show()
        
    def plot_img(self,n_laue,lambda_min,lambda_max,fname="",detectorOffset=[0,0]):
        """
        plot spots with image in the background
        """
        if fname != "":
            data = np.load(fname)
        fig = plt.figure(figsize=(8, 8))
        ax2d = fig.add_subplot(111)
        if fname != "":
            ax2d.pcolormesh(data, cmap="OrRd", shading='gouraud',
                  vmin=-0, vmax=400)
        spots = self.calc_laue_spots(n_laue,lambda_min,lambda_max)
        
        # calculate spot intensities
        alphas = []
        for hkl_vec, v_screen, v_screen_px, sf_sq  in spots:
            alphas.append( (sf_sq*( 1 / np.linalg.norm(hkl_vec))**2)**0.3 )
        alphas = np.array(alphas)
        alphas = alphas / alphas.max()
        
        # plot spots
        for idx, spot in enumerate(spots):
            hkl_vec, v_screen, v_screen_px, sf_sq = spot
            if hkl_vec != [2,2,2]: # supress 0th order
                ax2d.scatter([v_screen_px[0]+detectorOffset[0]/self.detector.b_vec[0]*self.detector.pixels_vec[0]],
                             [v_screen_px[1]+detectorOffset[1]/self.detector.b_vec[1]*self.detector.pixels_vec[1]],
                             marker="o", facecolors='none', edgecolors='k',s=250,alpha=alphas[idx])
                ax2d.text(min(v_screen_px[0]+detectorOffset[0]/self.detector.b_vec[0]*self.detector.pixels_vec[0],450),
                          min(v_screen_px[1]+detectorOffset[1]/self.detector.b_vec[1]*self.detector.pixels_vec[1]+12,475),
                          str(hkl_vec), fontsize=11,alpha=alphas[idx])
        ax2d.set_xlim(0, 500)
        ax2d.set_ylim(0, 500)
        ax2d.axes.xaxis.set_ticks([])
        ax2d.axes.yaxis.set_ticks([])
        fig.tight_layout()
        plt.show()

    def calculate_laue_spot_k_vec(self, hkl_vec):
        K_hkl = np.zeros(3)
        for j in range(3):
            K_hkl += self.lattice.B_rot.T[j] * hkl_vec[j]
        # correct k' which corresponds to the indices h,k,l
        k_out_vec = np.array(
            [K_hkl[0], K_hkl[1], (K_hkl[2]**2 - K_hkl[0]**2 - K_hkl[1]**2) / (2 * K_hkl[2])])
        return k_out_vec

    def trace_onto_screen(self, vec):
        """
        ray-trace reflex onto the detector
        """
        n_det = self.detector.V_rot[2]
        divisor = n_det[0]*vec[0] + n_det[1]*vec[1] + n_det[2]*vec[2]
        if divisor == 0.0:
            raise Exception("div 0!")
        lamb = self.detector.l * n_det[2] / divisor
        intersect = lamb * vec
        return intersect

    def pixelCoordinates(self, vec_on_detector):
        # shift vector so that screen is in the origin
        v_orig = vec_on_detector - np.array([0.0, 0.0, self.detector.l])

        # represent in the basis of the screen
        v_detector_coord = np.dot(self.detector.V_rot, v_orig)

        # represent in the pixel basis
        v_pixel_coord = np.array([v_detector_coord[0] / (self.detector.b_vec[0] / self.detector.pixels_vec[0]),
                                  v_detector_coord[1] / (self.detector.b_vec[1] / self.detector.pixels_vec[1])])
        
        # shift, so that origin of the pixels is in the top lhs
        v_pixel_coord_screen_orig = v_pixel_coord + np.array([(self.detector.pixels_vec[0] - 1) / 2,
                                                              (self.detector.pixels_vec[1] - 1) / 2])
        return v_pixel_coord_screen_orig

    def structure_factor(self, k_vec):
        sf = 0j
        for j in range(len(self.lattice.f_base)):
            f_j = self.lattice.f_base[j]
            r_j = self.lattice.r_base_rot[j]
            dot = r_j[0]*k_vec[0] + r_j[1]*k_vec[1] + r_j[2]*k_vec[2]
            # print(dot)
            sf += f_j * np.exp( - 1j * dot)
        return sf

    def generate_plt_objects(self):
        #sample (cuboid)
        cube_l = 0.003
        self.corners = []
        for f1 in [-0.5, 0.5]:
            for f2 in [-0.5, 0.5]:
                for f3 in [-0.5, 0.5]:
                    self.corners.append([f1*cube_l, f2*cube_l, f3*cube_l])
        thresh = 0.0000001
        self.indexPairs = []  # index pairs which are plotted
        for c1 in self.corners:
            for c2 in self.corners:
                diff = ((c2[0]-c1[0])**2 + (c2[1]-c1[1])
                        ** 2 + (c2[2]-c1[2])**2)**0.5
                # print(diff,abs(diff-l),,)
                if abs(diff-cube_l) < thresh:
                    self.indexPairs.append(
                        [self.corners.index(c1), self.corners.index(c2)])
        self.corners = np.array(self.corners)
    
