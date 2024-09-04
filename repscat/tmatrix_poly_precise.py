"""
@author: Maxim Vavilin maxim@vavilin.de
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import repscat as rs
import cmasher as cmr
from matplotlib.ticker import ScalarFormatter


class TmatrixPolyPrecise:
    """Polychromatic T-matrix in form (k k lam lam j j m m)
    units 1/m^2
    """

    def __init__(self, k1_list, k2_lists, vals, radius=None):
        """Domain should be k1_list, k2_list"""
        # self._check_domain_vs_vals(k1_list.shape, k2_list.shape, vals.shape)
        self._k1_list = k1_list
        self._k2_lists = k2_lists

        self._vals = vals
        self._radius = radius
        self._jay_max = vals.shape[4]

    # def _check_domain_vs_vals(self, k1shape, k2shape, vshape):
    #     if not (k1shape[0] == vshape[0] and k2shape[0] == vshape[1]):
    #         raise ValueError(
    #             f"Domain shape does not correspond to values"
    #         )

    @property
    def k1_list(self):
        return self._k1_list

    @property
    def k2_lists(self):
        return self._k2_lists

    @property
    def vals(self):
        return self._vals

    @property
    def radius(self):
        return self._radius

    @property
    def jay_max(self):
        return self._jay_max

    def scatter(self, incident):
        if self._vals.shape[1::2] != incident.vals.shape:
            raise ValueError("Tmatrix shape does not correspond to incident k_list")

        measure_arr = np.zeros(self._k2_lists.shape)
        incident_interpol_arr = np.zeros(
            (self._k2_lists.shape[0], *incident.vals.shape), dtype=complex
        )
        for i_k1, k1 in np.ndenumerate(self._k1_list):
            k2_small_list = self._k2_lists[i_k1]
            incident_interpol_arr[i_k1] = incident.interpolate(k2_small_list).vals
            measure_arr[i_k1] = (
                np.diff(k2_small_list, append=np.squeeze(k2_small_list[-1]))
                * k2_small_list
            )

        scattered_vals = np.einsum(
            "pq,pqabijmn,pqbjn->paim",
            measure_arr,
            self._vals,
            incident_interpol_arr,
            optimize=True,
        )
        return rs.WaveFunctionAngularMomentum(self._k1_list, scattered_vals)

    def get_second_theta_list(self, first_theta_list):
        num_theta = len(first_theta_list)
        theta_min = first_theta_list[-1]
        return np.linspace(theta_min, np.pi, num_theta)
    

    def get_energy_momentum_out(self, incident_pw):
        
        incident_am = incident_pw.in_angular_momentum_basis(self.jay_max)

        # self._vals = self._vals*0
        scattered_am = self.scatter(incident_am)

        # First part of the outgoing field interferes with the incident field
        scattered_pw_first_part = scattered_am.in_plane_wave_basis(
            incident_pw.theta_list, incident_pw.phi_list
        )

        # Second part of the outgoing field does not interfere with the incident field
        second_theta_list = self.get_second_theta_list(incident_pw.theta_list)
        scattered_pw_second_part = scattered_am.in_plane_wave_basis(
            second_theta_list, incident_pw.phi_list
        )
        
        # scattered_pw_first_part
        
        # scattered_pw_second_part.plot_angular_photon_density()
        # plt.show()
        
        # Interpolate incident field to larger k1_list
        incident_interp_pw = incident_pw.interpolate(scattered_am.k_list, incident_pw.theta_list)   #(+)

        energy_out = (incident_interp_pw + scattered_pw_first_part).energy() + scattered_pw_second_part.energy()
        momentum_out = (incident_interp_pw + scattered_pw_first_part).momentum_z() + scattered_pw_second_part.momentum_z()
        
        print('Mom_in, Mom_out: ', incident_pw.momentum_z(), momentum_out)
        
        momentum_in=incident_pw.momentum_z()
        print('Momentum percentage: ', 100*(momentum_in-momentum_out)/momentum_in)
        
        # (incident_interp_pw + scattered_pw_first_part).plot_angular_photon_density()
        # scattered_pw_second_part.plot_angular_photon_density()
        # plt.show()
        
        
        return energy_out, momentum_out
    
    def compute_transfer(self, incident_pw):
        
        energy_in = incident_pw.energy()
        momentum_in = incident_pw.momentum_z()
        
        energy_out, momentum_out = self.get_energy_momentum_out(incident_pw) 
        
        return (
            energy_in - energy_out,
            momentum_in - momentum_out
            )
        

    def plot(self, xlim, ylim, *indices):
        
        i_lam1, i_lam2, i_jay1, i_jay2, i_m1, i_m2 = rs.get_indices(
            self._jay_max, *indices
        )

        k1_mesh = np.zeros_like(self._k2_lists)
        k2_mesh = np.zeros_like(self._k2_lists)

        for i_k1, k1 in np.ndenumerate(self._k1_list):
            for i_k2, k2 in np.ndenumerate(np.squeeze(self._k2_lists[i_k1])):
                k1_mesh[i_k1, i_k2] = k1
                k2_mesh[i_k1, i_k2] = k2

        plot_vals_real = np.squeeze(
            np.real(self._vals[:, :, i_lam1, i_lam2, i_jay1, i_jay2, i_m1, i_m2])
        )
        plot_vals_imag = np.squeeze(
            np.imag(self._vals[:, :, i_lam1, i_lam2, i_jay1, i_jay2, i_m1, i_m2])
        )

        maxval = max(np.max(np.abs(plot_vals_real)), np.max(np.abs(plot_vals_imag)))
        minval = -maxval

        plt.rcParams.update({'font.size': 10})

        fig, axs = plt.subplots(nrows=1, ncols=2)
        cmap = cmr.fusion  # cmr.prinsenvlag  # pylint: disable=no-member
        fig.subplots_adjust(
            bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.4, hspace=0.1
        )
        cb_ax = fig.add_axes([0.85, 0.22, 0.02, 0.56])
        cb_ax.yaxis.set_offset_position('right')
           
        norm = mpl.colors.Normalize(vmin=minval, vmax=maxval)

        formatter = ScalarFormatter(useMathText=True) # useMathText for prettier formatting
        formatter.set_powerlimits((0,0)) # this will force scientific notation

        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cb_ax,
            orientation="vertical",
            # label= '$\\text{m}^{2}$',
            format=formatter
        )
        fig.text(0.87, 0.7875, '$\\text{m}^{2}$')
        # cb_ax.yaxis.set_major_formatter(formatter)

        axs[0].pcolormesh(k1_mesh*1000, k2_mesh*1000, plot_vals_real, cmap=cmap, norm=norm)
        axs[0].set_title("$\Re{\widetilde T^{111}_{111}(k_1,k_2)}$")

        axs[0].set_xlabel("Scattered wavenumber $k_1$, $\\text{$\mu$m}^{-1}$")
        # axs[0].xaxis.set_major_locator(plt.MaxNLocator(4))

        axs[0].set_ylabel("Incident wavenumber $k_2$, $\\text{$\mu$m}^{-1}$")
        axs[0].set_aspect('equal', adjustable='box')
        
        axs[0].set_xlim(xlim)
        axs[0].set_ylim(ylim)

        axs[1].pcolormesh(k1_mesh*1000, k2_mesh*1000, plot_vals_imag, cmap=cmap, norm=norm)
        axs[1].set_title("$\Im{\widetilde T^{111}_{111}(k_1,k_2)}$")
        axs[1].set_xlabel("Scattered wavenumber $k_1$, $\\text{$\mu$m}^{-1}$")
        axs[1].set_aspect('equal', adjustable='box')
        axs[1].yaxis.set_tick_params(labelleft=False)
        
        axs[1].set_xlim(xlim)
        axs[1].set_ylim(ylim)

        
        
    # def restore_diagonal(self, *indices):
    #      i_lam1, i_lam2, i_jay1, i_jay2, i_m1, i_m2 = rs.get_indices(
    #         self._jay_max, *indices
    #     )

    #     k1_mesh = np.zeros_like(self._k2_lists)
    #     k2_mesh = np.zeros_like(self._k2_lists)

    #     for i_k1, k1 in np.ndenumerate(self._k1_list):
    #         for i_k2, k2 in np.ndenumerate(np.squeeze(self._k2_lists[i_k1])):
    #             k1_mesh[i_k1, i_k2] = k1
    #             k2_mesh[i_k1, i_k2] = k2

    #     plot_vals_real = np.squeeze(
    #         np.real(self._vals[:, :, i_lam1, i_lam2, i_jay1, i_jay2, i_m1, i_m2])
    #     )
    #     plot_vals_imag = np.squeeze(
    #         np.imag(self._vals[:, :, i_lam1, i_lam2, i_jay1, i_jay2, i_m1, i_m2])
    #     )
        
    #     # Plot 
        
        