"""
@author: Maxim Vavilin maxim@vavilin.de
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import copy
from wigners import wigner_3j

# from repscat.boost_tmat import boost_tmat_vals
import matplotlib as mpl

import repscat as rs


class TmatrixPoly:
    """Polychromatic T-matrix in form (k k lam lam j j m m)
    units 1/m^2
    """

    def __init__(self, k1_list, k2_list, vals, radius=None):
        """Domain should be k1_list, k2_list"""
        self._check_domain_vs_vals(k1_list.shape, k2_list.shape, vals.shape)
        self._k1_list = k1_list
        self._k2_list = k2_list

        self._vals = vals
        self._radius = radius
        self._jay_max = vals.shape[4]

    def _check_domain_vs_vals(self, k1shape, k2shape, vshape):
        if not (k1shape[0] == vshape[0] and k2shape[0] == vshape[1]):
            raise ValueError(
                f"Domain shape does not correspond to values"
            )


    @property
    def k1_list(self):
        return self._k1_list

    @property
    def k2_list(self):
        return self._k2_list

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
        if not (np.allclose(self._k2_list, incident.k_list)):
            print(self._k2_list - incident.k_list)
            raise ValueError(
                f"Tmatrix k2_list does not correspond to incident k_list: {self._k2_list[0]}, {incident.k_list[0]}"
            )

        k1_list = self._k1_list
        k2_list = self._k2_list
        measure = np.diff(k2_list, append=k2_list[-1]) * k2_list
        scattered_vals = np.einsum(
            "p,kpabijmn,pbjn->kaim",
            measure,
            self._vals,
            incident.vals,
            optimize=True,
        )

        return rs.WaveFunctionAngularMomentum(k1_list, scattered_vals)

    def plot(self, *indices):
        i_lam1, i_lam2, i_jay1, i_jay2, i_m1, i_m2 = rs.get_indices(self._jay_max, *indices)
        k1_mesh, k2_mesh = np.meshgrid(self._k1_list, self._k2_list, indexing="ij")
        plot_vals_real = np.squeeze(
            np.real(self._vals[:, :, i_lam1, i_lam2, i_jay1, i_jay2, i_m1, i_m2])
        )
        plot_vals_imag = np.squeeze(
            np.imag(self._vals[:, :, i_lam1, i_lam2, i_jay1, i_jay2, i_m1, i_m2])
        )

        maxval = max(np.max(np.abs(plot_vals_real)), np.max(np.abs(plot_vals_imag)))
        minval = -maxval
        fig, axs = plt.subplots(nrows=2, ncols=1)
        cmap = mpl.cm.seismic  # pylint: disable=no-member
        fig.subplots_adjust(
            bottom=0.1, top=0.9, left=0.1, right=0.8, wspace=0.4, hspace=0.1
        )
        cb_ax = fig.add_axes([0.83, 0.1, 0.02, 0.8])
        norm = mpl.colors.Normalize(vmin=minval, vmax=maxval)
        fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cb_ax,
            orientation="vertical",
        )
        axs[0].pcolormesh(k1_mesh, k2_mesh, plot_vals_real, cmap=cmap, norm=norm)
        axs[0].set_ylabel("Real part")
        axs[1].pcolormesh(k1_mesh, k2_mesh, plot_vals_imag, cmap=cmap, norm=norm)
        axs[1].set_ylabel("Imaginary part")
        
        plt.savefig('tmat_elem.png')        

    def compute_transfer(self, incident):

        scattered = self.scatter(incident)
        incident_interp = incident.interpolate(scattered.k_list)
        outgoing = incident_interp + scattered

        return {
            'energy': incident.energy() - outgoing.energy(),
            'momentum_z': incident.momentum_z() - outgoing.momentum_z()
        }
