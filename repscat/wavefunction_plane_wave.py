"""
@author: Maxim Vavilin maxim@vavilin.de
"""

import numpy as np
import matplotlib.pyplot as plt
import repscat as rs
from scipy.interpolate import RegularGridInterpolator
from matplotlib.ticker import FormatStrFormatter, MultipleLocator

class WaveFunctionPlaneWave:
    def __init__(self, k_list, theta_list, phi_list, vals):
        """Domain must be k_list, theta_list, phi_list or k, theta_list, phi_list"""
        if len(vals.shape) == 3:
            self._type = "monochromatic"
            self._k = k_list
        elif len(vals.shape) == 4:
            self._type = "polychromatic"
            self._k_list = k_list
        else:
            raise ValueError("Wrong shape of values")

        self._vals = vals
        self._theta_list = theta_list
        self._phi_list = phi_list
        self._check_input()
        self._info={}

    def _check_input(self):
        if self._type == "polychromatic":
            if (
                self._vals.shape[0] != len(self._k_list)
                or self._vals.shape[2] != len(self._theta_list)
                or self._vals.shape[3] != len(self._phi_list)
            ):
                raise ValueError("Values do not correspond to k_list")

    def plot(self, linestyle='-'):
        # plt.plot(self._k_list, self._vals[:, 0, 2, 0], linestyle)
        plt.plot(self._theta_list, np.abs(self._vals[50, 0, :, 0]), linestyle)
            
    def plot_angular_photon_density(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        # Legend: k wavenumber, l helicity, t theta, p phi
        angular_photon_density = self._get_angular_photon_density()
        
        # ax.xaxis.set_major_formatter(FormatStrFormatter('%g $\pi$'))
        # ax.xaxis.set_major_locator(MultipleLocator(base=0.25))
        # labels = [r'$-\pi/4$', '$0$', r'$\pi/4$', r'$\pi/2$', r'$3\pi/4$', r'$\pi$']
        # ax.set_xticklabels(labels)
        
        ax.set_ylabel(r"Angular photon density $N_\theta$")
        ax.set_xlabel(r"Polar angle $\theta$")
        
        ax.plot(self._theta_list/np.pi, angular_photon_density)
        
        # indices = self._analyze_theta_content()
        # ax.plot(self._theta_list[indices]/np.pi, angular_photon_density[indices], '--')
        
        # plt.savefig(f"figs/Angular_photon_density.png", bbox_inches='tight')
        return ax
        
    def _get_angular_photon_density(self):
        k_list = self._k_list
        theta_list = self._theta_list
        phi_list = self._phi_list
        
        return np.einsum(
            "kltp,k,t,p->t",
            np.square(np.abs(self._vals)),
            np.diff(k_list, append=k_list[-1]) * k_list,
            np.sin(theta_list),
            np.diff(phi_list, append=phi_list[-1]),
            optimize=True
        )
        
    def plot_wavenumber_photon_density(self, ax=None):
        if ax is None:
            fig, ax = plt.subplots()

        wavenumber_photon_density = self.get_wavenumber_photon_density()
        
        ax.set_ylabel(r"Photon density per wavenumber $N_k$")
        ax.set_xlabel(r"Wavenumber, 1/micron")
        
        ax.plot(self._k_list*1000, wavenumber_photon_density)
        
        # plt.savefig(f"figs/Angular_photon_density.png", bbox_inches='tight')
        return ax
        
    def get_wavenumber_photon_density(self):
        k_list = self._k_list
        theta_list = self._theta_list
        phi_list = self._phi_list
        
        return np.einsum(
            "kltp,t,k,p->k",
            np.square(np.abs(self._vals)),
            np.diff(theta_list, append=theta_list[-1]) * np.sin(theta_list),
            k_list,
            np.diff(phi_list, append=phi_list[-1]),
            optimize=True
        )
        

        

    @property
    def info(self):
        return self._info
    
    @property
    def k_list(self):
        return self._k_list

    @property
    def theta_list(self):
        return self._theta_list

    @property
    def phi_list(self):
        return self._phi_list

    @property
    def vals(self):
        return self._vals

    def __add__(self, other):
        if not (
            np.allclose(self._k_list, other.k_list)
            and np.allclose(self._theta_list, other.theta_list)
            and np.allclose(self._phi_list, other.phi_list)
        ):
            raise ValueError("Domains of wavefunctions do not correspond")
        return WaveFunctionPlaneWave(
            self._k_list, self._theta_list, self._phi_list, self._vals + other.vals
        )
        
    def __sub__(self, other):
        if not (
            np.allclose(self._k_list, other.k_list)
            and np.allclose(self._theta_list, other.theta_list)
            and np.allclose(self._phi_list, other.phi_list)
        ):
            raise ValueError("Domains of wavefunctions do not correspond")
        return WaveFunctionPlaneWave(
            self._k_list, self._theta_list, self._phi_list, self._vals - other.vals
        )

    ##### Quantities and their transfer

    def photons(self):
        if self._type == "polychromatic":
            k_list = self._k_list
            theta_list = self._theta_list
            phi_list = self._phi_list

            photons = np.einsum(
                "bacd,b,c,d->",
                np.square(np.abs(self._vals)),
                np.diff(k_list, append=k_list[-1]) * k_list,
                np.diff(theta_list, append=theta_list[-1]) * np.sin(theta_list),
                np.diff(phi_list, append=phi_list[-1]),
            )
            return photons

    def energy(self):

        k_list = self._k_list
        theta_list = self._theta_list
        phi_list = self._phi_list

        energy = (
            np.einsum(
                "bacd,b,c,d->",
                np.square(np.abs(self._vals)),
                np.diff(k_list, append=k_list[-1]) * np.square(k_list),
                np.diff(theta_list, append=theta_list[-1]) * np.sin(theta_list),
                np.diff(phi_list, append=phi_list[-1]),
            )
            * rs.C_0_SI
            * rs.H_BAR_SI
            * 1e9
        )
        return energy

    def momentum_z(self):
        if self._type == "polychromatic":
            k_list = self._k_list
            theta_list = self._theta_list
            phi_list = self._phi_list

            momentum_z = (
                np.einsum(
                    "bacd,b,c,d->",
                    np.square(np.abs(self._vals)),
                    np.diff(k_list, append=k_list[-1]) * np.square(k_list),
                    np.diff(theta_list, append=theta_list[-1])
                    * np.sin(2 * theta_list)
                    / 2,  # sin()*cos()
                    np.diff(phi_list, append=phi_list[-1]),
                )
                * rs.H_BAR_SI
                * 1e9
            )

            return momentum_z
    def momentum_x(self):
        k_list = self._k_list
        theta_list = self._theta_list
        phi_list = self._phi_list

        momentum_x = (
            np.einsum(
                "bacd,b,c,d->",
                np.square(np.abs(self._vals)),
                np.diff(k_list, append=k_list[-1]) * np.square(k_list),
                np.diff(theta_list, append=theta_list[-1]),
                * np.sin(theta_list),
                * np.sin(theta_list),
                * np.cos(phi_list),
                np.diff(phi_list, append=phi_list[-1]),
            )
            * rs.H_BAR_SI
            * 1e9
        )

        return momentum_x
    
    def momentum_y(self):
        k_list = self._k_list
        theta_list = self._theta_list
        phi_list = self._phi_list

        momentum_y = (
            np.einsum(
                "bacd,b,c,d->",
                np.square(np.abs(self._vals)),
                np.diff(k_list, append=k_list[-1]) * np.square(k_list),
                np.diff(theta_list, append=theta_list[-1]),
                * np.sin(theta_list),
                * np.sin(theta_list),
                * np.sin(phi_list),
                np.diff(phi_list, append=phi_list[-1]),
            )
            * rs.H_BAR_SI
            * 1e9
        )

        return momentum_y

    def normalize(self):
        energy_ref = 1e-3
        energy = self.energy()
        if np.abs(energy - energy_ref) > 1e-5:
            print(f"Consider rescaling from {energy} to {energy_ref}")
            print("Multiply by : ", np.sqrt(energy_ref / energy))

    def in_angular_momentum_basis(self, jay_max):
        am_vals = rs.am_from_pw_vals(
            self._vals,
            self._theta_list,
            self._phi_list,
            jay_max,
        )
        return rs.WaveFunctionAngularMomentum(self._k_list, am_vals)
    
    def interpolate(self, k_list_new, theta_list_new):
        interp_vals = np.zeros(
            (
                len(k_list_new),
                self._vals.shape[1],
                len(theta_list_new),
                self._vals.shape[3],
            ),
            dtype=complex,
        )
        k_theta_new_grid = np.meshgrid(k_list_new, theta_list_new, indexing='ij')
        k_theta_new_list = np.reshape(k_theta_new_grid, (2, -1), order='C').T
        # print('k_theta_new_list.shape ', k_theta_new_list.shape)
        # print('self._k_list.shape ', self._k_list[0], self._k_list[-1], self._k_list.shape)
        # print('self._theta_list.shape ', self._theta_list[0], self._theta_list[-1], self._theta_list.shape)
        for i_lam in range(self._vals.shape[1]):
            for i_phi in range(self._vals.shape[3]):
                    real_interp = RegularGridInterpolator((self._k_list, self._theta_list), np.squeeze(np.real(self._vals[:, i_lam, :, i_phi])),bounds_error=False, fill_value=0)
                    # print('real int val ', real_interp([0.00817445, 0.00629578]))
                    # print(real_interp(k_theta_new_list).shape)
                    real_part = real_interp(k_theta_new_list).reshape((len(k_list_new), len(theta_list_new)))
                    imag_interp = RegularGridInterpolator((self._k_list, self._theta_list), np.squeeze(np.imag(self._vals[:, i_lam, :, i_phi])),bounds_error=False, fill_value=0)
                    imag_part = imag_interp(k_theta_new_list).reshape((len(k_list_new), len(theta_list_new)))

                    interp_vals[:, i_lam, :, i_phi] = real_part + 1j * imag_part

        return WaveFunctionPlaneWave(k_list_new, theta_list_new, self._phi_list, interp_vals)
    
    ##### Boost representation
    
    def _analyze_theta_content(self):
        angular_photon_density = self._get_angular_photon_density()
        
        max_val = np.max(angular_photon_density)
        
        threshold = max_val/100
        
        indices = np.where(angular_photon_density > threshold)
        indices = np.arange(indices[0][-1]+1) #  to include zero index
        return indices

    def _boost_domain(self, xi):
        boosted_theta_list = np.zeros_like(self._theta_list)
        boosted_k_array = np.zeros((len(self._theta_list), len(self._k_list)))
        boosted_theta_list = np.arccos(
            (np.cos(self._theta_list) + np.tanh(xi)) / (1 + np.cos(self._theta_list) * np.tanh(xi))
        )
        boosted_k_array = np.einsum(
            "k,e->ek", self._k_list, np.cosh(xi) + np.cos(self._theta_list) * np.sinh(xi)
        )
        return boosted_k_array, boosted_theta_list

    def _interpolate_boosted_vals(
        self, uniform_k_list, boosted_k_array, boosted_theta_list
    ):
        interp_vals = np.empty(self._vals.shape, dtype=complex)
        for i_lam in [0, 1]:
            for i_phi in range(len(self._phi_list)):
                for i_theta in range(len(boosted_theta_list)):
                    real_part = np.interp(
                        uniform_k_list,
                        boosted_k_array[i_theta, :],
                        np.real(self._vals[:, i_lam, i_theta, i_phi]),
                    )
                    imag_part = np.interp(
                        uniform_k_list,
                        boosted_k_array[i_theta, :],
                        np.imag(self._vals[:, i_lam, i_theta, i_phi]),
                    )
                    interp_vals[:, i_lam, i_theta, i_phi] = real_part + 1j * imag_part
        return interp_vals

    def get_uniform_truncated_boosted_k_list(self, xi):
        # theta_indices = self._analyze_theta_content()
        theta_min = self.theta_list[0]
        theta_max = self.theta_list[-1]
            
        if xi >= 0:
            k_min = self.k_list[0] * ( np.cosh(xi) + np.cos(theta_max)*np.sinh(xi) )
            k_max = self.k_list[-1] * ( np.cosh(xi) + np.cos(theta_min)*np.sinh(xi) )
        else:
            k_min = self.k_list[0] * ( np.cosh(xi) + np.cos(theta_min)*np.sinh(xi) )
            k_max = self.k_list[-1] * ( np.cosh(xi) + np.cos(theta_max)*np.sinh(xi) )
        
        return np.linspace(k_min, k_max, len(self.k_list))
    
    def boost(self, xi):
        """Active boost is meant"""

        uniform_truncated_boosted_k_list = self.get_uniform_truncated_boosted_k_list(xi)

        boosted_k_array, boosted_theta_list = self._boost_domain(xi)
        
        interpolated_vals = self._interpolate_boosted_vals(
            uniform_truncated_boosted_k_list, boosted_k_array, boosted_theta_list
        )

        return WaveFunctionPlaneWave(
            uniform_truncated_boosted_k_list, boosted_theta_list, self._phi_list, interpolated_vals
        )
    
        
    