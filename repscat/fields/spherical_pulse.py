# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name
# pylint: disable=unnecessary-pass

"""
Representations for spherical pulse
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import repscat as rs
# import repscat.aux_funcs as af
# import repscat.visuals as vis
# import treams.special as sp
# from repscat.constants import C_0
# from repscat.aux_funcs import cartesian_product

# import itertools
# from numpy.linalg import multi_dot

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "monospace",
        "font.monospace": "Computer Modern Typewriter",
    }
)

def get_spherical_pulse_wavefunction(lam, jay, m, center_wavelength, width_time, norm_fac, num_k, jay_max, k_list = None):
    ''' radial_type should be 'j', 'h-' or 'h+'
    '''
    if k_list is None:
        k_list = rs.get_k_list_gaussian(center_wavelength, width_time, num_k)

    vals = np.zeros((num_k, 2, jay_max, 2 * jay_max + 1), dtype=complex)
    i_lam = 0 if lam == 1 else 1
    i_jay = jay - 1
    i_m = m + jay_max
    k0 = 2*np.pi/center_wavelength
    vals[:, i_lam, i_jay, i_m] = norm_fac * rs.gaussian_wavenumber(k_list, k0, width_time)

    return rs.WaveFunctionAngularMomentum(k_list, vals)


# class WaveFunctionAngularMomentum(Representation):
#     def _get_domain(self):
#         if self.info.parameters["num_k"] > 1:
#             return {"k_list": Adapter.get_k_list_gaussian(self._info)}
#         if self.info.parameters["num_k"] == 1:
#             return {"k": 2 * np.pi / self.info.parameters["wavelength"]}

#     def _compute_vals(self):
#         # radial_function = self.info.parameters["radial_function"]
#         max_jay = self._info.parameters["max_jay"]
#         J = self._info.parameters["J"]
#         m = self._info.parameters["m"]

#         angular_vals = np.zeros((2, max_jay, 2 * max_jay + 1), dtype=complex)
#         i_lam = 0 if self._info.parameters["lam"] == 1 else 1
#         i_J = J - 1
#         i_m = m + max_jay
#         angular_vals[i_lam, i_J, i_m] = 1
#         if self.info.parameters["num_k"] > 1:
#             k_0 = 2 * np.pi / self._info.parameters["center_wavelength"]
#             dt = self._info.parameters["width_time"]
#             k_list = self._domain["k_list"]
#             wavenumber_vals = (
#                 af.gaussian_wavenumber(k_list, k_0, dt)
#                 * self.info.parameters["norm_fac"]
#             )
#             vals = np.einsum(
#                 "k,ajm->kajm", wavenumber_vals, angular_vals, optimize="greedy"
#             )
#             return vals
#         if self.info.parameters["num_k"] == 1:
#             return angular_vals

#     def _check_info(self, info):
#         if info.field_name != "SphericalPulse":
#             raise ValueError("Wrong physical field")
#         if info.representation_name != "WF_AM":
#             raise ValueError("Wrong wave function type")
#         # if not {"center_wavelength", "width_time", "num_k"} <= set(info.parameters):
#         #     raise ValueError("Need more info")

#     def get_spacetime_domain(self, plot_params):
#         num_r = plot_params["num_r"]
#         num_theta = plot_params["num_theta"]
#         t_list = plot_params["t_list"]
#         max_r = plot_params["max_r"]
#         num_phi = plot_params["num_phi"]
#         if num_phi%2 == 0:
#             raise ValueError('num_phi must be odd')

#         max_theta = np.pi
#         min_theta = 0
#         theta_list = np.linspace(min_theta, max_theta, num_theta, endpoint=True)

#         min_phi = 0
#         max_phi = 2*np.pi
#         phi_list = np.linspace(min_phi, max_phi, num_phi, endpoint=True)

#         min_r = 0
        
#         print(
#             f"Radius of view circle: {int(max_r)} nm",
#         )
#         r_list = np.linspace(min_r, max_r, num_r, endpoint=True)

#         spacetime_domain = {
#             "r_list": r_list,
#             "theta_list": theta_list,
#             "phi_list": phi_list,
#             "t_list": t_list,
#         }
#         return spacetime_domain


# class WaveFunctionPlaneWave(Representation):
#     def _get_domain(self):
#         return {
#             "k_list": af.get_k_list_gaussian(
#                 self.info.parameters["center_wavelength"],
#                 self.info.parameters["width_time"],
#                 self.info.parameters["num_k"],
#             ),
#             "eta_list": np.linspace(
#                 -1, 1, self._info.parameters["num_eta"], endpoint=True
#             ),
#             "phi_list": np.linspace(
#                 0, 2 * np.pi, self._info.parameters["num_phi"], endpoint=True
#             ),
#         }

#     def _compute_vals(self):
#         J = self.info.parameters["J"]
#         m = self.info.parameters["m"]
#         lam = self.info.parameters["lam"]
#         dt = self.info.parameters["width_time"]
#         sqrt_fac = np.sqrt((2 * J + 1) / (4 * np.pi))

#         k_0 = 2 * np.pi / self.info.parameters["center_wavelength"]
#         k_list = self.domain["k_list"]
#         eta_list = self.domain["eta_list"]
#         phi_list = self.domain["phi_list"]

#         small_d_list = sqrt_fac * sp.wignersmalld(J, m, lam, np.arccos(eta_list))
#         exp_k_list = af.gaussian_wavenumber(k_list, k_0, dt)
#         exp_phi_list = np.exp(1j * phi_list * m)

#         lam_arr = np.array([1, 0]) if lam == 1 else np.array([0, 1])
#         vals = (
#             np.einsum("a,b,c,d->abcd", lam_arr, exp_k_list, small_d_list, exp_phi_list)
#             * self.info.parameters["norm_fac"]
#         )

#         return vals

#     def _check_info(self, info):
#         if info.field_name != "SphericalPulse":
#             raise ValueError("Wrong physical field")
#         if info.representation_name != "WF_PW":
#             raise ValueError("Wrong wave function type")
#         if not {
#             "num_k",
#             "num_eta",
#             "num_phi",
#             "center_wavelength",
#             "width_time",
#             "norm_fac",
#         } <= set(info.parameters):
#             raise ValueError("Need more info")
