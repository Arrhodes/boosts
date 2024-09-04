# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name
# pylint: disable=unnecessary-pass

"""
Representations for spherical pulse
"""
import numpy as np
from reptemscat.representation import Representation, Adapter
import reptemscat.aux_funcs as af
import treams.special as sp
from reptemscat.constants import C_0
from reptemscat.aux_funcs import cartesian_product

# import itertools
from numpy.linalg import multi_dot


class WaveFunctionAngularMomentum(Representation):
    def _get_domain(self):
        return {"k_list": Adapter.get_k_list_gaussian(self._info)}

    def _compute_vals(self):
        k_0 = 2 * np.pi / self._info.parameters["center_wavelength"]
        dt = self._info.parameters["width_time"]
        k_list = self._domain["k_list"]
        max_J = self._info.parameters["max_J"]
        J = self._info.parameters["J"]
        m = self._info.parameters["m"]

        vals = np.zeros((2, len(k_list), (max_J +1 )**2 - 1), dtype = complex)
        i_lam = 0 if self._info.parameters["lam"] == 1 else 1
        vals[i_lam,:,af.get_idx(J, m)] = (
            af.gaussian_wavenumber(self._domain["k_list"], k_0, dt)
            * self.info.parameters["norm_fac"]
        )
        return vals

    def _check_info(self, info):
        if info.field_name != "SphericalPulse":
            raise ValueError("Wrong physical field")
        if info.representation_name != "WF_AM":
            raise ValueError("Wrong wave function type")
        # if not {"center_wavelength", "width_time", "num_k"} <= set(info.parameters):
        #     raise ValueError("Need more info")


class WaveFunctionPlaneWave(Representation):
    def _get_domain(self):
        return {
            "k_list": af.get_k_list_gaussian(
                self.info.parameters["center_wavelength"],
                self.info.parameters["width_time"],
                self.info.parameters["num_k"],
            ),
            "eta_list": np.linspace(
                -1, 1, self._info.parameters["num_eta"], endpoint=True
            ),
            "phi_list": np.linspace(
                0, 2 * np.pi, self._info.parameters["num_phi"], endpoint=True
            ),
        }

    def _compute_vals(self):
        J = self.info.parameters["J"]
        m = self.info.parameters["m"]
        lam = self.info.parameters["lam"]
        dt = self.info.parameters["width_time"]
        sqrt_fac = np.sqrt((2 * J + 1) / (4 * np.pi))

        k_0 = 2 * np.pi / self.info.parameters["center_wavelength"]
        k_list = self.domain["k_list"]
        eta_list = self.domain["eta_list"]
        phi_list = self.domain["phi_list"]

        small_d_list = sqrt_fac * sp.wignersmalld(J, m, lam, np.arccos(eta_list))
        # print(small_d_list)
        exp_k_list = af.gaussian_wavenumber(k_list, k_0, dt)
        # print(exp_k_list)
        exp_phi_list = np.exp(1j * phi_list * m)
        # print(exp_phi_list)
        lam_list = np.array([1, 0]) if lam == 1 else np.array([0, 1])

        # print(small_d_list[-1], exp_k_list[-1], exp_phi_list[-1])
        # print(self.info.parameters["norm_fac"])
        vals = (
            np.einsum("a,b,c,d->abcd", lam_list, exp_k_list, small_d_list, exp_phi_list)
            * self.info.parameters["norm_fac"]
        )
        # print(np.abs(vals[0, 25, :, 5]))
        return vals


    def _check_info(self, info):
        if info.field_name != "SphericalPulse":
            raise ValueError("Wrong physical field")
        if info.representation_name != "WF_PW":
            raise ValueError("Wrong wave function type")
        if not {
            "num_k",
            "num_eta",
            "num_phi",
            "center_wavelength",
            "width_time",
            "norm_fac",
        } <= set(info.parameters):
            raise ValueError("Need more info")
