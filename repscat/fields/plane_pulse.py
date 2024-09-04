# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name
# pylint: disable=unnecessary-pass
"""
@author: Maxim Vavilin maxim.vavilin@kit.edu
"""

import numpy as np
from repscat.representation import Representation, Adapter
from repscat.aux_funcs import get_k_list_gaussian
import copy

# from reptemscat.transfer import get_quantity_wf_pw

# from reptemscat.aux_funcs import gaussian_wavenumber
from repscat.init_wf import (
    get_plane_pulse_wave_func_am,
    get_plane_pulse_wave_func_pw,
)


class WaveFunctionPlaneWave(Representation):
    def _compute_vals(self):
        return get_plane_pulse_wave_func_pw(
            self._domain["k_list"],
            self._domain["eta_list"],
            self._domain["phi_list"],
            self._info.parameters["center_wavelength"],
            self._info.parameters["width_time"],
            self._info.parameters["width_space"],
            self._info.parameters["norm_fac"],
            self._info.parameters["if_positive_helicity"],
            # self._info.parameters["if_positive_eta"],
        )

    def _get_domain(self):
        raise ValueError('Use custom domain for eta')
        # return {
        #     "k_list": get_k_list_gaussian(
        #     self.info.parameters["center_wavelength"],
        #     self.info.parameters["width_time"],
        #     self.info.parameters["num_k"],
        # ),
        #     "eta_list": np.linspace(
        #         0.975, 1, self._info.parameters["num_eta"], endpoint=True
        #     ),
        #     "phi_list": np.linspace(
        #         0, 2 * np.pi, self._info.parameters["num_phi"], endpoint=True
        #     ),
        # }

    def _check_info(self, info):
        if info.field_name != "PlanePulse":
            raise ValueError("Wrong physical field")
        if info.basis_type != "WF_PW":
            raise ValueError("Wrong wave function type")
        if not {
            "if_positive_helicity",
            # "if_positive_eta",
            "center_wavelength",
            "width_time",
            "width_space",
            "norm_fac",
        } <= set(info.parameters):
            raise ValueError("Need more info")

class WaveFunctionAngularMomentum(Representation):
    def _get_domain(self):
        return {"k_list": Adapter.get_k_list_gaussian(self._info)}

    def _compute_vals(self):
        return get_plane_pulse_wave_func_am(
            self._domain["k_list"],
            self._info.parameters["max_jay"],
            self._info.parameters["center_wavelength"],
            self._info.parameters["width_time"],
            self._info.parameters["width_space"],
            self._info.parameters["norm_fac"],
            self._info.parameters["if_positive_helicity"],
            # self._info.parameters["if_positive_eta"],
        )

    def _check_info(self, info):
        if info.field_name != "PlanePulse":
            raise ValueError("Wrong physical field")
        if info.basis_type != "WF_AM":
            raise ValueError("Wrong wave function type")
        # if not {
        #     "if_positive_helicity",
        #     "if_positive_eta",
        #     "max_J",
        #     "num_k",
        #     "center_wavelength",
        #     "width_time",
        #     "width_space",
        #     "norm_fac",
        # } <= set(info.parameters):
        #     raise ValueError("Need more info")

                    
