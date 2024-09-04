# pylint: disable=missing-function-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name
# pylint: disable=unnecessary-pass

"""
Custom pulse
"""
import numpy as np
from repscat.representation import Representation

class WaveFunctionPlaneWave(Representation):
    def _compute_vals(self):
        return 0

    def _get_domain(self):
        return 0

    def _check_info(self, info):
        if info.basis_type != "WF_PW":
            raise ValueError("Wave function should be in plane wave basis.")

class WaveFunctionAngularMomentum(Representation):
    def _get_domain(self):
        return 0

    def _compute_vals(self):
        return 0

    def _check_info(self, info):
        if info.basis_type != "WF_AM":
            raise ValueError("Wave function should be in angular momentum basis.")

    def get_spacetime_domain(self, plot_params):
        num_r = plot_params["num_r"]
        num_theta = plot_params["num_theta"]
        t_list = plot_params["t_list"]
        max_r = plot_params["max_r"]
        num_phi = plot_params["num_phi"]

        max_theta = np.pi
        min_theta = 0
        theta_list = np.linspace(min_theta, max_theta, num_theta, endpoint=True)

        min_phi = 0
        max_phi = 2*np.pi
        phi_list = np.linspace(min_phi, max_phi, num_phi, endpoint=True)

        min_r = 0
        
        print(
            f"Radius of view circle: {int(max_r)} nm",
        )
        r_list = np.linspace(min_r, max_r, num_r, endpoint=True)

        spacetime_domain = {
            "r_list": r_list,
            "theta_list": theta_list,
            "phi_list": phi_list,
            "t_list": t_list,
        }
        return spacetime_domain