"""
@author: Maxim Vavilin maxim@vavilin.de
"""


import os
import numpy as np
from repscat.config import DATA_PATH
import matplotlib.pyplot as plt
import copy
from wigners import wigner_3j

# from repscat.boost_tmat import boost_tmat_vals
import matplotlib as mpl

# from repscat.fields.custom_pulse import WaveFunctionAngularMomentum as Custom_pulse_wf_am
from repscat.constants import C_0_SI, H_BAR_SI, EPSILON_0_SI


class Silicon:
    """Contains info about silicon"""

    def __init__(self):
        self._SILICON_PATH = os.path.join(DATA_PATH, "Si_n_k.txt")
        self._refr_idx_list_experiment = None
        self._extinction_list_experiment = None
        self._k_list_experiment = self._read_k_list_experiment()
        self._eps_list_experiment = self._get_eps_list_experiment()

    def _read_k_list_experiment(self):
        with open(self._SILICON_PATH, "r") as f:
            lines = f.readlines()
            lines = lines[1:]
        lam_n_k_data = np.zeros((np.shape(lines)[0], 3), dtype=float)
        for i_line, line in enumerate(lines):
            for i in range(3):
                lam_n_k_data[i_line, i] = float(line.split()[i])
        wavelength_data_list = lam_n_k_data[:, 0]
        k_list_experiment = 2 * np.pi / np.flip(wavelength_data_list)

        return k_list_experiment

    def _get_eps_list_experiment(self):
        with open(self._SILICON_PATH, "r") as f:
            lines = f.readlines()
            lines = lines[1:]
        lamnk_data_arr = np.zeros((np.shape(lines)[0], 3), dtype=float)
        for i_line, line in enumerate(lines):
            for i in range(3):
                lamnk_data_arr[i_line, i] = float(line.split()[i])

        refr_data_list = np.flip(lamnk_data_arr[:, 1])
        self._refr_idx_list_experiment = refr_data_list

        ext_data_list = np.flip(lamnk_data_arr[:, 2])
        self._extinction_list_experiment = ext_data_list

        eps_list = (
            refr_data_list**2
            - ext_data_list**2
            + 1j * (refr_data_list * ext_data_list)
        )
        return eps_list

    def plot(self, ax=None, k_list=None):
        if ax is None:
            fig, ax = plt.subplots()
        if k_list is None:
            ax.plot(self._k_list_experiment, np.real(self._eps_list_experiment))
            ax.plot(self._k_list_experiment, np.imag(self._eps_list_experiment))
        else:
            if k_list[0]<self._k_list_experiment[0] or k_list[-1]>self._k_list_experiment[-1]:
                raise ValueError('Bad k-list:', k_list[0], "<", self._k_list_experiment[0])
            refr_interpol_list = np.interp(
                k_list, self._k_list_experiment, self._refr_idx_list_experiment
            )
            ext_interpol_list = np.interp(
                k_list, self._k_list_experiment, self._extinction_list_experiment
            )
            
            ax.plot(k_list*1000, refr_interpol_list, '-k', label=r"$n(k)$")
            ax.plot(k_list*1000, ext_interpol_list, '--k', label=r"$\kappa(k)$")
        return ax


    @property
    def k_list_experiment(self):
        return self._k_list_experiment

        # eps_list = interpolate_silicon_epsilon(k_list, silicon_n_k_path)

    # def interpolate_silicon_epsilon(k_list, silicon_n_k_path_name):
    #     with open(silicon_n_k_path_name, "r") as f:
    #         lines = f.readlines()
    #         lines = lines[1:]
    #     lamnk_data_arr = np.zeros((np.shape(lines)[0], 3), dtype=float)
    #     for i_line, line in enumerate(lines):
    #         for i in range(3):
    #             lamnk_data_arr[i_line, i] = float(line.split()[i])
    #     wavelength_data_list = lamnk_data_arr[:, 0]
    #     k_data_list = 2 * np.pi / np.flip(wavelength_data_list)
    #     if k_list[0]<k_data_list[0] or k_list[-1]>k_data_list[-1]:
    #         raise ValueError(f'k_list outside of experimental values {k_list[0]}<{k_data_list[0]} or {k_list[-1]}>{k_data_list[-1]}')
    #     refr_data_list = np.flip(lamnk_data_arr[:, 1])
    #     ext_data_list = np.flip(lamnk_data_arr[:, 2])
    #     refr_interpol_list = np.interp(k_list, k_data_list, refr_data_list)
    #     ext_interpol_list = np.interp(k_list, k_data_list, ext_data_list)
    #     eps_list = (
    #         refr_interpol_list**2
    #         - ext_interpol_list**2
    #         + 1j * (refr_interpol_list * ext_interpol_list)
    #     )
    #     # k_eps_arr = np.stack((k_list, eps_list), 0)
    #     # fig, ax = plt.subplots( nrows=1, ncols=1 )
    #     # ax.plot(k_data_list,  refr_data_list)
    #     # ax.plot(k_data_list,  ext_data_list)
    #     # plt.show()
    #     # print(f'Minimal k: {k_data_list[0]}\nMaximal k: {k_data_list[-1]}')
    #     return eps_list
