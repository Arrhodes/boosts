"""
@author: Maxim Vavilin maxim@vavilin.de
"""
import os
import numpy as np
import treams
import treams.io
from repscat.config import DATA_PATH
from repscat.aux_funcs import interpolate_silicon_epsilon
from repscat.tmatrix_diag import TmatrixDiagonal
import h5py
import matplotlib.pyplot as plt


def get_silicon_tmat(radius, max_J, k_list, distance=0):
    ''' Gives np array of shape (2, (max_J +1)**2 - 1 , 2, (max_J +1)**2 - 1)
    '''
    # print("Computing T-matrix")
    silicon_n_k_path = os.path.join(DATA_PATH, "Si_n_k.txt")
    eps_list = interpolate_silicon_epsilon(k_list, silicon_n_k_path) 
    tmat = compute_tmat_sphere_vals(eps_list, k_list, max_J, radius, distance) # make     if isinstance(k_list, list):
        # with h5py.File('two_spheres', "w") as tmat_file:
        #     treams.io.save_hdf5(
        #         tmat_file,
        #         tmat,
        #         name="SiliconBall",
        #         uuid=None,
        #         uuid_version=4,
        #         lunit="nm",

    return TmatrixDiagonal(k_list, np.array(tmat))


def compute_tmat_sphere_vals(eps_list, k_list, max_J, radius, distance=0):
    air = treams.Material()
    # change_mat = treams.changepoltype(basis=my_basis)

    tmat = np.empty((len(k_list), 2, 2, max_J, max_J, 2 * max_J + 1, 2 * max_J + 1), dtype = complex)
    for i_k, k in np.ndenumerate(k_list):
        i_k=i_k[0]
        silicon = treams.Material(eps_list[i_k])
        silicon_sphere = treams.TMatrix.sphere(max_J, k, radius, [silicon, air])
        # if distance != 0:
        #     print("Translating", i_k)
        #     tr_vec = np.array([0, 0, distance])
        #     silicon_sphere = (
        #         treams.TMatrix.cluster(
        #             [silicon_sphere, silicon_sphere], [tr_vec, -tr_vec]
        #         )
        #         .interaction.solve()
        #         .expand(treams.SphericalWaveBasis.default(max_J))
        #     )
        tmat[i_k] = 2 * change_basis_ordering(np.array(silicon_sphere), max_J)
    return tmat
    # if k_eps_arr.ndim == 1:
    
def get_gold_cluster_tmat(radius, distance, max_J, k_list):
    ''' Gives np array of shape (2, (max_J +1)**2 - 1 , 2, (max_J +1)**2 - 1)
    '''
    gold_n_k_path = os.path.join(DATA_PATH, "Au_n_k.txt")
    eps_list = interpolate_silicon_epsilon(k_list, gold_n_k_path) 
    tmat = compute_tmat_vals_gold_quadrat(eps_list, k_list, max_J, radius, distance) # make     if isinstance(k_list, list):
        # with h5py.File('two_spheres', "w") as tmat_file:
        #     treams.io.save_hdf5(
        #         tmat_file,
        #         tmat,
        #         name="SiliconBall",
        #         uuid=None,
        #         uuid_version=4,
        #         lunit="nm",

    return TmatrixDiagonal(k_list, np.array(tmat))


def compute_tmat_vals_gold_quadrat(eps_list, k_list, max_J, radius, distance):
    air = treams.Material()
    # change_mat = treams.changepoltype(basis=my_basis)

    tmat = np.empty((len(k_list), 2, 2, max_J, max_J, 2 * max_J + 1, 2 * max_J + 1), dtype = complex)
    for i_k, k in np.ndenumerate(k_list):
        i_k=i_k[0]
        gold = treams.Material(eps_list[i_k])
        golden_sphere = treams.TMatrix.sphere(max_J, k, radius, [gold, air])

        tr_vec1 = np.array([distance, distance, 0])
        tr_vec2 = np.array([distance, -distance, 0])
        tr_vec3 = np.array([-distance, distance, 0])
        tr_vec4 = np.array([-distance, -distance, 0])
        golden_triangle = (
            treams.TMatrix.cluster(
                [golden_sphere, golden_sphere, golden_sphere, golden_sphere], [tr_vec1, tr_vec2, tr_vec3, tr_vec4]
                #[golden_sphere], [tr_vec1]
            )
            .interaction.solve()
            .expand(treams.SphericalWaveBasis.default(max_J))
        )
        tmat[i_k] = 2 * change_basis_ordering(np.array(golden_triangle), max_J)
    return tmat
    # if k_eps_arr.ndim == 1:

def get_gold_sphere_tmat(radius, max_J, k_list, distance=0):
    ''' Gives np array of shape (2, (max_J +1)**2 - 1 , 2, (max_J +1)**2 - 1)
    '''
    # print("Computing T-matrix")
    silicon_n_k_path = os.path.join(DATA_PATH, "Au_n_k.txt")
    eps_list = interpolate_silicon_epsilon(k_list, silicon_n_k_path) 
    tmat = compute_tmat_sphere_vals(eps_list, k_list, max_J, radius, distance) # make     if isinstance(k_list, list):
        # with h5py.File('two_spheres', "w") as tmat_file:
        #     treams.io.save_hdf5(
        #         tmat_file,
        #         tmat,
        #         name="SiliconBall",
        #         uuid=None,
        #         uuid_version=4,
        #         lunit="nm",

    return TmatrixDiagonal(k_list, np.array(tmat))



def change_basis_ordering(tmat, max_J):
    my_tmat = np.zeros((2, 2, max_J, max_J, 2 * max_J + 1, 2 * max_J + 1), dtype=complex)
    idx_1 = 0
    for J_1 in range(1, max_J + 1):
        i_J_1 = J_1 - 1
        for m_1 in range(-J_1, J_1 + 1):
            i_m_1 = m_1 + max_J
            for i_lam_1 in [0, 1]:
                idx_2 = 0
                for J_2 in range(1, max_J + 1):
                    i_J_2 = J_2 - 1
                    for m_2 in range(-J_2, J_2 + 1):
                        i_m_2 = m_2 + max_J
                        for i_lam_2 in [0, 1]:
                            my_tmat[i_lam_1, i_lam_2, i_J_1, i_J_2, i_m_1, i_m_2] = tmat[idx_1, idx_2]
                            idx_2 += 1
                idx_1 += 1
    return my_tmat

    # Lams = tmat.basis.pol
    # # T_list = []
    # my_tmat = np.zeros((2, (max_J +1)**2 - 1 , 2, (max_J +1)**2 - 1), dtype = complex)
    # for i_Lam1, Lam1 in np.ndenumerate(np.array([1, 0])):
    #     for i_Lam2, Lam2 in np.ndenumerate(np.array([1, 0])):
    #         T_lam1_lam2 = tmat[np.outer(Lams == Lam1, Lams == Lam2)]
    #         T_lam1_lam2 = np.array(T_lam1_lam2).reshape(
    #             max_J * (max_J + 2), max_J * (max_J + 2)
    #         )
    #         my_tmat[i_Lam1, :, i_Lam2, :] = T_lam1_lam2
    #         # T_list.append(T_lam1_lam2)

    # tmat_arr_new = np.block([[T_list[0], T_list[1]], [T_list[2], T_list[3]]])
    # new_basis = treams.SphericalWaveBasis(
    #     [(l, m, s) for s in range(2) for l in range(1, max_J) for m in range(-l, l + 1)]
    # )

    # tmat_new = treams.TMatrix(
    #     tmat_arr_new,
    #     k0=tmat.k0,
    #     material=tmat.material,
    #     basis=new_basis,
    #     poltype=tmat.poltype,
    # )
    # return my_tmat


# def get_k_eps_arr(center_wavelength, num_k):
#     k_eps_path = os.path.join(
#         DATA_PATH,
#         "wavenumber_epsilon_arrays",
#         f"k_eps_arr_[center_wavelength]=[{center_wavelength}]_[num_k]=[{num_k}].npy",
#     )
#     with open(k_eps_path, "rb") as file:
#         return np.load(file)
