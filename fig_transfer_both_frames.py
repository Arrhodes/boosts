"""
@author: Maxim Vavilin maxim.vavilin@kit.edu
"""

import os
import math
import numpy as np
from numpy import linalg as la
import matplotlib.pyplot as plt
import matplotlib as mpl
import treams
import copy
import time
import treams.special as sp

import repscat as rs


import fig_interaction_cross_section_of_speed
import fig_absorption_of_speed
import mpl_axes_aligner

def boost(energy, momentum, xi):
    energy_new = np.cosh(xi) * energy + np.sinh(xi) * momentum * rs.C_0_SI
    mom_new = np.sinh(xi) * energy / rs.C_0_SI + np.cosh(xi) * momentum
    return energy_new, mom_new

def get_significand(x):
    if x == 0:
        return 0
    exponent = math.floor(math.log10(abs(x)))
    significand = x / (10 ** exponent)
    return significand

def scale_ax2_to_match_ax1(ax1, ax2):
    ymax1_significand = get_significand(ax1.get_ylim()[1])
    ymax2_significand = get_significand(ax2.get_ylim()[1])

    scale_factor = ymax1_significand / ymax2_significand
    ax2.set_ylim(0, ax2.get_ylim()[1] * scale_factor)
    


# def save_figure(
#     file_name, y_val, y_label, xi_list, cross_sec_args, absorption_or_crossec, if_grid
# ):
#     fig, ax1 = plt.subplots()
    
#     linecolor = "darkred"
#     ax1.plot(np.tanh(xi_list), y_val, linestyle="-", color=linecolor, zorder=3)

#     ax1.set_xlabel("v/c")
#     ax1.set_ylabel(y_label, color=linecolor)
#     ax1.tick_params(axis="y", labelcolor=linecolor)
    
#     ax1.set_xticks(np.arange(-0.75, 0.75+0.25, 0.25))

#     ax1.set_zorder(1)  # puts first plot on top of the second
#     ax1.patch.set_visible(False)
    
#     ax1.xaxis.grid(True, linestyle=':', color='black', linewidth=0.5)
#     ax1.yaxis.grid(True, linestyle=':', color='black', linewidth=0.5)

#     if absorption_or_crossec == "crossec":
#         ax2 = ax1.twinx()
#         fig_interaction_cross_section_of_speed.plot(*cross_sec_args, ax2)
#         scale_ax2_to_match_ax1(ax1, ax2)
#         mpl_axes_aligner.align.yaxes(ax1, 0, ax2, 0, 0.05)
        
#     elif absorption_or_crossec == "absorption":
#         ax2 = ax1.twinx()
#         fig_absorption_of_speed.plot(*cross_sec_args, ax2)
#         scale_ax2_to_match_ax1(ax1, ax2)
#         mpl_axes_aligner.align.yaxes(ax1, 0, ax2, 0, 0.05)
    
#     fig.savefig(file_name, bbox_inches="tight")


def get_energy_difference(incident_pw, outgoing_pw):
    return incident_pw.energy() - outgoing_pw.energy()

def get_momentum_difference(incident_pw, outgoing_pw):
    return incident_pw.momentum_z() - outgoing_pw.momentum_z()

# def compute_transfer_pulse_lab():
#     energy_diff_lab, momentum_diff_lab = boost(
#             get_energy_difference(inc_pw_obj, out_pw_obj),
#             get_momentum_difference(inc_pw_obj, out_pw_obj),
#             xi,
#         )

#     energy_transfer_list_lab.append(energy_diff_lab)
#     momentum_transfer_list_lab.append(momentum_diff_lab)
    
    
def get_silicon_sphere_mass(radius):
    radius_si = radius*1e-9
    rho_silicon = 2330
    return 4/3 * np.pi * radius_si**3 * rho_silicon
    
def get_relativistic_energy_from(momentum, mass):
    return rs.C_0_SI*np.sqrt(mass**2 * rs.C_0_SI**2 + momentum**2)

def compute_transfer_object_comoving(energy_from_pulse_comoving, momentum_from_pulse_comoving, radius):
    mass_of_sphere = get_silicon_sphere_mass(radius)
    momentum_of_object = momentum_from_pulse_comoving
    relativistic_energy_diff_of_object = get_relativistic_energy_from(momentum_of_object, mass_of_sphere) - get_relativistic_energy_from(0, mass_of_sphere)
    
    thermal_gain_of_object = energy_from_pulse_comoving - relativistic_energy_diff_of_object
    
    return thermal_gain_of_object, relativistic_energy_diff_of_object, momentum_of_object

def get_momentum_from(xi, mass_of_sphere):
    v = rs.C_0_SI*np.tanh(xi)
    gamma = np.sqrt(1-(v/rs.C_0_SI)**2)
    return mass_of_sphere*v*gamma
    
def compute_transfer_object_lab(xi_list, energy_from_pulse_comoving, momentum_from_pulse_comoving, radius):
    energy_diff_pulse_lab, momentum_diff_pulse_lab = boost(energy_from_pulse_comoving, momentum_from_pulse_comoving, xi_list)
    
    mass_of_sphere = get_silicon_sphere_mass(radius)
    
    momentum_diff_of_objec_lab = momentum_diff_pulse_lab
    momentum_before_object_lab = get_momentum_from(xi_list, mass_of_sphere)
    momentum_after_object_lab = momentum_before_object_lab + momentum_diff_of_objec_lab
    
    relativistic_energy_diff_object_lab = get_relativistic_energy_from(momentum_after_object_lab, mass_of_sphere) - get_relativistic_energy_from(momentum_before_object_lab, mass_of_sphere)

    thermal_gain_of_object_lab = energy_diff_pulse_lab - relativistic_energy_diff_object_lab
    
    return thermal_gain_of_object_lab, relativistic_energy_diff_object_lab, momentum_diff_of_objec_lab
    
    
    
def get_second_theta_list(first_theta_list):
    num_theta = len(first_theta_list)
    theta_min = first_theta_list[-1]
    return np.linspace(theta_min, np.pi, num_theta)

def get_energy_momentum_out(tmat_name, jay_max, incoming_pw_obj):
    
    incoming_am_obj = incoming_pw_obj.in_angular_momentum_basis(jay_max)
    if tmat_name == 'silicon':
        radius = 150
        tmat_diag_obj = rs.get_silicon_tmat(radius, jay_max, incoming_am_obj.k_list)
    elif tmat_name == 'gold_cluster':
        radius = 50
        distance = 100
        tmat_diag_obj = rs.get_gold_cluster_tmat(radius, distance, jay_max, incoming_am_obj.k_list)
    elif tmat_name == 'gold_sphere':
        radius = 150
        tmat_diag_obj = rs.get_gold_sphere_tmat(radius, jay_max, incoming_am_obj.k_list)

    scat_am_obj = tmat_diag_obj.scatter(incoming_am_obj)

    # First part of the outgoing field interferes with the incident field
    scat_pw_obj_first_part = scat_am_obj.in_plane_wave_basis(
        incoming_pw_obj.theta_list, incoming_pw_obj.phi_list
    )

    # Second part of the outgoing field does not interfere with the incident field
    second_theta_list = get_second_theta_list(incoming_pw_obj.theta_list)
    scat_pw_obj_second_part = scat_am_obj.in_plane_wave_basis(
        second_theta_list, incoming_pw_obj.phi_list
    )
    
    energy_out = (incoming_pw_obj + scat_pw_obj_first_part).energy() + scat_pw_obj_second_part.energy()
    momentum_out = (incoming_pw_obj + scat_pw_obj_first_part).momentum_z() + scat_pw_obj_second_part.momentum_z()
    
    # fig, ax = plt.subplots()
    # incoming_pw_obj.plot_angular_photon_density(ax)
    # (incoming_pw_obj + scat_pw_obj_first_part).plot_angular_photon_density(ax)
    # (scat_pw_obj_second_part).plot_angular_photon_density(ax)
    # plt.show()
    
    return energy_out, momentum_out
    
def compute_transfer_pulse_comoving(tmat_name, incident_pw_lab, jay_max, xi_list):
    xi_max = xi_list[-1]
    len_xi = len(xi_list)
    
    energy_diffs = []
    momentum_diffs = []

    for xi in xi_list:
        # print("xi = ", xi)
        # Switches to object frame
        inc_pw_obj = incident_pw_lab.boost(-xi)
        
        energy_in = inc_pw_obj.energy()
        momentum_in = inc_pw_obj.momentum_z()
        
        energy_out, momentum_out = get_energy_momentum_out(tmat_name, jay_max, inc_pw_obj)
        
        energy_diffs.append(energy_in - energy_out)
        momentum_diffs.append(momentum_in - momentum_out)

    np.save(f"xi_list_{xi_max}_{len_xi}.npy", np.array(xi_list))
    np.save(f"Energy_transfer_pulse_{xi_max}_{len_xi}.npy", energy_diffs)
    np.save(f"Momentum_transfer_pulse_{xi_max}_{len_xi}.npy", momentum_diffs)

    return np.real(energy_diffs), np.real(momentum_diffs)

def plot_energies(xi_list, e_field, e_thermal, e_relativistic):
    fig, ax = plt.subplots()
    
    ax.plot(np.tanh(xi_list), e_field, linestyle="-", color="darkred")
    ax.plot(np.tanh(xi_list), e_relativistic, linestyle="--", color="darkolivegreen")
    ax.plot(np.tanh(xi_list), e_thermal, linestyle="-", color="darkolivegreen")

    ax.set_xlabel("v/c")

    ax.xaxis.grid(True)
    ax.yaxis.grid(True)
    
    plt.show()
    
    return ax



def add_crossec(tmat_name, *cross_sec_args, ax1=None):
        ax2 = ax1.twinx()
        fig_interaction_cross_section_of_speed.plot(tmat_name, *cross_sec_args, ax=ax2)
        # scale_ax2_to_match_ax1(ax1, ax2)
        mpl_axes_aligner.align.yaxes(ax1, 0, ax2, 0, 0.05)

def add_absorption(tmat_name, *absorb_args, ax1=None):
        ax2 = ax1.twinx()
        fig_absorption_of_speed.plot(tmat_name, *absorb_args, ax=ax2)
        # scale_ax2_to_match_ax1(ax1, ax2)
        mpl_axes_aligner.align.yaxes(ax1, 0, ax2, 0, 0.05)

def plot_momentum_obj(
    tmat_name,
    y_val,
    xi_list,
    cross_sec_args,
    ax = None
):
    
    if ax is None:
        fig, ax1 = plt.subplots()
    
    y_label = "$\Delta P_z$, $\\si{\kilogram\metre\per\second}$"
    name = f"figs/Momentum_transfer_OBJ_frame_{len(xi_list)}_crossec_{tmat_name}.png"
    
    linecolor = "darkred"
    ax1.plot(np.tanh(xi_list), y_val, linestyle="-", color=linecolor, zorder=3)

    ax1.set_xlabel("v/c")
    ax1.set_ylabel(y_label, color=linecolor)
    ax1.tick_params(axis="y", labelcolor=linecolor)
    
    ax1.set_xticks(np.arange(-0.75, 0.75+0.25, 0.25))

    ax1.set_zorder(1)  
    ax1.patch.set_visible(False)
    
    ax1.xaxis.grid(True, linestyle=':', color='black', linewidth=0.5)
    ax1.yaxis.grid(True, linestyle=':', color='black', linewidth=0.5)

    add_crossec(tmat_name, *cross_sec_args, ax1=ax1)
    
    fig.savefig(name, bbox_inches="tight")

def plot_energy_obj(
    tmat_name,
    y_val,
    xi_list,
    cross_sec_args,
    ax = None
):
    
    if ax is None:
        fig, ax1 = plt.subplots()
    
    y_label = "$\Delta H$, $\\si{mJ}$"
    name = f"figs/Energy_transfer_OBJ_frame_{len(xi_list)}_absorption_{tmat_name}.png"
    
    linecolor = "darkred"
    ax1.plot(np.tanh(xi_list), y_val*1000, linestyle="-", color=linecolor, zorder=3)

    ax1.set_xlabel("v/c")
    ax1.set_ylabel(y_label, color=linecolor)
    ax1.tick_params(axis="y", labelcolor=linecolor)
    
    ax1.set_xticks(np.arange(-0.75, 0.75+0.25, 0.25))

    ax1.set_zorder(1)  
    ax1.patch.set_visible(False)
    
    ax1.xaxis.grid(True, linestyle=':', color='black', linewidth=0.5)
    ax1.yaxis.grid(True, linestyle=':', color='black', linewidth=0.5)

    add_absorption(tmat_name, *cross_sec_args, ax1=ax1)
    
    fig.savefig(name, bbox_inches="tight")
    
    
def plot_momentum_lab(
    tmat_name,
    y_val,
    xi_list,
    ax1 = None
):
    
    if ax1 is None:
        fig, ax1 = plt.subplots()
    
    y_label = "$\Delta P_z$, $\\si{\kilogram\metre\per\second}$"
    name = f"figs/Momentum_transfer_LAB_frame_{len(xi_list)}_{tmat_name}.png"
    
    linecolor = "darkred"
    ax1.plot(np.tanh(xi_list), y_val, linestyle="-", color=linecolor, zorder=3)

    ax1.set_xlabel("v/c")
    ax1.set_ylabel(y_label, color=linecolor)
    ax1.tick_params(axis="y", labelcolor=linecolor)
    
    # ax1.set_xticks(np.arange(-0.75, 0.75+0.25, 0.25))

    ax1.set_zorder(1) 
    ax1.patch.set_visible(False)
    
    ax1.xaxis.grid(True, linestyle=':', color='black', linewidth=0.5)
    ax1.yaxis.grid(True, linestyle=':', color='black', linewidth=0.5)
    
    plt.savefig(name, bbox_inches="tight")
    
def plot_energy_lab(
    tmat_name,
    y_val,
    xi_list,
    ax = None
):
    
    if ax is None:
        fig, ax1 = plt.subplots()
    
    y_label = "$\Delta H$, $\\si{mJ}$"
    name = f"figs/Energy_transfer_LAB_frame_{len(xi_list)}_{tmat_name}.png"
    
    linecolor = "darkred"
    ax1.plot(np.tanh(xi_list), y_val*1000, linestyle="-", color=linecolor, zorder=3)

    ax1.set_xlabel("v/c")
    ax1.set_ylabel(y_label, color=linecolor)
    ax1.tick_params(axis="y", labelcolor=linecolor)
    
    ax1.set_xticks(np.arange(-0.75, 0.75+0.25, 0.25))

    ax1.set_zorder(1)  
    ax1.patch.set_visible(False)
    
    ax1.xaxis.grid(True, linestyle=':', color='black', linewidth=0.5)
    ax1.yaxis.grid(True, linestyle=':', color='black', linewidth=0.5)
    
    fig.savefig(name, bbox_inches="tight")


def get_transfer_comoving(tmat_name, incident_pw, jay_max, xi_list, load=False):
    xi_max = xi_list[-1]
    len_xi = len(xi_list)
    if load is True:
        xi_list = np.load(f'xi_list_{xi_max}_{len_xi}.npy')
        energy_diff_comoving = np.load(f'Energy_transfer_pulse_{xi_max}_{len_xi}.npy')
        momentum_diff_comoving = np.load(f'Momentum_transfer_pulse_{xi_max}_{len_xi}.npy')
    else:
        energy_diff_comoving, momentum_diff_comoving = (
            compute_transfer_pulse_comoving(tmat_name, incident_pw, jay_max, xi_list)
        )
    return energy_diff_comoving, momentum_diff_comoving

def plot(tmat_name, incident_pw, jay_max, xi_list, load=False):
    # Transfer in comoving frame: load or compute and save
    energy_diff_comoving, momentum_diff_comoving = get_transfer_comoving(tmat_name, incident_pw, jay_max, xi_list, load=load)
    
    # Transfer in lab frame
    energy_diff_lab, momentum_diff_lab = boost(energy_diff_comoving, momentum_diff_comoving, xi_list)

    aux_args = (
        incident_pw.info["central_wavelength"],
        xi_list[-1],
        len(xi_list),
        150, # was radius before
        jay_max,
    )

    plot_momentum_obj(tmat_name, momentum_diff_comoving, xi_list, aux_args)
    plot_energy_obj(tmat_name, energy_diff_comoving, xi_list, aux_args)
    plot_momentum_lab(tmat_name, momentum_diff_lab, xi_list)
    plot_energy_lab(tmat_name, energy_diff_lab, xi_list)
    



def aux_plots():
    # Object in comoving frame
    # thermal_gain_of_object_comoving, relativistic_energy_of_object_comoving, momentum_of_object_comoving = (
    #     compute_transfer_object_comoving(energy_transfer_pulse_comoving, momentum_transfer_pulse_comoving, radius)
    # )
    # plot_energies(xi_list, energy_transfer_pulse_comoving, thermal_gain_of_object_comoving, relativistic_energy_of_object_comoving)
    
    # Object in laboratory frame
    # thermal_gain_of_object_lab, relativistic_energy_of_object_lab, momentum_of_object_lab = (
    #     compute_transfer_object_lab(xi_list, energy_transfer_pulse_comoving, momentum_transfer_pulse_comoving, radius)
    # )
    # plot_energies(xi_list, energy_diff_pulse_lab, thermal_gain_of_object_lab, relativistic_energy_of_object_lab)
    
    return 0