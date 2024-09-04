"""
@author: Maxim Vavilin maxim@vavilin.de
"""

import repscat as rs

def analyze_quantities(incident_pw, xi):
    E_rest = incident_pw.energy()
    print('E in pw: ', E_rest)
    print('Photons in pw: ', incident_pw.photons())
    Pz_rest = incident_pw.momentum_z()
    print('Pz in pw: ', Pz_rest)

    ### Quantities in boosted field
    incident_boosted_pw = incident_pw.boost(xi)

    print('Boosted E in pw: ', incident_boosted_pw.energy())
    print('Boosted Photons in pw: ', incident_boosted_pw.photons())
    print('Boosted P in pw: ', incident_boosted_pw.momentum_z())

    ### Boost quantities and compare (+)

    new_E, new_P = rs.boost_E_Pz(E_rest,Pz_rest,xi)
    print('New E = ', new_E)
    print('New P = ', new_P)