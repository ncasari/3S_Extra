cimport PDSim.scroll.common_scroll_geo as common
from PDSim.scroll.common_scroll_geo cimport _dcoords_inv_dphi_int,_coords_inv_d, coords_inv, coords_inv_dtheta, coords_norm, get_compression_chamber_index
from PDSim.scroll.common_scroll_geo cimport Gr, dGr_dtheta, dGr_dphi, INVOLUTE_FI, INVOLUTE_FO, INVOLUTE_OI, INVOLUTE_OO
from PDSim.scroll.common_scroll_geo cimport fFx_p,fFy_p,fMO_p
cimport PDSim.scroll.symm_scroll_geo as symm_scroll_geo

from PDSim.scroll.common_scroll_geo import polycentroid, polyarea

from libc.math cimport sqrt,sin,cos,tan,atan2,acos,floor,M_PI as pi,pow,atan

import numpy as np
cimport numpy as np
import matplotlib.pyplot as plt

# Define python variables in this module for the integer
# constants for the control volumes
keyIsa = common.keyIsa
keyIs1 = common.keyIs1
keyIs2 = common.keyIs2
keyId1 = common.keyId1
keyId2 = common.keyId2
keyIdd = common.keyIdd
keyIddd = common.keyIddd
keyIc1_1 = common.keyIc1_1
keyIc1_2 = common.keyIc1_2
keyIc1_3 = common.keyIc1_3
keyIc1_4 = common.keyIc1_4
keyIc1_5 = common.keyIc1_5
keyIc2_1 = common.keyIc2_1
keyIc2_2 = common.keyIc2_2
keyIc2_3 = common.keyIc2_3
keyIc2_4 = common.keyIc2_4
keyIc2_5 = common.keyIc2_5

cpdef inline double Danfossmin2(double a, double b):
    return a if a<b else b

cpdef inline double Danfossmax2(double a, double b):
    return a if a>b else b

cdef bint overlap(double minv1, double maxv1, double minv2, double maxv2, double *min, double *max):
    """
    Returns True if the ranges overlap, with the pointers to the overlap range
    """
    cdef double e = 1e-14 #epsilon to deal with floating point accuracy
    cdef double _min = Danfossmax2(minv1, minv2)
    cdef double _max = Danfossmin2(maxv1, maxv2)

    if minv1-e <= _max <= maxv1+e and minv2-e <= _max <= maxv2+e and minv1-e <= _min <= maxv2+e and minv2-e <= _min <= maxv2+e:
        min[0] = _min
        max[0] = _max
        return True
    else:
        return False

cpdef tuple sortAnglesCCW(double t1, double t2):
    """
    Sort angles so that t2>t1 in a counter-clockwise sense
    idea from `StackOverflow <http://stackoverflow.com/questions/242404/sort-four-points-in-clockwise-order>`_
    more description: `SoftSurfer <http://softsurfer.com/Archive/algorithm_0101/algorithm_0101.htm>`_

    If the signed area of the triangle formed between the points on a unit circle with angles t1 and t2
    and the origin is positive, the angles are sorted counterclockwise. Otherwise, the angles
    are sorted in a counter-clockwise manner.  Here we want the angles to be sorted CCW, so
    if area is negative, swap angles
    
    Area obtained from the cross product of a vector from origin 
    to 1 and a vector to point 2, so use right hand rule to get 
    sign of cross product with unit length
    """

    if (cos(t1)*sin(t2)-cos(t2)*sin(t1)<0):
        # Swap angles
        t1,t2 = t2,t1
    while (t1 > t2):
        # Make t2 bigger than t1
        t2=t2 + 2*pi;
    return (t1,t2)

cdef class DanfossGeoVals(geoVals):
    """
    This is a custom class that add the geometric parameters that are
    required for the three-arc PMP
    """

    def __repr__(self):
        s = "(Danfoss-hybrid)" + geoVals.__repr__(self)
        for atr in ['xa_arc3','ya_arc3','ra_arc3','t1_arc3','t2_arc3']:
            s += atr+': '+str(getattr(self,atr))+'\n'
        return s

cpdef tuple phi_s1_sa(double theta, DanfossGeoVals geo):
    """
    # As in Bell, DOI: 10.1016/j.ijrefrig.2014.05.029 - typo in published work, C=-1, not C=1
    r_b,ro,phi_oie,phi_i0,phi_o0,phi_fie,theta,delta = symbols('r_b,ro,phi_oie,phi_i0,phi_o0,phi_fie,theta,delta')
    ro = r_b*pi - r_b*(phi_i0-phi_o0) #rb*pi-t
    om = phi_fie - theta + 3*pi/2
    # End point on fixed scroll
    xfi = r_b*cos(phi_fie)+r_b*(phi_fie-phi_i0)*sin(phi_fie)
    yfi = r_b*sin(phi_fie)-r_b*(phi_fie-phi_i0)*cos(phi_fie)
    # Other point on orbiting scroll
    phi_ssa = phi_fie - pi + delta
    xssa = -r_b*cos(phi_ssa)-r_b*(phi_ssa-phi_o0)*sin(phi_ssa)+ro*cos(om)
    yssa = -r_b*sin(phi_ssa)+r_b*(phi_ssa-phi_o0)*cos(phi_ssa)+ro*sin(om)
    d2 = (xfi-xssa)**2+(yfi-yssa)**2 # squared distance
    factor(expand_trig(simplify((diff(d2, delta)))),[sin(delta),cos(delta)])
    """
    cdef double A,B,C,S,u1,u2

    ro_rb = geo.ro/geo.rb
    A = (geo.phi_fi0-geo.phi_fie)+ro_rb*cos(theta)
    dA_dtheta = -ro_rb*sin(theta)
    B = 1+ro_rb*sin(theta)
    dB_dtheta = ro_rb*cos(theta)
    C = -1
    S = sqrt(A**2 + B**2 - C**2)
    u1 = (A - S)/(B-C)
    u2 = (A + S)/(B-C)

    delta1 = 2*atan(u1)
    delta2 = 2*atan(u2)

    du1_dA = (S-A)/(B-C)/S
    du2_dA = (S+A)/(B-C)/S

    du1_dB = ((B-C)*(-B/S)-(A-S))/(B-C)**2
    du2_dB = ((B-C)*(+B/S)-(A+S))/(B-C)**2

    ddelta1_dtheta = 2/(1+u1**2)*(du1_dA*dA_dtheta + du1_dB*dB_dtheta)
    ddelta2_dtheta = 2/(1+u2**2)*(du2_dA*dA_dtheta + du2_dB*dB_dtheta)

    if abs(delta1) < 1:
        return geo.phi_fie - pi + delta1, ddelta1_dtheta
    elif abs(delta2) < 1:
        return geo.phi_fie - pi + delta2, ddelta2_dtheta
    else:
        raise ValueError

cpdef tuple phi_s2_sa(double theta, DanfossGeoVals geo):
    """
    Formulate the problem like in Bell, DOI: 10.1016/j.ijrefrig.2014.05.029, see the appendix.
    Basically problem ends up being quite similar to the symmetric scroll problem with a different
    argument in the trig functions.  In the symmetric scroll case, cos(-geo.phi_fie+geo.phi_oie+theta)
    turns into cos(theta)

    Sympy code:
    r_b,ro,phi_oie,phi_i0,phi_o0,phi_fie,theta,delta = symbols('r_b,ro,phi_oie,phi_i0,phi_o0,phi_fie,theta,delta',real=True)
    om = phi_fie - theta + 3*pi/2
    ro = r_b*pi - r_b*(phi_i0-phi_o0) #rb*pi-t
    # End point on orbiting scroll
    xoi = -r_b*cos(phi_oie)-r_b*(phi_oie-phi_i0)*sin(phi_oie)+ro*cos(om)
    yoi = -r_b*sin(phi_oie)+r_b*(phi_oie-phi_i0)*cos(phi_oie)+ro*sin(om)
    # Other end point on fixed scroll
    phi_ssa = phi_oie-pi+delta
    xssa = r_b*cos(phi_ssa)+r_b*(phi_ssa-phi_o0)*sin(phi_ssa)
    yssa = r_b*sin(phi_ssa)-r_b*(phi_ssa-phi_o0)*cos(phi_ssa)
    d2 = (xoi-xssa)**2+(yoi-yssa)**2 # squared distance
    trigsimp(factor(expand_trig(simplify(diff(d2, delta))),[sin(delta),cos(delta)]))
    """
    cdef double A,B,C,S,u1,u2

    ro_rb = geo.ro/geo.rb
    A = (geo.phi_oi0-geo.phi_oie)+ro_rb*cos(-geo.phi_fie+geo.phi_oie+theta)
    dA_dtheta = -ro_rb*sin(-geo.phi_fie+geo.phi_oie+theta)
    B = 1+ro_rb*sin(-geo.phi_fie+geo.phi_oie+theta)
    dB_dtheta = ro_rb*cos(-geo.phi_fie+geo.phi_oie+theta)
    C = -1
    S = sqrt(A**2 + B**2 - C**2)
    u1 = (A - S)/(B-C)
    u2 = (A + S)/(B-C)

    delta1 = 2*atan(u1)
    delta2 = 2*atan(u2)

    du1_dA = (S-A)/(B-C)/S
    du2_dA = (S+A)/(B-C)/S

    du1_dB = ((B-C)*(-B/S)-(A-S))/(B-C)**2
    du2_dB = ((B-C)*(+B/S)-(A+S))/(B-C)**2

    ddelta1_dtheta = 2/(1+u1**2)*(du1_dA*dA_dtheta + du1_dB*dB_dtheta)
    ddelta2_dtheta = 2/(1+u2**2)*(du2_dA*dA_dtheta + du2_dB*dB_dtheta)

    if abs(delta1) < 1:
        return geo.phi_oie - pi + delta1, ddelta1_dtheta
    elif abs(delta2) < 1:
        return geo.phi_oie - pi + delta2, ddelta2_dtheta
    else:
        raise ValueError

cpdef CVInvolutes CVangles(double theta, DanfossGeoVals geo, int index):
    """
    Get the involute angles for the inner and outer involutes which form the given control volume

    Returns
    -------
    CVInvolutes : a :class:`PDSim.scroll.common_scroll_geo.CVInvolutes` class instance
    """
    cdef int Nc1, Nc2, alpha
    cdef CVInvolutes CV = CVInvolutes()
    cdef double theta_star
    cdef double phi_s_sa,  d_phi_s_sa_dtheta

    # The break angle where the s2 chamber just begins
    theta_break = geo.phi_fie - geo.phi_oie
    # Define the effective crank angle for path 2
    if theta_break < 1e-14:
        # This means the geometry is actually symmetric, because the break angle is actually equal to zero
        theta_star = theta
    elif theta > theta_break:
        # At the break angle, the effective angle is zero
        theta_star = theta - theta_break
    else:
        # At the break angle, the effective angle is a full rotation or 2*pi radians
        theta_star = theta + 2*pi - theta_break

    if index == common.keyIs1:
        phi_s_sa, d_phi_s_sa_dtheta = phi_s1_sa(theta, geo)
        CV.Outer.involute = common.INVOLUTE_FI
        CV.Outer.phi_0 = geo.phi_fi0
        CV.Outer.phi_max = geo.phi_fie
        CV.Outer.phi_min = geo.phi_fie-theta
        CV.Inner.involute = common.INVOLUTE_OO
        CV.Inner.phi_0 = geo.phi_oo0
        CV.Inner.phi_max = phi_s_sa
        CV.Inner.phi_min = geo.phi_fie-pi-theta
        CV.Inner.dphi_max_dtheta = d_phi_s_sa_dtheta
        CV.Outer.dphi_max_dtheta = 0
        CV.Inner.dphi_min_dtheta = CV.Outer.dphi_min_dtheta = -1
        CV.has_line_1 = True
        CV.has_line_2 = False

    elif common.keyIc1_1 <= index <= common.keyIc1_10:
        # index 1001 is c1.1, 1002 is c1.2, etc.
        # alpha is the index of the compression pocket,
        # alpha = 1 is the outermost chamber on the #1 path
        alpha = index - 1000

        if alpha > Nc(theta, geo, 1):
            raise KeyError("Requested alpha [{0:d}] is not possible; N_c_max is {1:d}".format(alpha, Nc(theta, geo, 1)))
        
        CV.Outer.involute = common.INVOLUTE_FI
        CV.Outer.phi_0 = geo.phi_fi0
        CV.Outer.phi_max = geo.phi_fie-theta-2*pi*(alpha-1)
        CV.Outer.phi_min = geo.phi_fie-theta-2*pi*alpha
        CV.Inner.involute = common.INVOLUTE_OO
        CV.Inner.phi_0 = geo.phi_oo0
        CV.Inner.phi_max = geo.phi_fie-pi-theta-2*pi*(alpha-1)
        CV.Inner.phi_min = geo.phi_fie-pi-theta-2*pi*alpha
        CV.Inner.dphi_max_dtheta = CV.Outer.dphi_max_dtheta = -1
        CV.Inner.dphi_min_dtheta = CV.Outer.dphi_min_dtheta = -1
        CV.has_line_1 = False
        CV.has_line_2 = False

    elif index == common.keyId1:
        Nc1 = Nc(theta, geo, 1)
        CV.Outer.involute = common.INVOLUTE_FI
        CV.Outer.phi_0 = geo.phi_fi0
        CV.Outer.phi_max = geo.phi_fie-theta-2.0*pi*Nc1
        CV.Outer.phi_min = geo.phi_oos + pi
        CV.Outer.dphi_max_dtheta = -1
        CV.Outer.dphi_min_dtheta = 0

        CV.Inner.involute = common.INVOLUTE_OO
        CV.Inner.phi_0 = geo.phi_oo0
        CV.Inner.phi_max = geo.phi_fie-pi-theta-2.0*pi*Nc1
        CV.Inner.phi_min = geo.phi_oos
        CV.Inner.dphi_max_dtheta = -1
        CV.Inner.dphi_min_dtheta = 0
        CV.has_line_1 = False
        CV.has_line_2 = True

    elif index == common.keyIs2:
        phi_s_sa, d_phi_s_sa_dtheta = phi_s2_sa(theta, geo)
        CV.Outer.involute = common.INVOLUTE_OI
        CV.Outer.phi_0 = geo.phi_oi0
        CV.Inner.involute = common.INVOLUTE_FO
        CV.Inner.phi_0 = geo.phi_fo0
        CV.Outer.phi_max = geo.phi_oie
        CV.Inner.phi_max = phi_s_sa #geo.phi_oie - pi # The point on the FO involute that is conjugate at theta_break
        CV.Outer.phi_min = geo.phi_oie - theta_star
        CV.Inner.phi_min = geo.phi_oie - pi - theta_star
        CV.Inner.dphi_max_dtheta = d_phi_s_sa_dtheta
        CV.Outer.dphi_max_dtheta = 0
        CV.Inner.dphi_min_dtheta = CV.Outer.dphi_min_dtheta = -1
        CV.has_line_1 = True
        CV.has_line_2 = False

    elif common.keyIc2_1 <= index <= common.keyIc2_10:
        # index 2001 is c2.1, 2002 is c2.2, etc.
        # alpha is the index of the compression pocket,
        # alpha = 1 is the outermost chamber on the #2 path
        alpha = index - 2000

        if alpha > Nc(theta, geo, 2):
            raise KeyError("Requested alpha [{0:d}] is not possible; N_c_max is {1:d}".format(alpha, Nc(theta, geo, 2)))
        
        CV.Outer.involute = common.INVOLUTE_OI
        CV.Outer.phi_0 = geo.phi_oi0
        CV.Inner.involute = common.INVOLUTE_FO
        CV.Inner.phi_0 = geo.phi_fo0

        CV.Outer.phi_max = geo.phi_oie - theta_star - 2*pi*(alpha-1)
        CV.Inner.phi_max = geo.phi_oie - pi - theta_star - 2*pi*(alpha-1)
        CV.Outer.phi_min = CV.Outer.phi_max - 2*pi
        CV.Inner.phi_min = CV.Inner.phi_max - 2*pi

        CV.Inner.dphi_max_dtheta = CV.Outer.dphi_max_dtheta = -1
        CV.Inner.dphi_min_dtheta = CV.Outer.dphi_min_dtheta = -1
        CV.has_line_1 = False
        CV.has_line_2 = False

    elif index == common.keyId2:
        Nc2 = Nc(theta, geo, 2)
        CV.Outer.involute = common.INVOLUTE_OI
        CV.Outer.phi_0 = geo.phi_oi0
        CV.Outer.phi_max = geo.phi_oie - theta_star - 2*pi*(Nc2)
        CV.Outer.phi_min = geo.phi_fos + pi
        CV.Outer.dphi_max_dtheta = -1
        CV.Outer.dphi_min_dtheta = 0

        CV.Inner.involute = common.INVOLUTE_FO
        CV.Inner.phi_0 = geo.phi_fo0
        CV.Inner.phi_max = geo.phi_oie - pi - theta_star - 2*pi*(Nc2)
        CV.Inner.phi_min = geo.phi_fos
        CV.Inner.dphi_max_dtheta = -1
        CV.Inner.dphi_min_dtheta = 0
        CV.has_line_1 = False
        CV.has_line_2 = True

    else:
        raise KeyError("index [{s:d}] not valid".format(s=index))

    return CV

cpdef double theta_d(DanfossGeoVals geo, int path) except *:
    """
    Discharge angle for the first path, s1, c1.x, d1 in the range 0,2*pi

    Condition for discharge angle is:

    geo.phi_ooe - theta - alpha*2*pi  = geo.phi_oos

    NOTE:
    For externally asymmetric scroll wrap, theta_d is the same for both paths.  You can
    see this clearly from looking at a plot of the scroll wraps and their control volumes
    """
    cdef CVInvolutes angles
    if path == 1 or path == 2:
        # Determine the "ending" angle on the orbiting scroll that is in
        # contact with the ending angle of the fixed scroll at theta=0
        N_c_max = floor((geo.phi_fie-(geo.phi_oos+pi))/(2*pi)) # At theta = 0
        return geo.phi_fie - pi - 2*pi*N_c_max - geo.phi_oos
    else:
        raise ValueError

cpdef int Nc(double theta, DanfossGeoVals geo, int path):
    """
    The number of pairs of compression chambers in existence at a given
    crank angle

    Arguments:
        theta : float
            The crank angle in radians.
        geo : DanfossGeoVals instance
        path : int
            The path index; 1 for s1, c1.x, d1

    Returns:
        Nc : int
            Number of pairs of compressions chambers

    NOTE:
    For externally asymmetric scroll wrap, theta_d is the same for both paths.  You can
    see this clearly from looking at a plot of the scroll wraps and their control volumes

    """
    cdef CVInvolutes angles
    if path == 1:
        angles = CVangles(theta, geo, common.keyIs1)
        return int(floor((angles.Inner.phi_min-geo.phi_oos)/(2*pi)))
    elif path == 2:
        angles = CVangles(theta, geo, common.keyIs2)
        return int(floor((angles.Inner.phi_min-geo.phi_fos)/(2*pi)))
    else:
        raise ValueError("path is invalid: "+str(path))

cpdef int getNc(double theta, DanfossGeoVals geo, int path):
    """
    A passthrough alias to be the same as the symmetric geometry module

    See Nc()
    """
    return Nc(theta, geo, path)


cpdef double Green_circle(double t, double r, double x0, double y0):
    """ Anti-derivative for arc segment on fixed scroll """
    return r*(r*t + x0*sin(t) - y0*cos(t))

cdef double Green_circle_orb(double t, double r, double x0, double y0, double r_o, double Theta):
    """ Anti-derivative for arc segment on orbiting scroll """
    return r*(r*t - r_o*sin(t - Theta) + x0*sin(t) - y0*cos(t))

cdef double dGreen_circle_orb_dtheta(double t, double r, double x0, double y0, double r_o, double Theta):
    """ Derivative of anti-derivative for arc segment on fixed scroll with respect to crank angle """
    return -r*r_o*cos(t-Theta)

cpdef VdVstruct SA(double theta, DanfossGeoVals geo):
    r = (2*pi*geo.rb-geo.t)/2.0

    cdef double xee,yee,xse,yse,xoie,yoie,xooe,yooe,xfie,yfie,xssa,yssa
    cdef double dx_2_dtheta, dy_2_dtheta, dx_3_dtheta, dy_3_dtheta, dxssa_dphi=0, dyssa_dphi=0, dxssa_dtheta=0, dyssa_dtheta=0, dxfie_dtheta=0, dyfie_dtheta=0

    phi_s_sa, d_phi_s_sa_dtheta = phi_s1_sa(theta, geo)

    common._coords_inv_d_int(geo.phi_oie + pi,geo,0.0,common.INVOLUTE_FI, &xee, &yee)
    common._coords_inv_d_int(geo.phi_oie - pi,geo,0.0,common.INVOLUTE_FO, &xse, &yse)
    common._coords_inv_d_int(geo.phi_oie,geo,theta,common.INVOLUTE_OI, &xoie, &yoie)
    common._coords_inv_d_int(geo.phi_ooe,geo,theta,common.INVOLUTE_OO, &xooe, &yooe)
    common._coords_inv_d_int(geo.phi_fie,geo,theta,common.INVOLUTE_FI, &xfie, &yfie)
    common._coords_inv_d_int(phi_s_sa,geo,theta,common.INVOLUTE_OO, &xssa, &yssa)
    x0, y0 = (xee+xse)/2, (yee+yse)/2

    x_1 = xse; y_1 = yse
    x_2 = xoie; y_2 = yoie
    x_3 = xooe; y_3 = yooe
    x_4 = xee; y_4 = yee

    beta = atan2(yee-y0,xee-x0)


    ## ------------------------ VOLUME -------------------------------
    A_wall = 0.5*(Gr(geo.phi_oie + pi, geo, theta, INVOLUTE_FI) - Gr(geo.phi_fie, geo, theta, INVOLUTE_FI))
    A_circle = 0.5*(Green_circle(beta+pi, r, x0, y0) - Green_circle(beta, r, x0, y0))
    A_line1 = 0.5*(x_1*y_2 - x_2*y_1)
    A_line2 = 0.5*(x_2*y_3 - x_3*y_2)
    A_oo = 0.5*(Gr(geo.phi_oie, geo, theta, INVOLUTE_OO) - Gr(phi_s_sa, geo, theta, INVOLUTE_OO))
    A_line3 = 0.5*(xssa*yfie - yssa*xfie)

    V = geo.h*(A_wall + A_circle + A_line1 + A_line2 + A_oo + A_line3)

    dx_1_dtheta, dy_1_dtheta = 0.0, 0.0
    common.coords_inv_dtheta(geo.phi_oie, geo, theta, INVOLUTE_OI, &dx_2_dtheta, &dy_2_dtheta)
    common.coords_inv_dtheta(geo.phi_ooe, geo, theta, INVOLUTE_OO, &dx_3_dtheta, &dy_3_dtheta)
    dx_4_dtheta, dy_4_dtheta = 0.0, 0.0

    dA_wall_dtheta = 0
    dA_circle_dtheta = 0.0
    dA_line1_dtheta = 0.5*(x_1*dy_2_dtheta + y_2*dx_1_dtheta - x_2*dy_1_dtheta - y_1*dx_2_dtheta)
    dA_line2_dtheta = 0.5*(x_2*dy_3_dtheta + y_3*dx_2_dtheta - x_3*dy_2_dtheta - y_2*dx_3_dtheta)
    dA_oo_dtheta = 0.5*(0
                      +dGr_dtheta(geo.phi_oie, geo, theta, INVOLUTE_OO)
                      -dGr_dphi(phi_s_sa, geo, theta, INVOLUTE_OO)*d_phi_s_sa_dtheta
                      -dGr_dtheta(phi_s_sa, geo, theta, INVOLUTE_OO)
                      )
    dxfie_dtheta = 0
    dyfie_dtheta = 0
    coords_inv_dtheta(phi_s_sa, geo, theta, INVOLUTE_OO, &dxssa_dtheta, &dyssa_dtheta)
    _dcoords_inv_dphi_int(phi_s_sa, geo, theta, INVOLUTE_OO, &dxssa_dphi, &dyssa_dphi)
    dxssa_dtheta += dxssa_dphi*d_phi_s_sa_dtheta
    dyssa_dtheta += dyssa_dphi*d_phi_s_sa_dtheta
    dA_line3_dtheta = 0.5*(xssa*dyfie_dtheta + yfie*dxssa_dtheta - xfie*dyssa_dtheta - yssa*dxfie_dtheta)

    dV = geo.h*(dA_wall_dtheta + dA_circle_dtheta + dA_line1_dtheta + dA_line2_dtheta + dA_oo_dtheta + dA_line3_dtheta)

    cdef VdVstruct VdV = VdVstruct.__new__(VdVstruct)
    VdV.V = V
    VdV.dV = dV
    return VdV

cpdef dict SA_forces(double theta, DanfossGeoVals geo):
    cdef CVInvolutes angles = CVangles(theta, geo, keyIs1)
    r = (2*pi*geo.rb-geo.t)/2.0

    xee,yee = coords_inv(geo.phi_fie,geo,0.0,'fi')
    xse,yse = coords_inv(geo.phi_foe-2*pi,geo,0.0,'fo')
    xoie,yoie = coords_inv(geo.phi_oie,geo,theta,'oi')
    xooe,yooe = coords_inv(geo.phi_ooe,geo,theta,'oo')
    x0, y0 = (xee+xse)/2, (yee+yse)/2
    
    # Calculate the force terms divided by the pressure acting on the outer edge of the orbiting scroll
    fx_p = fFx_p(geo.phi_ooe, geo, theta, angles.Inner.involute) - fFx_p(angles.Inner.phi_max, geo, theta, angles.Inner.involute)
    fy_p = fFy_p(geo.phi_ooe, geo, theta, angles.Inner.involute) - fFy_p(angles.Inner.phi_max, geo, theta, angles.Inner.involute)
    M_O_p = fMO_p(geo.phi_ooe, geo, theta, angles.Inner.involute) - fMO_p(angles.Inner.phi_max, geo, theta, angles.Inner.involute)

    # Add contribution from the end of the scroll, connecting the inner and outer involutes
    x1, y1 = coords_inv(geo.phi_ooe, geo, theta, 'oo')
    x2, y2 = coords_inv(geo.phi_oie, geo, theta, 'oi')
    xmid = (x1+x2)/2; ymid = (y1+y2)/2
    nx1, ny1 = coords_norm(geo.phi_ooe, geo, theta, 'oo')
    ny1, nx1 = -nx1[0], ny1[0]
    # Make sure you get the normal pointing towards the orbiting scroll!
    # The cross product of line going from inner to outer scroll wrap ends 
    # with normal should be negative
    if np.cross([x1-x2,y1-y2,0],[nx1, ny1, 0])[2] > 0:
        nx1 *= -1
        ny1 *= -1
    # Length is the thickness of scroll, height is scroll height
    A_line = geo.t*geo.h
    fx_p += A_line*nx1
    fy_p += A_line*ny1
    fO_p_line = [A_line*nx1, A_line*ny1, 0.0]
    THETA = geo.phi_fie-theta-pi/2
    r_line = [xmid - geo.ro*cos(THETA), ymid - geo.ro*sin(THETA), 0.0]
    cross = np.cross(r_line, fO_p_line)
    M_O_p += cross[2]

    cx = 0
    cy = 0

    return dict(cx = cx,
                cy = cy,
                fz_p = SA(theta,geo).V/geo.h,
                fx_p = fx_p,
                fy_p = fy_p,
                M_O_p = M_O_p
                )

cpdef VdVstruct DD(double theta, DanfossGeoVals geo):

    theta_m = geo.phi_fie - theta - pi/2

    Ainv_fi = 0.5*(Gr(geo.phi_oos + pi, geo, theta, common.INVOLUTE_FI)
                  -Gr(geo.phi_fis, geo, theta, common.INVOLUTE_FI))
    A_farc2 = 0.5*(Green_circle(geo.t1_arc2,geo.ra_arc2,geo.xa_arc2,geo.ya_arc2)
                  -Green_circle(geo.t2_arc2,geo.ra_arc2,geo.xa_arc2,geo.ya_arc2)
                  )
    x_1 = geo.xa_arc2 + geo.ra_arc2*cos(geo.t1_arc2)
    y_1 = geo.ya_arc2 + geo.ra_arc2*sin(geo.t1_arc2)
    x_2 = geo.xa_arc1 + geo.ra_arc1*cos(geo.t1_arc1)
    y_2 = geo.ya_arc1 + geo.ra_arc1*sin(geo.t1_arc1)
    A_line1 = 0.5*(x_1*y_2 - x_2*y_1)

    A_farc3 = 0.5*(Green_circle(geo.t2_arc3,geo.ra_arc3,geo.xa_arc3,geo.ya_arc3)
                  -Green_circle(geo.t1_arc3,geo.ra_arc3,geo.xa_arc3,geo.ya_arc3)
                  )

    A_farc1 = 0.5*(Green_circle(geo.t2_arc1,geo.ra_arc1,geo.xa_arc1,geo.ya_arc1)
                  -Green_circle(geo.t1_arc1,geo.ra_arc1,geo.xa_arc1,geo.ya_arc1)
                  )

    # Line from fixed scroll to orbiting scroll
    x1_fixorb, y1_fixorb = coords_inv(geo.phi_oos+pi, geo, theta, 'fi')
    x2_fixorb, y2_fixorb = coords_inv(geo.phi_oos, geo, theta, 'oo')
    A_line_fixorb = 0.5*(x1_fixorb*y2_fixorb - x2_fixorb*y1_fixorb)

    A_oarc2 = 0.5*(Green_circle_orb(geo.t1_arc2,geo.ra_arc2,geo.xa_arc2,geo.ya_arc2, geo.ro, theta_m)
                  -Green_circle_orb(geo.t2_arc2,geo.ra_arc2,geo.xa_arc2,geo.ya_arc2, geo.ro, theta_m)
                  )
    x_5 = -geo.xa_arc2 - geo.ra_arc2*cos(geo.t1_arc2) + geo.ro*cos(theta_m)
    y_5 = -geo.ya_arc2 - geo.ra_arc2*sin(geo.t1_arc2) + geo.ro*sin(theta_m)
    x_6 = -geo.xa_arc1 - geo.ra_arc1*cos(geo.t1_arc1) + geo.ro*cos(theta_m)
    y_6 = -geo.ya_arc1 - geo.ra_arc1*sin(geo.t1_arc1) + geo.ro*sin(theta_m)
    A_line3 = 0.5*(x_5*y_6 - x_6*y_5)

    A_oarc3 = 0.5*(Green_circle_orb(geo.t2_arc3,geo.ra_arc3,geo.xa_arc3,geo.ya_arc3, geo.ro, theta_m)
                  -Green_circle_orb(geo.t1_arc3,geo.ra_arc3,geo.xa_arc3,geo.ya_arc3, geo.ro, theta_m)
                 )

    A_oarc1 = 0.5*(Green_circle_orb(geo.t2_arc1,geo.ra_arc1,geo.xa_arc1,geo.ya_arc1, geo.ro, theta_m)
                  -Green_circle_orb(geo.t1_arc1,geo.ra_arc1,geo.xa_arc1,geo.ya_arc1, geo.ro, theta_m)
                 )

    # Line from orbiting scroll to fixed scroll
    x1_orbfix, y1_orbfix = coords_inv(geo.phi_fos+pi, geo, theta, 'oi')
    x2_orbfix, y2_orbfix = coords_inv(geo.phi_fos, geo, theta, 'fo')
    A_line_orbfix = 0.5*(x1_orbfix*y2_orbfix - x2_orbfix*y1_orbfix)

    Ainv_oi = 0.5*(Gr(geo.phi_fos + pi, geo, theta, common.INVOLUTE_OI)
                  -Gr(geo.phi_ois, geo, theta, common.INVOLUTE_OI))

    V = geo.h*(Ainv_fi + A_farc2 + A_line1 + A_farc3 + A_farc1 + A_line_fixorb
             + A_oarc2 + A_line3 + A_oarc3 + A_oarc1 + A_line_orbfix + Ainv_oi)

    ## ------------------------ DERIVATIVE -------------------------------
    dAinv_fi_dtheta = 0.5*(dGr_dtheta(geo.phi_oos+pi, geo, theta, common.INVOLUTE_FI)
                          -dGr_dtheta(geo.phi_fis, geo, theta, common.INVOLUTE_FI)
                       )
    dA_farc2_dtheta = 0.0
    dA_line1_dtheta = 0.0
    dA_farc1_dtheta = 0.0
    dA_farc3_dtheta = 0.0
    dx1_fixorb_dtheta, dy1_fixorb_dtheta, dx2_orbfix_dtheta, dy2_orbfix_dtheta = 0.0, 0.0, 0.0, 0.0
    dx2_fixorb_dtheta = dx_5_dtheta = dx_6_dtheta = dx1_orbfix_dtheta = -geo.ro*sin(theta_m)*(-1)
    dy2_fixorb_dtheta = dy_5_dtheta = dy_6_dtheta = dy1_orbfix_dtheta = geo.ro*cos(theta_m)*(-1)
    dA_line_fixorb_dtheta = 0.5*(x1_fixorb*dy2_fixorb_dtheta + y2_fixorb*dx1_fixorb_dtheta - x2_fixorb*dy1_fixorb_dtheta - y1_fixorb*dx2_fixorb_dtheta)
    dA_oarc2_dtheta = 0.5*(dGreen_circle_orb_dtheta(geo.t1_arc2,geo.ra_arc2,geo.xa_arc2,geo.ya_arc2, geo.ro, theta_m)
                          -dGreen_circle_orb_dtheta(geo.t2_arc2,geo.ra_arc2,geo.xa_arc2,geo.ya_arc2, geo.ro, theta_m)
                           )
    dA_line3_dtheta = 0.5*(x_5*dy_6_dtheta + y_6*dx_5_dtheta - x_6*dy_5_dtheta - y_5*dx_6_dtheta)
    dA_oarc3_dtheta = 0.5*(dGreen_circle_orb_dtheta(geo.t2_arc3,geo.ra_arc3,geo.xa_arc3,geo.ya_arc3, geo.ro, theta_m)
                          -dGreen_circle_orb_dtheta(geo.t1_arc3,geo.ra_arc3,geo.xa_arc3,geo.ya_arc3, geo.ro, theta_m)
                           )
    dA_oarc1_dtheta = 0.5*(dGreen_circle_orb_dtheta(geo.t2_arc1,geo.ra_arc1,geo.xa_arc1,geo.ya_arc1, geo.ro, theta_m)
                          -dGreen_circle_orb_dtheta(geo.t1_arc1,geo.ra_arc1,geo.xa_arc1,geo.ya_arc1, geo.ro, theta_m)
                           )
    dA_line_orbfix_dtheta = 0.5*(x1_orbfix*dy2_orbfix_dtheta + y2_orbfix*dx1_orbfix_dtheta - x2_orbfix*dy1_orbfix_dtheta - y1_orbfix*dx2_orbfix_dtheta)
    dAinv_oi_dtheta = 0.5*(dGr_dtheta(geo.phi_fos+pi, geo, theta, common.INVOLUTE_OI)
                          -dGr_dtheta(geo.phi_ois, geo, theta, common.INVOLUTE_OI)
                       )

    dV = geo.h*(dAinv_fi_dtheta + dA_farc2_dtheta + dA_line1_dtheta + dA_farc3_dtheta + dA_farc1_dtheta + dA_line_fixorb_dtheta
              + dA_oarc2_dtheta + dA_line3_dtheta + dA_oarc1_dtheta + dA_oarc3_dtheta + dA_line_orbfix_dtheta + dAinv_oi_dtheta)

    cdef VdVstruct VdV = VdVstruct.__new__(VdVstruct)
    VdV.V = V
    VdV.dV = dV
    return VdV

cpdef dict DD_forces(double theta, DanfossGeoVals geo):
    """
    Call the symmetric one, since for internal symmetry, the DD chamber is the same
    TODO: update when we have internal asymmetry
    """
    return symm_scroll_geo.DD_forces(theta, geo)

cpdef VdVstruct DDD(double theta, DanfossGeoVals geo):
    cdef VdVstruct Vd1 = common.VdV(theta, geo, CVangles(theta, geo, common.keyId1))
    cdef VdVstruct Vd2 = common.VdV(theta, geo, CVangles(theta, geo, common.keyId2))
    cdef VdVstruct Vdd = DD(theta, geo)

    cdef VdVstruct VdV = VdVstruct.__new__(VdVstruct)
    VdV.V = Vd1.V + Vd2.V + Vdd.V
    VdV.dV = Vd1.dV + Vd2.dV + Vdd.dV
    return VdV

cpdef dict DDD_forces(double theta, DanfossGeoVals geo):

    theta_m = geo.phi_fie - theta - pi/2

    exact_dict = {}
    anglesd1 = CVangles(theta, geo, common.keyId1)
    anglesd2 = CVangles(theta, geo, common.keyId2)
    Ad1 = common.VdV(theta, geo, anglesd1).V/geo.h
    Ad2 = common.VdV(theta, geo, anglesd2).V/geo.h
    _D1_forces = common.forces(theta,geo,anglesd1,Ad1)
    _D2_forces = common.forces(theta,geo,anglesd2,Ad2)
    _DD_forces = DD_forces(theta,geo)
    for key in _D1_forces:
        exact_dict[key] = _D1_forces[key]+_D2_forces[key]+_DD_forces[key]
    exact_dict['cx'] = geo.ro*cos(theta_m)/2.0
    exact_dict['cy'] = geo.ro*sin(theta_m)/2.0

    return exact_dict

cpdef VdVstruct VdV(int index, double theta, DanfossGeoVals geo):
    # Map the angle into [0, 2*pi], so long as it is not less than -20*pi
    cdef double theta_02PI = (theta + 20*pi) % (2*pi)
    cdef CVInvolutes angles = CVangles(theta_02PI, geo, index)
    return common.VdV(theta_02PI, geo, angles)

cdef int D_as_C_index(int index, double theta, DanfossGeoVals geo):
    """
    This function determines whether the conditions are such that
    the d1 chamber should be treated like the c1.max and/or the d2
    chamber should be treated as c2.max

    Returns
    -------
    iCV : int
        Index of the chamber that is to be used for the evaluation
    """
    cdef int before_discharge
    cdef double angle1, angle2, anglediff
    # If the cross-product of vectors from the origin to points
    # on the unit circle (cos(t1), sin(t1)) and (cos(t2), sin(t2))
    # are negative, then then the angles are sorted in a clockwise sense
    path = 1 if index == keyId1 else 2
    t2 = theta_d(geo, path)
    before_discharge = (cos(theta)*sin(t2)-sin(theta)*cos(t2)>0)
    angle1, angle2 = sortAnglesCCW(theta, t2)
    anglediff = angle2-angle1
    # Check that you are both to the "left" of the discharge angle in a rotaional
    # sense and just to the left of it 
    if before_discharge and anglediff < 1e-8:
        alpha = getNc(theta, geo, path)
        return get_compression_chamber_index(path, alpha)
    else:
        return index

cpdef dict forces(int index, double theta, DanfossGeoVals geo, CVInvolutes angles = None):
    """
    Calculate the force terms from the geometry
    """

    # Specialized treatments for SA, DD, DDD
    if index == common.keyIsa:
        return SA_forces(theta, geo)
    elif index == common.keyIdd:
        return DD_forces(theta, geo)
    elif index == common.keyIddd:
        return DDD_forces(theta, geo)
    else:
        # Generalized treatment for other chambers
        if angles is None:
            angles = CVangles(theta, geo, index)
        V = common.VdV(theta, geo, angles).V
        if V == 0.0 and (index == keyIs1 or index == keyIs2):
            V += 1e-8
        if index == keyId1 or index == keyId2:
            # Special case when just before the discharge angle, at which point
            # the force function of the c1.alpha or c2.alpha should be used 
            # instead of that of d1 or d2
            angles = CVangles(theta, geo, D_as_C_index(index, theta, geo))
            V = common.VdV(theta, geo, angles).V
            return common.forces(theta, geo, angles, V/geo.h)

        return common.forces(theta, geo, angles, V/geo.h)
    
cpdef CVcoords(CVkey, DanfossGeoVals geo, double theta, int Ninv=1000):
    """ 
    Return a tuple of numpy arrays for x,y coordinates for the curves which 
    determine the boundary of the control volume

    Returns
    -------
    x : numpy array
        X-coordinates of the outline of the control volume
    y : numpy array
        Y-coordinates of the outline of the control volume
    """

    cdef int Nc1 = Nc(theta, geo, 1), Nc2 = Nc(theta, geo, 2)

    if CVkey == 'sa':

        r = (2*pi*geo.rb-geo.t)/2.0

        xee, yee = coords_inv(geo.phi_oie + pi, geo, 0.0, 'fi')
        xse, yse = coords_inv(geo.phi_oie - pi, geo, 0.0, 'fo')
        xoie, yoie = coords_inv(geo.phi_oie, geo, theta, 'oi')
        xooe, yooe = coords_inv(geo.phi_ooe, geo, theta, 'oo')

        xwall, ywall = coords_inv(np.linspace(geo.phi_fie, geo.phi_oie + pi, 300), geo, 0.0, 'fi')
        xoo, yoo = coords_inv(np.linspace(geo.phi_oie, phi_s1_sa(theta, geo)[0], 300), geo, theta, 'oo')
        x0,y0 = (xee+xse)/2,(yee+yse)/2

        beta = atan2(yee-y0,xee-x0)
        t = np.linspace(beta,beta+pi,1000)
        x,y = x0+r*np.cos(t),y0+r*np.sin(t)

        return np.r_[x,xoie,xooe,xoo,xwall,x[0]],np.r_[y,yoie,yooe,yoo,ywall,y[0]]

    elif (CVkey == 's1' or CVkey == 's2' or CVkey == 'd1' or CVkey == 'd2'
          or CVkey.startswith('c1.') or CVkey.startswith('c2.')):

        if common.get_compressor_CV_index(CVkey) < 0:
            raise ValueError('CVkey [{key:s}] is invalid'.format(key = CVkey))

        # Get the bounding involute angles
        CV = CVangles(theta, geo, common.get_compressor_CV_index(CVkey))

        # Arrays of linearly spaced involute angles on each bounding involute
        phi_outer = np.linspace(CV.Outer.phi_min, CV.Outer.phi_max, Ninv)
        phi_inner = np.linspace(CV.Inner.phi_max, CV.Inner.phi_min, Ninv)
        
        # Key for the outer and inner involutes (one of 'fi', 'oo', 'oi', 'fi')
        key_outer = common.involute_index_to_key(CV.Outer.involute)
        key_inner = common.involute_index_to_key(CV.Inner.involute)

        x1, y1 = coords_inv(phi_outer, geo, theta, key_outer)
        x2, y2 = coords_inv(phi_inner, geo, theta, key_inner)

        #  Return the coordinates for the CV
        return np.r_[x1,x2],np.r_[y1,y2]

    elif CVkey == 'dd':
        theta_m = geo.phi_fie - theta - pi/2
        t = np.linspace(geo.t1_arc1,geo.t2_arc1,700)
        (x_farc1,y_farc1)=(
            geo.xa_arc1+geo.ra_arc1*np.cos(t),
            geo.ya_arc1+geo.ra_arc1*np.sin(t))
        (x_oarc1,y_oarc1)=(
           -geo.xa_arc1-geo.ra_arc1*np.cos(t)+geo.ro*cos(theta_m),
           -geo.ya_arc1-geo.ra_arc1*np.sin(t)+geo.ro*sin(theta_m))

        t=np.linspace(geo.t2_arc2,geo.t1_arc2,300)
        (x_farc2,y_farc2)=(
            geo.xa_arc2+geo.ra_arc2*np.cos(t),
            geo.ya_arc2+geo.ra_arc2*np.sin(t))
        (x_oarc2,y_oarc2)=(
           -geo.xa_arc2-geo.ra_arc2*np.cos(t)+geo.ro*cos(theta_m),
           -geo.ya_arc2-geo.ra_arc2*np.sin(t)+geo.ro*sin(theta_m))

        t = np.linspace(geo.t1_arc3,geo.t2_arc3,700)
        (x_farc3,y_farc3)=(
            geo.xa_arc3+geo.ra_arc3*np.cos(t),
            geo.ya_arc3+geo.ra_arc3*np.sin(t))
        (x_oarc3,y_oarc3)=(
           -geo.xa_arc3-geo.ra_arc3*np.cos(t)+geo.ro*cos(theta_m),
           -geo.ya_arc3-geo.ra_arc3*np.sin(t)+geo.ro*sin(theta_m))

        phi=np.linspace(geo.phi_fis,geo.phi_fos+pi,Ninv)
        (x_finv,y_finv)=coords_inv(phi,geo,theta,'fi')
        (x_oinv,y_oinv)=coords_inv(phi,geo,theta,'oi')

        x=np.r_[x_farc2,x_farc3,x_farc1,x_finv,x_oarc2,x_oarc1,x_oinv,x_farc2[0]]
        y=np.r_[y_farc2,y_farc3,y_farc1,y_finv,y_oarc2,y_oarc1,y_oinv,y_farc2[0]]
        return x,y
    else:
        raise KeyError('Could not match this CVkey: '+CVkey)
        
cpdef HTAnglesClass HT_angles(double theta, DanfossGeoVals geo, key):
    """
    Return the heat transfer bounding angles for the given control volume

    Parameters
    ----------
    theta : float
        Crank angle in the range [:math:`0,2\pi`]
    geo : DanfossGeoVals instance
    key : string
        Key for the control volume following the scroll compressor
        naming conventions

    Returns
    -------
    angles : HTAngles Class
        with the attributes:
        phi_1_i: maximum involute angle on the inner involute of the wrap
        that forms the outer wall of the CV

        phi_2_i: minimum involute angle on the inner involute of the wrap
        that forms the outer wall of the CV

        phi_1_o: maximum involute angle on the outer involute of the wrap
        that forms the inner wall of the CV

        phi_2_o: minimum involute angle on the outer involute of the wrap
        that forms the inner wall of the CV

    Notes
    -----
    The keys s1, c1.x, and d1 have as their outer wrap the fixed scroll

    The keys s2, c2.x, and d2 have as their outer wrap the orbiting scroll

    "Minimum", and "Maximum" refer to absolute values of the angles

    Raises
    ------
    If key is not valid, raises a KeyError
    """
    cdef HTAnglesClass angles = HTAnglesClass.__new__(HTAnglesClass)    
    cdef int index = common.get_compressor_CV_index(key)
    cdef CVInvolutes CV = CVangles(theta, geo, index)

    angles.phi_1_i = CV.Outer.phi_max
    angles.phi_2_i = CV.Outer.phi_min
    angles.phi_i0 = CV.Outer.phi_0

    angles.phi_1_o = CV.Inner.phi_max
    angles.phi_2_o = CV.Inner.phi_min
    angles.phi_o0 = CV.Inner.phi_0

    return angles

cdef _radial_leakage_angles(CVInvolutes CV_up, CVInvolutes CV_down, double *angle_min, double *angle_max):
    return overlap(CV_down.Inner.phi_min, CV_down.Inner.phi_max, CV_up.Outer.phi_min, CV_up.Outer.phi_max, angle_min, angle_max)

cpdef get_radial_leakage_angles(double theta, DanfossGeoVals geo, long index_up, long index_down):
    
    cdef double phi_min, phi_max

    # Bounding angles for each control volume to be considered here
    cdef CVInvolutes CV_up = CVangles(theta, geo, index_up)
    cdef CVInvolutes CV_down = CVangles(theta, geo, index_down)

    return _radial_leakage_angles(CV_up, CV_down, &phi_min, &phi_max)    
    
cpdef double radial_leakage_area(double theta, DanfossGeoVals geo, int index_up, int index_down, int location = common.UP) except *:
    """
    Notes
    -----
    Normally the higher pressure chamber is inside and the lower pressure chamber is outside.  But this might not be the case for some CV for theta<pi

    Also, you can tell which involute is outside by comparing radii of curvature.  The larger radius will always be towards the outside. (Right?)
    """
    cdef double phi_min, phi_max, phi_0
    
    # Bounding angles for each control volume to be considered here    
    cdef CVInvolutes CV_up = CVangles(theta, geo, index_up)
    cdef CVInvolutes CV_down = CVangles(theta, geo, index_down)
    
    phi_0 = (CV_up.Outer.phi_0 + CV_down.Inner.phi_0)/2

    if _radial_leakage_angles(CV_up, CV_down, &phi_min, &phi_max):
        return geo.delta_radial*geo.rb*((phi_max**2-phi_min**2)/2-phi_0*(phi_max-phi_min))
    else:    
        return 0.0