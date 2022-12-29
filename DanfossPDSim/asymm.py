from __future__ import print_function
# -*- coding: utf-8 -*-
from PDSim.misc.datatypes import arraym
from PDSim.core.containers import ControlVolume
from PDSim.scroll.core import Scroll
from PDSim.scroll import common_scroll_geo, symm_scroll_geo
from PDSim.flow import flow_models
from math import pi, cos
from CoolProp.CoolProp import PropsSI
from PDSim.flow.flow import FlowPath

try:
    from PDSim.plot.plots import debug_plots
except ImportError:
    debug_plots = None

from DanfossPDSim.core import DanfossScroll
from DanfossPDSim import asymm_scroll_geo
from DanfossPDSim.core import plotScrollSet

from scipy.optimize import newton
from CoolProp import State
import numpy as np
import copy
import itertools

TYPE_RADIAL = flow_models.TYPE_RADIAL
TYPE_FLANK = flow_models.TYPE_FLANK
TYPE_DISABLED = flow_models.TYPE_DISABLED


class AsymmetricScroll(DanfossScroll):

    def __init__(self):
        DanfossScroll.__init__(self)

        self.__before_pi1__ = False
        self.__before_pi2__ = False

        self.scroll_geo_module = asymm_scroll_geo

    def Nc_max(self):
        return [asymm_scroll_geo.Nc(0, self.geo, 1), asymm_scroll_geo.Nc(pi + 1e-10, self.geo, 2)]

    @property
    def Vdisp(self):
        """
        Displacement of machine in m^3.  The S1 pocket seals at 2*pi, the S2 
        pocket seals at theta_break radians and becomes a new S2 pocket
        """
        return self.V_s1(2 * pi)[0] + self.V_s2(self.theta_break() - 1e-14)[0]

    def theta_break(self):
        return self.geo.phi_fie - self.geo.phi_oie

    @property
    def theta_d(self):
        return asymm_scroll_geo.theta_d(self.geo, 1)

    @property
    def Vratio(self):
        Vratio1 = self.V_s1(2 * pi)[0] / self.V_d1(self.theta_d + 1e-12)[0]
        Vratio2 = (self.V_s2(self.theta_break() - 1e-14)[0] /
                    self.V_d2(self.theta_d + 1e-12)[0])
        return Vratio1, Vratio2

    def V_sa(self, theta):
        if self.geo.is_symmetric():
            V, dV = symm_scroll_geo.SA(theta, self.geo)
            return V + getattr(self, "SA_dead_volume", 0), dV
        else:    
            VdV = asymm_scroll_geo.SA(theta, self.geo)
            return VdV.V + getattr(self, "SA_dead_volume", 0), VdV.dV

    def V_s1(self, theta):
        VdV = asymm_scroll_geo.VdV(asymm_scroll_geo.keyIs1, theta, self.geo)
        return VdV.V + 1e-8, VdV.dV

    def V_s2(self, theta):
        VdV = asymm_scroll_geo.VdV(asymm_scroll_geo.keyIs2, theta, self.geo)
        return VdV.V + 1e-8, VdV.dV

    def V_c1(self, theta, alpha=1):
        angles = asymm_scroll_geo.CVangles(theta, self.geo, common_scroll_geo.get_compression_chamber_index(1, alpha))
        VdV = common_scroll_geo.VdV(theta, self.geo, angles)
        return VdV.V, VdV.dV

    def V_c2(self, theta, alpha=1):
        angles = asymm_scroll_geo.CVangles(theta, self.geo, common_scroll_geo.get_compression_chamber_index(2, alpha))
        VdV = common_scroll_geo.VdV(theta, self.geo, angles)
        return VdV.V, VdV.dV

    def V_d1(self, theta, full_output=False):
        VdV = asymm_scroll_geo.VdV(asymm_scroll_geo.keyId1, theta, self.geo)
        return VdV.V, VdV.dV

    def V_d2(self, theta):
        VdV = asymm_scroll_geo.VdV(asymm_scroll_geo.keyId2, theta, self.geo)
        return VdV.V, VdV.dV

    def V_dd(self, theta):
        VdV = asymm_scroll_geo.DD(theta, self.geo)
        return VdV.V, VdV.dV

    def V_ddd(self, theta, alpha=1, full_output=False):

        """
        Wrapper around the compiled code for DDD
        
        theta: angle in range [0,2*pi]
        alpha: index of compression chamber pair; 1 is for outermost set
        """
        if full_output == True:
            VdV = asymm_scroll_geo.DDD(theta, self.geo)
        else:
            VdV = asymm_scroll_geo.DDD(theta, self.geo)

        return VdV.V, VdV.dV

    def auto_add_CVs(self, inletState, outletState):
        """
        Adds all the control volumes for the asymmetric scroll compressor.
        
        Parameters
        ----------
        inletState
            A :class:`State <CoolProp.State.State>` instance for the inlet to the scroll set.  Can be approximate
        outletState
            A :class:`State <CoolProp.State.State>` instance for the outlet to the scroll set.  Can be approximate
            
        Notes
        -----
        Uses the indices of 
        
        ============= ===================================================================
        CV            Description
        ============= ===================================================================
        ``sa``        Suction Area
        ``s1``        Suction chamber on side 1
        ``s2``        Suction chamber on side 2
        ``d1``        Discharge chamber on side 1
        ``d2``        Discharge chamber on side 2
        ``dd``        Central discharge chamber
        ``ddd``       Merged discharge chamber
        ``c1.i``      The i-th compression chamber on side 1 (i=1 for outermost chamber)
        ``c2.i``      The i-th compression chamber on side 2 (i=1 for outermost chamber)
        ============= ===================================================================
        """

        # Add all the control volumes that are easy.  Suction area and suction chambera
        sa_becomes = ['sa', 's1']
        if self.geo.is_symmetric():
            sa_becomes.append('s2')
        self.add_CV(ControlVolume(
                key='sa', initialState=inletState.copy(),
                VdVFcn=self.V_sa,
                becomes=sa_becomes))
        self.add_CV(ControlVolume(key='s1', initialState=inletState.copy(),
                VdVFcn=self.V_s1,
                becomes='c1.1'))
        self.add_CV(ControlVolume(key='s2', initialState=inletState.copy(),
                VdVFcn=self.V_s2,
                becomes='c2.1' if self.geo.is_symmetric() else 's2'))

        # Discharge chambers are also easy.  Assume that you start with 'ddd' chamber merged.
        # No problem if this isn't true.
        self.add_CV(ControlVolume(key='d1',
                                  initialState=outletState.copy(),
                                  VdVFcn=self.V_d1,
                                  exists=False))
        self.add_CV(ControlVolume(key='d2',
                                  initialState=outletState.copy(),
                                  VdVFcn=self.V_d2,
                                  exists=False))
        self.add_CV(ControlVolume(key='dd',
                                  initialState=outletState.copy(),
                                  VdVFcn=self.V_dd,
                                  exists=False))
        self.add_CV(ControlVolume(key='ddd',
                                  initialState=outletState.copy(),
                                  VdVFcn=self.V_ddd,
                                  discharge_becomes='dd'))

        for path, VdVFcn in zip([1, 2], [self.V_c1, self.V_c2]):

            # The break angle where the s2 chamber just begins
            theta_break = self.geo.phi_fie - self.geo.phi_oie

            Nc_max = max(asymm_scroll_geo.Nc(0, self.geo, path),
                         asymm_scroll_geo.Nc(theta_break + 1e-10, self.geo, path))

            for alpha in range(1, Nc_max + 1):

                key = 'c' + str(path) + '.' + str(alpha)
                if alpha == 1:
                    # It is the outermost pair of compression chambers,
                    # start off with the inlet state
                    initState = inletState.copy()
                else:
                    # It is not the first CV, more involved analysis
                    # Assume isentropic compression from the inlet state at the end of the suction process
                    T1 = inletState.T  # [K]
                    s1 = inletState.s  # [kJ/kg/K]
                    rho1 = inletState.rho  # [kg/m3]
                    V1 = self.V_s1(2 * pi)[0]  # [m3]
                    angle_of_interest = 0.0 if path == 1 else (theta_break + 1e-10)
                    V2 = VdVFcn(angle_of_interest, alpha)[0]  # [m3]
                    # Mass is constant, so rho1*V1 = rho2*V2
                    rho2 = rho1 * V1 / V2
                    if rho2 < 0:
                        print('V1: {V1:g} V2: {V2:g} rho1: {rho1:g} rho2: {rho2:g}'.format(rho1=rho1, rho2=rho2, V1=V1,
                                                                                           V2=V2))
                    # Now don't know temperature or pressure, but you can assume
                    # it is isentropic to find the temperature.  Could in principle
                    # also be obtained from DS flash call in CoolProp
                    T2 = newton(lambda T: PropsSI('S', 'T', T, 'D', rho2, inletState.Fluid) / 1000.0 - s1, T1)
                    initState = State.State(inletState.Fluid, dict(T=T2, D=rho2)).copy()

                if path == 1:
                    # Conventional treatment, just like the symmetric compressor
                    if alpha < Nc_max:
                        # Does not change definition at discharge angle
                        disc_becomes = key
                        # It is not the innermost pair of chambers, becomes another
                        # set of compression chambers at the end of the rotation
                        becomes = 'c' + str(path) + '.' + str(alpha + 1)
                        exists = True
                    else:
                        # It is the innermost pair of chambers, becomes discharge
                        # chamber at the discharge angle
                        disc_becomes = 'd' + str(path)
                        becomes = key  # Not used
                        exists = True
                else:

                    if Nc_max == 1:
                        # It is the innermost pair of chambers, becomes discharge
                        # chamber at the discharge angle, but starts not existing
                        disc_becomes = 'd' + str(path)
                        becomes = key
                        exists = True
                    else:
                        if self.geo.is_symmetric():
                            if alpha < Nc_max:
                                # Does not change definition at discharge angle
                                disc_becomes = key
                                # It is not the innermost pair of chambers, becomes another
                                # set of compression chambers at the end of the rotation
                                becomes = "c" + str(path) + "." + str(alpha + 1)
                                exists = True
                            else:  # and alpha == Nc_max where Nc_max > 1
                                disc_becomes = 'd' + str(path)
                                becomes = key  # Not used
                                exists = True 
                        else:
                            if alpha < Nc_max:
                                # Does not change definition at discharge angle
                                disc_becomes = key
                                # It is not the innermost pair of chambers, continues as the same
                                # chamber at the end of the rotation
                                becomes = "c" + str(path) + "." + str(alpha)
                                exists = True
                            else:  # and alpha == Nc_max where Nc_max > 1
                                disc_becomes = 'd' + str(path)
                                becomes = key  # Not used, because it does not exist at the end of the rotation
                                exists = False # Does not exist at the beginning

                self.add_CV(ControlVolume(key=key,
                                          initialState=initState.copy(),
                                          VdVFcn=VdVFcn,
                                          VdVFcn_kwargs={'alpha': alpha},
                                          discharge_becomes=disc_becomes,
                                          becomes=becomes,
                                          exists=exists))

    def get_discharge_port_blockage_poly(self, theta):
        xdd, ydd = asymm_scroll_geo.CVcoords('dd', self.geo, theta)
        xd1, yd1 = asymm_scroll_geo.CVcoords('d1', self.geo, theta)
        Ncmax = asymm_scroll_geo.Nc(0, self.geo, 1)
        Nc = asymm_scroll_geo.Nc(theta, self.geo, 1)

        if Nc == Ncmax:
            xc1_N, yc1_N = asymm_scroll_geo.CVcoords('c1.' + str(Ncmax), self.geo, theta)
        else:
            xc1_N, yc1_N = None, None

        if Nc == Ncmax - 1:
            xc1_Nm1, yc1_Nm1 = asymm_scroll_geo.CVcoords('c1.' + str(Ncmax - 1), self.geo, theta)
        else:
            xc1_Nm1, yc1_Nm1 = None, None

        return dict(xdd=xdd, ydd=ydd,
                    xd1=xd1, yd1=yd1,
                    xc1_N=xc1_N, yc1_N=yc1_N,
                    xc1_Nm1=xc1_Nm1, yc1_Nm1=yc1_Nm1
                    )

    def step_callback(self, t, h, Itheta):
        """
        Here we test whether the control volumes need to be
        a) Merged
        b) Adjusted because you are at the discharge angle
        
        """

        # This gets called at every step, or partial step
        self.theta = t

        def angle_difference(angle1, angle2):
            # Due to the periodicity of angles, you need to handle the case where the
            # angles wrap around - suppose theta_d is 6.28 and you are at an angles of 0.1 rad
            # , the difference should be around 0.1, not -6.27
            # 
            # This brilliant method is from http://blog.lexique-du-net.com/index.php?post/Calculate-the-real-difference-between-two-angles-keeping-the-sign
            # and the comment of user tk
            return (angle1 - angle2 + pi) % (2 * pi) - pi

        def IsAtMerge(eps=0.001, eps_d1_higher=0.01, eps_dd_higher=0.00001):
            pressures = [self.CVs['d1'].State.p,
                         self.CVs['d2'].State.p,
                         self.CVs['dd'].State.p]
            p_max = max(pressures)
            p_min = min(pressures)
            if abs(p_min / p_max - 1) < eps_dd_higher:
                return True
            # For over compression cases, the derivatives don't tend to drive
            # the pressures together, and as a result you need to relax the 
            # convergence quite a bit
            elif angle_difference(t, asymm_scroll_geo.theta_d(self.geo, 1)) > 1.2 and abs(
                    p_min / p_max - 1) < eps_d1_higher:
                return True
            else:
                return False

        disable = False

        if t < self.theta_break() < t + h and self.__before_pi2__ == False:
            # print('stepping almost to theta_break(' + str(self.theta_break()) + ') radians')
            # A normal step, but step to just short of theta_break radians
            disable = True
            h = self.theta_break() - t - 1.1e-10
            self.__before_pi2__ = True
            self.__before_pi1__ = False

        elif self.__before_pi2__ == True:
            # print('just short of theta_break radians to pass theta = theta_break')

            # Make sure to evaluate the number of compression chambers 
            # on path #2 to the right of theta_break
            Nc2 = asymm_scroll_geo.Nc(self.theta + 2.2e-10, self.geo, 2)

            # c2.Nc2 created from c2.(Nc2-1)
            # Compression chambers in their values
            if Nc2 > 1:
                for I in range(Nc2, 1, -1):
                    self.CVs['c2.' + str(I)].State = self.CVs['c2.' + str(I - 1)].State.copy()
                    self.CVs['c2.' + str(I)].exists = True

            if Nc2 > 0:
                self.CVs['c2.1'].State = self.CVs['s2'].State.copy()
                self.CVs['c2.1'].exists = True

            # Reset s2 to inlet state of SA.  If you use its present state, the volume ratio over 
            # the step is ENORMOUS, and the temperature goes to a very(!) large value
            self.CVs['s2'].State = self.CVs['sa'].State.copy()

            self.__before_pi2__ = False
            self.__before_pi1__ = True

            self.update_existence()

            # Re-calculate the CV volumes
            V, dV = self.CVs.volumes(t + 2.2e-10)  # Make sure we evaluate at theta > theta_break
            # Update the matrices using the new CV definitions
            self.T[self.CVs.exists_indices, Itheta] = self.CVs.T
            self.p[self.CVs.exists_indices, Itheta] = self.CVs.p
            self.m[self.CVs.exists_indices, Itheta] = arraym(self.CVs.rho) * V
            self.rho[self.CVs.exists_indices, Itheta] = arraym(self.CVs.rho)

            # Adaptive makes steps of h/4 3h/8 12h/13 and h/2 and h
            # Make sure step does not hit any *right* at theta_d
            # That is why it is 2.2e-10 rather than 2.0e-10
            h = 2.2e-10
            disable = 'no_integrate'

        elif t < asymm_scroll_geo.theta_d(self.geo, 1) < t + h and self.__before_discharge2__ == False:
            # Take a step almost up to the discharge angle, but all the control volumes
            # are evaluated just like normal
            # print('stepping almost to theta_d(' + str(asymm_scroll_geo.theta_d(self.geo,1)) + ') radians; normal CV')
            disable = True
            h = asymm_scroll_geo.theta_d(self.geo, 1) - t - 1e-10
            self.__before_discharge2__ = True
            self.__before_dischargeswapped__ = False

        elif self.__before_discharge2__ == True:
            # At the discharge angle
            # print('Just to the left of the discharge angle; control volumes swapped')
            ########################
            #   Reassign chambers
            ########################

            # Find chambers with a discharge_becomes flag
            for key in self.CVs.exists_keys:
                if self.CVs[key].discharge_becomes in self.CVs.keys:
                    # Set the state of the "new" chamber to be the old chamber
                    oldCV = self.CVs[key]
                    if oldCV.exists == True:
                        newCV = self.CVs[oldCV.discharge_becomes]
                        newCV.State.update({'T': oldCV.State.T, 'D': oldCV.State.rho})
                        oldCV.exists = False
                        newCV.exists = True
                    else:
                        raise AttributeError("old CV doesn't exist")

            self.__before_discharge2__ = False
            self.__before_dischargeswapped__ = True

            self.update_existence()

            # Re-calculate the CV volumes
            V, dV = self.CVs.volumes(t + 2.2e-10)  # Make sure we evaluate at theta > theta_d
            # Update the matrices using the new CV definitions
            self.T[self.CVs.exists_indices, Itheta] = self.CVs.T
            self.p[self.CVs.exists_indices, Itheta] = self.CVs.p
            self.m[self.CVs.exists_indices, Itheta] = arraym(self.CVs.rho) * V
            self.rho[self.CVs.exists_indices, Itheta] = arraym(self.CVs.rho)

            # Adaptive makes steps of h/4 3h/8 12h/13 and h/2 and h
            # Make sure step does not hit any *right* at theta_d
            # That is why it is 2.2e-8 rather than 2.0e-8
            h = 2.2e-10
            disable = 'no_integrate'  # This means that no actual update will be made, simply a copy from the old CV to the new CV

        elif self.CVs['d1'].exists and IsAtMerge():

            # Build the volume vector using the old set of control volumes (pre-merge)
            V, dV = self.CVs.volumes(t)

            # print('merging')

            if self.__hasLiquid__ == False:

                # Density
                rhod1 = self.CVs['d1'].State.rho
                rhod2 = self.CVs['d2'].State.rho
                rhodd = self.CVs['dd'].State.rho
                # Pressure
                pd1 = self.CVs['d1'].State.p
                pd2 = self.CVs['d2'].State.p
                pdd = self.CVs['dd'].State.p
                # Internal energy
                ud1 = self.CVs['d1'].State.u
                ud2 = self.CVs['d2'].State.u
                udd = self.CVs['dd'].State.u
                # Internal energy
                Td1 = self.CVs['d1'].State.T
                Td2 = self.CVs['d2'].State.T
                Tdd = self.CVs['dd'].State.T
                # Volumes
                Vdict = dict(zip(self.CVs.exists_keys, V))
                Vd1 = Vdict['d1']
                try:
                    Vd2 = Vdict['d2']
                except:
                    Vd2 = 0
                Vdd = Vdict['dd']

                Vddd = Vd1 + Vd2 + Vdd
                m = rhod1 * Vd1 + rhod2 * Vd2 + rhodd * Vdd
                U_before = ud1 * rhod1 * Vd1 + ud2 * rhod2 * Vd2 + udd * rhodd * Vdd
                rhoddd = m / Vddd
                # Guess the mixed temperature as a volume-weighted average
                T = (Td1 * Vd1 + Td2 * Vd2 + Tdd * Vdd) / Vddd
                p = (pd1 * Vd1 + pd2 * Vd2 + pdd * Vdd) / Vddd
                # Must conserve mass and internal energy (instantaneous mixing process)
                Fluid = self.CVs['ddd'].State.Fluid
                T_u = newton(lambda x: PropsSI('U', 'T', x, 'D', rhoddd, Fluid) / 1000.0 - U_before / m, T)

                self.CVs['ddd'].State.update({'T': T_u, 'D': rhoddd})
                U_after = self.CVs['ddd'].State.u * self.CVs['ddd'].State.rho * Vddd

                DeltaU = m * (U_before - U_after)
                if abs(DeltaU) > 1e-5:
                    raise ValueError('Internal energy not sufficiently conserved in merging process')

                self.CVs['d1'].exists = False
                self.CVs['d2'].exists = False
                self.CVs['dd'].exists = False
                self.CVs['ddd'].exists = True

                self.update_existence()

                # Re-calculate the CV
                V, dV = self.CVs.volumes(t)
                self.T[self.CVs.exists_indices, Itheta] = self.CVs.T
                self.p[self.CVs.exists_indices, Itheta] = self.CVs.p
                self.m[self.CVs.exists_indices, Itheta] = arraym(self.CVs.rho) * V
                self.rho[self.CVs.exists_indices, Itheta] = arraym(self.CVs.rho)

            else:
                raise NotImplementedError('no flooding yet')
            disable = 'no_integrate'

        elif t > asymm_scroll_geo.theta_d(self.geo, 1) and self.__before_dischargeswapped__ == True:
            self.__before_dischargeswapped__ = False
            disable = False

        elif t > self.theta_break() and self.__before_pi1__ == True:
            # print('just past theta_break radians')
            self.__before_pi1__ = False
            disable = False
            h = 1e-6

        return disable, h

    def HT_angles(self, theta, geo, key):
        try:
            return asymm_scroll_geo.HT_angles(theta, self.geo, key)
        except KeyError:
            return None

    def DDD_to_S(self, FlowPath, flankFunc=None, path=1, **kwargs):
        if flankFunc is None:
            flankFunc = self.FlankLeakage
        # If there are any compression chambers, don't evaluate this flow
        # since the compression chambers "get in the way" of flow directly from 
        # ddd to s1 and s2
        if asymm_scroll_geo.Nc(self.theta, self.geo, path) > 0:
            return 0.0
        else:
            return flankFunc(FlowPath)

    def SA_S1(self, FlowPath, X_d=1.0, X_d_precompression=None):
        """
        A wrapper for the flow between the suction area and the S1 chamber
        
        Notes
        -----
        If geo.phi_ie_offset is greater than 0, the offset geometry will be 
        used to calculate the flow area.  Otherwise the conventional analysis 
        will be used.
        """

        if X_d_precompression is not None:
            dV = self.V_s1(self.theta)[1]
            if dV < 0:
                X_d = X_d_precompression

        # Max width is 2*r_o (see paper by Bell in IJR, but 1-cos(pi) gives a factor of 2),
        # so no need for additional factor of 2
        if self.geo.is_symmetric():
            FlowPath.A = symm_scroll_geo.Area_s_sa(self.theta, self.geo)
        else:
            FlowPath.A = self.geo.h * self.geo.ro * (1 - cos(self.theta))

        try:
            mdot = X_d * flow_models.IsentropicNozzle(
                    FlowPath.A, FlowPath.State_up, FlowPath.State_down)
            return mdot
        except ZeroDivisionError:
            return 0.0

    def SA_S2(self, FlowPath, X_d=1.0, X_d_precompression=None):
        """
        A wrapper for the flow between the suction area and the S2 chamber
        
        Notes
        -----
        If geo.phi_ie_offset is greater than 0, the offset geometry will be 
        used to calculate the flow area.  Otherwise the conventional analysis 
        will be used.
        """

        if X_d_precompression is not None:
            dV = self.V_s2(self.theta)[1]
            if dV < 0:
                X_d = X_d_precompression

        # Max width is 2*r_o (see paper by Bell in IJR, but 1-cos(pi) gives a factor of 2),
        # so no need for additional factor of 2
        if self.geo.is_symmetric():
            FlowPath.A = symm_scroll_geo.Area_s_sa(self.theta, self.geo)
        else:
            FlowPath.A = (self.geo.h * self.geo.ro * (1 - cos(self.theta - self.theta_break())))

        try:
            mdot = X_d * flow_models.IsentropicNozzle(
                FlowPath.A, FlowPath.State_up, FlowPath.State_down)
            return mdot
        except ZeroDivisionError:
            return 0.0

    def D_to_DD(self, FlowPath, X_d=1.0, porting_delay_degrees=0, **kwargs):
        """
        Area between d1 and dd or d2 and dd chambers
        """
        _theta_d = asymm_scroll_geo.theta_d(self.geo, 1)
        if self.geo.is_symmetric():
            FlowPath.A = symm_scroll_geo.Area_d_dd(self.theta, self.geo)
        else:
            DELTAtheta = self.geo.phi_oos + pi - self.geo.phi_fis

            # phi_d1dd is the involute angle that we consider to be the point on 
            # the inner involute that defines the splitting line
            # if theta = theta_d, phi_d1dd is phi_oos+pi
            # if theta = theta_d+DELTAtheta, phi_d1dd is phi_fis
            if 2 * pi - _theta_d < DELTAtheta:
                phi_d1dd = self.geo.phi_fis
            elif -1e-10 <= (self.theta - _theta_d) < DELTAtheta:
                phi_d1dd = -(self.theta - (_theta_d + DELTAtheta)) + self.geo.phi_fis
            else:
                phi_d1dd = self.geo.phi_fis

            to_02PI = lambda angle: (angle + 2 * np.pi) % (2 * np.pi)
            shifted_theta = to_02PI(self.theta - porting_delay_degrees / 180.0 * np.pi)
            x1, y1 = common_scroll_geo.coords_inv(phi_d1dd, self.geo, shifted_theta, 'fi')
            x2, y2 = common_scroll_geo.coords_inv(self.geo.phi_oos, self.geo, shifted_theta, 'oo')
            FlowPath.A = ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (0.5) * self.geo.h

        def angle_difference(angle1, angle2):
            # Due to the periodicity of angles, you need to handle the case where the
            # angles wrap around - suppose theta_d is 6.28 and you are at an angles of 0.1 rad
            # , the difference should be around 0.1, not -6.27
            # 
            # This brilliant method is from http://blog.lexique-du-net.com/index.php?post/Calculate-the-real-difference-between-two-angles-keeping-the-sign
            # and the comment of user tk
            return (angle1 - angle2 + pi) % (2 * pi) - pi

        # No flow if you haven't gotten to the porting delay angle
        if angle_difference(self.theta, _theta_d) < porting_delay_degrees / 180.0 * np.pi:
            return 0.0

        try:
            return flow_models.IsentropicNozzle(FlowPath.A,
                                                FlowPath.State_up,
                                                FlowPath.State_down) * X_d
        except ZeroDivisionError:
            return 0.0

    def scroll_involute_axial_force(self, theta, p_backpressure=0):
        """
        Calculate the axial force generated by the pressure distribution 
        along the top of the orbiting scroll wrap.  The force profile returned is the NET 
        force obtained by subtracting the back pressure from the applied force
        
        Pressure along inner and outer walls is considered to act over one-half 
        of the thickness of the scroll wrap.
        
        Notes
        -----
        The following assumptions are employed:
        
        1. Involute extended to the base circle to account for discharge region
        2. Half of the width of the scroll is assumed to see the upstream pressure and the other half sees the downstream pressure
        
        The length of an involute section can be given by
        
        .. math::
            
            s = r_b\\left(\\frac{\\phi_2^2-\\phi_1^2}{2}-\\phi_{0}(\\phi_2-\\phi_1)\\right)
        
        Returns
        -------
        F: numpy array
            Axial force matrix from the top of the scroll generated by each control volume [kN]
        """

        def curve_length(phi_max, phi_min, phi_0):
            """
            Return the curve length of a scroll wrap (see Eq. 4.256 from Ian Bell thesis)
            """
            return self.geo.rb*(0.5*(phi_max**2-phi_min**2)-phi_0*(phi_max-phi_min))

        _slice = range(len(theta))

        # Get the number of compression chambers in existence at each crank angle
        nC1 = np.array([asymm_scroll_geo.Nc(t, self.geo, 1) for t in theta])
        nC2 = np.array([asymm_scroll_geo.Nc(t, self.geo, 2) for t in theta])

        F = np.zeros_like(self.p)
        F = F[:, _slice]

        def get_s_CV(key):
            # Obtain the bounding angles for the chamber along the inner and outer involutes
            # bounding the cross-section of the CV
            angles = []
            for theta_ in theta:
                try:
                    # Calculate the angles for this value of theta (if possible)
                    angles.append(asymm_scroll_geo.CVangles(theta_, self.geo, common_scroll_geo.get_compressor_CV_index(key)))
                except:
                    # Put in a placeholder
                    angles.append(common_scroll_geo.CVInvolutes())

            # Determine the bounding angles for the length calculation based upon the 
            # involute of the chamber that is on the orbiting scroll
            invkey = common_scroll_geo.involute_index_to_key(angles[0].Inner.involute).decode('ascii')
            if invkey in ['oi', 'oo']:
                phi_max = np.array([angle.Inner.phi_max for angle in angles])
                phi_min = np.array([angle.Inner.phi_min for angle in angles])
                phi_0 = np.array([angle.Inner.phi_0 for angle in angles])
            else:
                phi_max = np.array([angle.Outer.phi_max for angle in angles])
                phi_min = np.array([angle.Outer.phi_min for angle in angles])
                phi_0 = np.array([angle.Outer.phi_0 for angle in angles])
            
            # Calculate the relevant curve length
            s = curve_length(phi_max=phi_max, phi_min=phi_min, phi_0=phi_0)
            return s

        standard_CVs = ['s1', 's2', 'd1', 'd2']
        c1keys = ['c1.' + str(I) for I in range(1, np.max(nC1) + 1)]
        c2keys = ['c2.' + str(I) for I in range(1, np.max(nC2) + 1)]
        CV_keys = standard_CVs + c1keys + c2keys

        # Calculate the lengths for the standard CV
        scoll = {key: get_s_CV(key) for key in CV_keys}

        # And the forces (again, all the ones that can be handled easily)
        for key in CV_keys:
            ICV = self.CVs.index(key)
            F[ICV, _slice] = scoll[key]*self.geo.t/2 * (self.p[ICV, _slice] - p_backpressure)
        
        # The sa chamber has bounding angles on the outer involute of the orbiting scroll that are governed
        # by the bounding angle of the s1 chamber (see for instance Fig 4.8 of Bell thesis)
        angles = [asymm_scroll_geo.CVangles(theta_, self.geo, common_scroll_geo.get_compressor_CV_index('s1')) for
                  theta_ in theta]
        invkey = common_scroll_geo.involute_index_to_key(angles[0].Inner.involute).decode('ascii')
        assert(invkey in ['oo'])
        ds_SA_o = curve_length(phi_max=self.geo.phi_ooe,
                               phi_min=np.array([angle.Inner.phi_max for angle in angles]),
                               phi_0=np.array([angle.Inner.phi_0 for angle in angles]))
        ICV = self.CVs.index('sa')
        F[ICV, :] = (ds_SA_o) * self.geo.t / 2 * (self.p[ICV, _slice] - p_backpressure)

        # Length for the DD chamber
        ds_DD_o = curve_length(phi_max = self.geo.phi_ois,
                               phi_min = self.geo.phi_oi0,
                               phi_0 = self.geo.phi_oi0)
        scoll['dd'] = ds_DD_o
        ICV = self.CVs.index('dd')
        F[ICV, :] = ds_DD_o*self.geo.t / 2 * (self.p[ICV, _slice] - p_backpressure)

        # The ddd chamber, obtained as sum of lengths from d1, d2, and dd chambers
        s = scoll['d1'] + scoll['d2'] + scoll['dd']
        ICV = self.CVs.index('ddd')
        F[ICV, :] = s*self.geo.t / 2 * (self.p[ICV, _slice] - p_backpressure)

        # Remove all the nan placeholders
        F[np.isnan(F)] = 0

        return F

    def pre_run(self, N):

        Scroll.pre_run(self)

        # A function closure that holds onto the key (tried initially to use lambda closure, but is late binding which didn't work.
        #  see http://docs.python-guide.org/en/latest/writing/gotchas/#late-binding-closures
        #
        class ForceFunction(object):
            def __init__(self, key):
                self.key = key

            def __call__(self, theta, geo):
                return asymm_scroll_geo.forces(self.key, theta, geo)

        # Connect asymmetric force functions
        self.CVs['sa'].ForceFcn = asymm_scroll_geo.SA_forces
        self.CVs['dd'].ForceFcn = asymm_scroll_geo.DD_forces
        self.CVs['ddd'].ForceFcn = asymm_scroll_geo.DDD_forces
        for CVstring, key in [['s1', asymm_scroll_geo.keyIs1], ['s2', asymm_scroll_geo.keyIs2],
                              ['d1', asymm_scroll_geo.keyId1], ['d2', asymm_scroll_geo.keyId2]]:
            self.CVs[CVstring].ForceFcn = ForceFunction(key)

        # Add the compression chambers too
        for path in [1, 2]:
            # Some chambers can come into existence after theta=pi, so check for there being more right after theta=pi too
            alpha_max = max(asymm_scroll_geo.Nc(0, self.geo, path), asymm_scroll_geo.Nc(np.pi + 1e-10, self.geo, path))
            for alpha in range(1, alpha_max + 1):
                CVstring = 'c{path:d}.{alpha:d}'.format(alpha=alpha, path=path)
                self.CVs[CVstring].ForceFcn = ForceFunction(1000 * path + alpha)

        # Test each function by calling it here, so if there is a catastrophic problem, we can trap it
        for CVkey in self.CVs.keys:
            if self.CVs[CVkey].ForceFcn is None:
                print('ForceFcn for CV[' + CVkey + '] is None. Why?')
            else:
                fcn_ok = False
                for t in np.linspace(0, 2*np.pi, 1000):
                    try:
                        self.CVs[CVkey].ForceFcn(t, self.geo)
                        fcn_ok = True
                        break
                    except:
                        pass
                if not fcn_ok:
                    print('ForceFcn for CV[' + CVkey + '] did not return a valid value at any angle. Why?')

        self.__before_discharge2__ = False
        self.__before_dischargeswapped__ = False

    def radial_leakage_angles(self, theta, key1Index, key2Index):
        """
        Get the bounding angles for a pair of control volumes at a give crank angle
        """
        return asymm_scroll_geo.get_radial_leakage_angles(theta, self.geo, key1Index, key2Index)

    def auto_add_leakage(self, 
                         flankFunc=None, 
                         radialFunc=None, 
                         radialFunc_kwargs={}, 
                         flankFunc_kwargs={}):
        """
        Add all the leakage terms for the compressor
        
        Parameters
        ----------
        flankFunc : function
            The function to be used for the flank leakage path
        flankFunc_kwargs : function
            Dictionary of terms to be passed to the flank leakage function
        radialFunc : function
            The function to be used for the radial leakage path
        radialFunc_kwargs : dict
            Dictionary of terms to be passed to the radial leakage function
            
        """

        if flankFunc is not None:
            # Add the flank leakages
            self.auto_add_flank_leakage(flankFunc, flankFunc_kwargs)

        if radialFunc is not None:
            # Add the radial leakages
            self.auto_add_radial_leakage(radialFunc, radialFunc_kwargs)

        # Add the Danfoss-specific flow terms
        DanfossScroll.add_Danfoss_flows(self)  # (DISABLED TEMPORARILY)

    def test_radial_at_angle(self, theta, scroll='orbiting', plot=False):

        # Walk along the scroll wrap, finding all the break angles, and calculating the length 
        # associated with each one, and then also which CV are to the "left" and "right"

        if scroll == 'orbiting':
            phi_0 = (self.geo.phi_oi0 + self.geo.phi_oo0)/2
            inner_break_angles = self.geo.phi_oie - theta - 2*np.pi*np.arange(0, 10)
            outer_break_angles = self.geo.phi_ooe - np.pi - theta - 2*np.pi*np.arange(0, 10)
            break_angles = inner_break_angles.tolist() + outer_break_angles.tolist()
            break_angles.append(asymm_scroll_geo.phi_s1_sa(theta, self.geo)[0])
            break_angles.append(self.geo.phi_ooe)
            break_angles.append(max(self.geo.phi_oos,self.geo.phi_ois))
            break_angles = np.array(break_angles)
            break_angles = np.array(sorted(break_angles[break_angles >= max(self.geo.phi_oos,self.geo.phi_ois)])[::-1])
        elif scroll == 'fixed':
            phi_0 = (self.geo.phi_fi0 + self.geo.phi_fo0)/2
            inner_break_angles = self.geo.phi_fie - theta - 2*np.pi*np.arange(0, 10)
            outer_break_angles = self.geo.phi_foe - np.pi - theta - 2*np.pi*np.arange(0, 10)
            break_angles = inner_break_angles.tolist() + outer_break_angles.tolist()
            break_angles.append(asymm_scroll_geo.phi_s1_sa(theta, self.geo)[0])
            break_angles.append(self.geo.phi_foe)
            break_angles.append(max(self.geo.phi_fos, self.geo.phi_fis))
            break_angles = np.array(break_angles)
            break_angles = np.array(sorted(break_angles[break_angles >= max(self.geo.phi_fos,self.geo.phi_fis)])[::-1])
        else:
            raise ValueError('For now...')

        def coords_midpoint(phi, scroll):
            rb = self.geo.rb
            ro = rb*(np.pi - self.geo.phi_fi0 + self.geo.phi_oo0)
            om = self.geo.phi_fie - theta + 3.0*pi/2.0
            if scroll == 'orbiting':
                x = -rb*np.cos(phi)-rb*(phi-phi_0)*np.sin(phi)+ro*np.cos(om)
                y = -rb*np.sin(phi)+rb*(phi-phi_0)*np.cos(phi)+ro*np.sin(om)
            elif scroll == 'fixed':
                x = rb*np.cos(phi)+rb*(phi-phi_0)*np.sin(phi)
                y = rb*np.sin(phi)-rb*(phi-phi_0)*np.cos(phi)
            return x, y

        def norms_midpoint(phi, scroll):
            rb = self.geo.rb
            ro = rb*(np.pi - self.geo.phi_fi0 + self.geo.phi_oo0)
            om = self.geo.phi_fie - theta + 3.0*pi/2.0
            if scroll == 'orbiting':
                nx = +np.sin(phi)
                ny = -np.cos(phi)
            elif scroll == 'fixed':
                nx = -np.sin(phi)
                ny = +np.cos(phi)
            return nx, ny

        def inpoly(testx, testy, xpoly, ypoly):
            """ http://www.heikkitoivonen.net/blog/2009/01/26/point-in-polygon-in-python/ """
            from matplotlib.path import Path
            p = Path(np.array((xpoly, ypoly)).T)
            return p.contains_point((testx, testy))

        def find_CV(x, y):
            """ Find the control volumes at a particular Cartesian coordinate """
            CVcandidates = ['sa','s1','s2','c1.1','c1.2','c2.1','c2.2','d1','d2']
            for CVkey in CVcandidates:
                try:
                    xpoly, ypoly = asymm_scroll_geo.CVcoords(CVkey, self.geo, theta)
                    if inpoly(x, y, xpoly, ypoly):
                        return CVkey
                except KeyError:
                    pass
            return None
        
        test_points = []
        L_models = 0
        for i in range(len(break_angles)-1):
            phi_max = break_angles[i]
            phi_min = break_angles[i+1]
            L = self.geo.rb*((phi_max**2-phi_min**2)/2 - phi_0*(phi_max-phi_min))
            # point along midline of scroll at average involute angle
            xm, ym = coords_midpoint((phi_max+phi_min)/2, scroll)

            # two points just past the scroll itself
            nxm, nym = norms_midpoint((phi_max+phi_min)/2, scroll)
            tt = np.array([-1.0, 1.0])*1.0001/2*self.geo.t
            xps, yps = xm + nxm*tt, ym + nym*tt

            these_test_points = []
            for xp, yp in zip(xps, yps):
                these_test_points.append((xp**2 + yp**2, xp, yp))
            test_points += these_test_points

            # Sort to get upstream and downstream keys as first and second
            dummy, xps, yps = zip(*sorted(these_test_points))
            key1, key2 = [find_CV(xp, yp) for xp, yp in zip(xps, yps)]
            try:
                L_model = self.radial_leakage_area(theta, key1, key2)/self.geo.delta_radial

                L_model_core = -1
                try:
                    ikey1 = common_scroll_geo.get_compressor_CV_index(key1)
                    ikey2 = common_scroll_geo.get_compressor_CV_index(key2)
                    L_model_core = Scroll.radial_leakage_area(self, theta, ikey1, ikey2)/self.geo.delta_radial
                except:
                    pass
                print(L, L_model, L_model_core, key1, key2)
                L_models += L_model
            except BaseException as BE:
                print(BE)
                print(L, 'but no model', key1, key2)
                L_models += L

        phi_max = break_angles[0]
        phi_min = break_angles[-1]
        overall_length = self.geo.rb*((phi_max**2-phi_min**2)/2 - phi_0*(phi_max-phi_min))
        print(overall_length,'m is the overall arc length for one scroll at theta:', theta)
        print(L_models,'m is the sum of model lengths for one scroll at theta:', theta)

        if plot:
            
            plotScrollSet(theta, self.geo, show=False, shaveOn=False)
            import matplotlib.pyplot as plt
            
            # Show the locations of each break point along the centerline
            x,y = coords_midpoint(break_angles,scroll)
            plt.plot(x, y, '.', ms=8)

            for test_point in test_points:
                dist,x,y = test_point
                plt.plot(x, y, 's', ms=8)

            plt.fill(*asymm_scroll_geo.CVcoords('d1', self.geo, theta), color='yellow')
            plt.fill(*asymm_scroll_geo.CVcoords('d2', self.geo, theta), color='blue')
            plt.show()

    def auto_add_radial_leakage(self, radialFunc, radialFunc_kwargs={}, plot=False):
        """
        A function to add all the radial leakage terms.  Here we add 
        all possible permutations of control volumes communicating
        with each other.  The overlap of their involute angles will
        determine whether they actually have some amount of radial leakage
        between chambers
        
        Parameters
        ----------
        radialFunc : function
            The function that will be called for each radial leakage
        radialFunc_kwargs : function
            Dictionary of terms to be passed to the radial leakage function
        """

        SACVs = ["s1", "s2"]
        CVs = ["s1", "s2"]
        for path in [1, 2]:
            Nc_max = max(
                asymm_scroll_geo.Nc(0, self.geo, path),
                asymm_scroll_geo.Nc(self.theta_break() + 1e-10, self.geo, 
                path),
            )
            for alpha in range(1, Nc_max + 1):
                keyc = 'c' + str(path) + '.' + str(alpha)
                CVs.append(keyc)
                if alpha == 1:
                    SACVs.append(keyc)
        CVs += ["d1", "d2"]

        # Flip the direction of the list of control volumes so that the inner
        # one is always the "upstream" part of the flow path
        CVs = CVs[::-1]

        thetavec = np.linspace(0, 2*np.pi, 1000)
        Asum = 0
        for other_key in SACVs:
            # !Special treatment for the sa chamber! 
            # Special treatment is needed because the sa chamber, unlike other 
            # chambers, has two involute portions (forming the "inner" boundary of the chamber)
            # but the outer boundary of the CV is not involute. This cannot be handled by 
            # the automatic calculations
            other_index = common_scroll_geo.get_compressor_CV_index(other_key)
            radialFuncSA_kwargs = radialFunc_kwargs.copy()
            radialFuncSA_kwargs['other_index'] = other_index
            A = []
            for theta_ in thetavec:
                try:
                    A.append(self.radial_leakage_sa_area(theta_, other_index))
                except KeyError:
                    A.append(0.0)
            # if np.sum(A) > 0:
            #     print(other_key, 'sa', np.sum(A)/self.geo.delta_radial)
            Asum += np.array(A)
            self.add_flow(
                FlowPath(
                    key1=other_key,
                    key2='sa',
                    MdotFcn=self.RadialLeakageSA,
                    MdotFcn_kwargs=radialFuncSA_kwargs,
            ))
        
        for (key1, key2) in itertools.combinations(CVs, 2):
            # Swap if the alpha=2 path is second
            if key1.endswith('.1') and key2.endswith('.2'):
                key2, key1 = key1, key2

            # Instantiate the flow path
            FP = FlowPath(
                key1=key1,
                key2=key2,
                MdotFcn=radialFunc,
                MdotFcn_kwargs=radialFunc_kwargs,
            )
            # Check whether at any angle there is some flow area. If so,
            # add the flow path
            A = []
            for theta_ in thetavec:
                try:
                    A.append(self.radial_leakage_area(theta_, FP.key1, FP.key2))
                except KeyError:
                    A.append(0.0)
            # if np.sum(A) > 0:
            #     print(FP.key1, FP.key2, np.sum(A)/self.geo.delta_radial)
            Asum += np.array(A)

            if np.mean(A) > 1e-12:
                # plt.plot(thetavec, A)
                # plt.title(' & '.join((key1, key2)))
                # plt.axvline(self.theta_d)
                # plt.show()
                # print(key1, key2)
                self.add_flow(FP)
        if plot:
            import matplotlib.pyplot as plt            
            plt.plot(thetavec, Asum/self.geo.delta_radial)
            plt.gca().set(xlabel=r'$\theta$', ylabel=r'$(\sum A_{\rm radial})/\delta_{\rm radial}$')
            plt.show()

    def auto_add_flank_leakage(self, flankFunc, flankFunc_kwargs={}):
        """
        A function to add all the flank leakage terms
        
        Parameters
        ----------
        flankFunc : function
            The function that will be called for each flank leakage
        flankFunc_kwargs : dictionary
            Dictionary of terms to be passed to the flank leakage function
        """

        #         import traceback, sys
        #         traceback.print_stack()

        # Always a s1-c1 leakage and s2-c2 leakage        
        self.add_flow(FlowPath(key1='s1', key2='c1.1', MdotFcn=flankFunc, MdotFcn_kwargs=flankFunc_kwargs))
        self.add_flow(FlowPath(key1='s2', key2='c2.1', MdotFcn=flankFunc, MdotFcn_kwargs=flankFunc_kwargs))

        for path in [1, 2]:

            Nc_max = max(asymm_scroll_geo.Nc(0, self.geo, path),
                         asymm_scroll_geo.Nc(self.theta_break() + 1e-10, self.geo, path))

            # Only add the DDD-S1 and DDD-S2 flow path if there is one set of
            # compression chambers.
            # if Nc_max == 1:
            #     flankFunc_kwargs_copy = copy.deepcopy(flankFunc_kwargs)
            #     flankFunc_kwargs_copy['path'] = path
            #     self.add_flow(FlowPath(key1 = 's' + str(path), key2 = 'ddd',MdotFcn = self.DDD_to_S, MdotFcn_kwargs = flankFunc_kwargs_copy))

            for alpha in range(1, Nc_max + 1):
                keyc = 'c' + str(path) + '.' + str(alpha)

                if alpha <= Nc_max - 1:
                    # Leakage between compression chambers along a path
                    self.add_flow(FlowPath(key1=keyc,
                                           key2='c' + str(path) + '.' + str(alpha + 1),
                                           MdotFcn=flankFunc,
                                           MdotFcn_kwargs=flankFunc_kwargs))

                elif alpha == Nc_max:
                    # Leakage between the discharge region and the innermost chamber
                    self.add_flow(FlowPath(key1=keyc, key2='ddd', MdotFcn=flankFunc, MdotFcn_kwargs=flankFunc_kwargs))

                flankFunc_kwargs_copy = copy.deepcopy(flankFunc_kwargs)
                # Update the flag so that this term will only be evaluated when the number of pairs of 
                # compression chambers in existence will be equal to
                flankFunc_kwargs_copy['Ncv_check'] = Nc_max - 1
                flankFunc_kwargs_copy['path'] = path

                if alpha == Nc_max - 1:
                    # Leakage between the discharge region and the next-most inner chamber when the innermost chambers
                    # have been swallowed into the discharge region
                    self.add_flow(
                        FlowPath(key1=keyc, key2='ddd', MdotFcn=flankFunc, MdotFcn_kwargs=flankFunc_kwargs_copy))
                    self.add_flow(FlowPath(key1=keyc, key2='d' + str(path), MdotFcn=flankFunc,
                                           MdotFcn_kwargs=flankFunc_kwargs_copy))

    def FlankLeakage(self, FP, Ncv_check=-1, path=-1):
        """
        Calculate the flank leakage flow rate
        
        Parameters
        ----------
        FP : FlowPath
        Ncv_check : int,optional
            If ``Ncv_check`` is greater than -1, this flow path will only be evaluated
            when the number of pairs of compression chambers is equal to this value
        """
        _evaluate = False
        t = -1.0  # Default (not-provided) value

        if Ncv_check > -1:
            Nc = asymm_scroll_geo.Nc(self.theta, self.geo, path)
            if Ncv_check == Nc:
                _evaluate = True
            else:
                _evaluate = False
        else:
            _evaluate = True

        if _evaluate:
            # Calculate the area
            FP.A = self.geo.h * self.geo.delta_flank
            return flow_models.FrictionCorrectedIsentropicNozzle(
                FP.A,
                FP.State_up,
                FP.State_down,
                self.geo.delta_flank,
                TYPE_FLANK,
                t,
                self.geo.ro
            )
        else:
            return 0.0

    def RadialLeakage(self, FP, t=-1):
        """
        Calculate the radial leakage flow rate
        
        Parameters
        ----------
        FP : FlowPath
            
        t : float,optional
            The thickness of the wrap to be used.  If not provided, the scroll
            wrap thickness
        """

        # Calculate the area - arc length of the upstream part of the flow path
        FP.A = self.radial_leakage_area(self.theta, FP.key1, FP.key2)

        if FP.A == 0.0:
            return 0.0

        # Allow you to change the length for the radial leakage path
        # by passing in a length other than the thickness of the scroll wrap
        # 
        if t <= 0:
            t = self.geo.t

        return flow_models.FrictionCorrectedIsentropicNozzle(FP.A,
                                                             FP.State_up,
                                                             FP.State_down, 
                                                             self.geo.delta_radial,
                                                             TYPE_RADIAL,
                                                             t)

    def radial_leakage_sa_area(self, theta, other_index):
        """ Calculate the area based on the other control volume """
        angles = asymm_scroll_geo.CVangles(theta, self.geo, other_index)
        phi_0 = (angles.Outer.phi_0 + angles.Inner.phi_0)/2
        phi_max = angles.Outer.phi_max
        phi_min = angles.Outer.phi_min
        if other_index == common_scroll_geo.get_compressor_CV_index('c1.1'):
            phi_min = asymm_scroll_geo.phi_s1_sa(theta, self.geo)[0]
        elif other_index == common_scroll_geo.get_compressor_CV_index('c2.1'):
            phi_min = asymm_scroll_geo.phi_s2_sa(theta, self.geo)[0]
        A = self.geo.delta_radial*self.geo.rb*((phi_max**2-phi_min**2)/2-phi_0*(phi_max-phi_min))
        if A < 0:
            return 0.0
        else:
            return A

    def radial_leakage_area(self, theta, key1, key2, *, angles=False):
        ikey1 = common_scroll_geo.get_compressor_CV_index(key1)
        if key2 == 'sa':
            return self.radial_leakage_sa_area(theta, ikey1)
        ikey2 = common_scroll_geo.get_compressor_CV_index(key2)
        CV_up = asymm_scroll_geo.CVangles(theta, self.geo, ikey1)
        CV_down = asymm_scroll_geo.CVangles(theta, self.geo, ikey2)

        # If they have the same inner and outer involutes, cannot have a flow path
        # between them
        if CV_up.Outer.involute == CV_down.Outer.involute:
            return 0.0
        if angles:
            return CV_up, CV_down
        return asymm_scroll_geo.radial_leakage_area(theta, self.geo, ikey1, ikey2)

    def RadialLeakageSA(self, FP, t=-1, other_index=None):
        """
        Calculate the radial leakage flow rate between a chamber and
        the SA chamber.  The flow angles will be determined exclusively
        by the other chamber
        
        Parameters
        ----------
        FP : FlowPath
            The instantiated FlowPath instance
        other_index: int
            The index of the other chamber communicating with the SA chamber
        t : float,optional
            The thickness of the wrap to be used.  If not provided, the scroll
            wrap thickness

        """
        assert(other_index is not None)

        # Calculate the area based on the other control volume
        FP.A = self.radial_leakage_sa_area(self.theta, other_index)

        if FP.A == 0.0:
            return 0.0

        # Allow you to change the length for the radial leakage path
        # by passing in a length other than the thickness of the scroll wrap
        # 
        if t <= 0:
            t = self.geo.t

        return flow_models.FrictionCorrectedIsentropicNozzle(
            FP.A, FP.State_up, FP.State_down, self.geo.delta_radial,
            TYPE_RADIAL, t)

    def tip_seal_bypass(self, FP, A=-1, Ncv_check=-1, path=-1):
        """
        A flow function for the bypass around the tip seal
        
        Parameters
        ----------
        A : float, optional
            If provided, use the keyword argument for the area, otherwise just use
            the value from the FlowPath instance
            
        Ncv_check : int, optional
            If provided, the mass flow rate is only evaluated when the number of
            pairs of compression chambers in existence is equal to ``Ncv_check``.
            This is useful for the next-innermost pair of chambers, so that they
            
        """

        if Ncv_check > -1:
            if Ncv_check == asymm_scroll_geo.Nc(0, self.geo, path):
                _evaluate = True
            else:
                _evaluate = False
        else:
            _evaluate = True

        if _evaluate:
            if A < 0:
                raise ValueError('Area [{A:g}] to tip_seal_bypass is negative'.format(A=A))

            # Set the area using the keyword argument area 
            FP.A = A

            return flow_models.IsentropicNozzle(FP.A,
                                                FP.State_up,
                                                FP.State_down)
        else:
            return 0.0

    def build_pressure_profile(self):
        """
        Build the pressure profile, tracking along s1,c1.x,d1,ddd and
        s2,c2.x,d2,ddd and store them in the variables summary.theta1_profile,
        summary.p1_profile, summary.theta2_profile, summary.p2_profile
        
        Each path is handled separately
        
        """

        # Calculate along one path to track one set of pockets through the whole process
        theta = self.t

        Nc_max1, Nc_max2 = self.Nc_max()

        #####  ********* PATH #1 *****************
        #####  ********* PATH #1 *****************
        #####  ********* PATH #1 *****************

        theta = self.t

        # s1 path is treated normally
        p1 = self.p[self.CVs.index('s1')].copy()

        # Outer compression chambers
        for alpha in range(1, Nc_max1):
            theta = np.append(theta, self.t + 2 * pi * alpha)
            p1 = np.append(p1, self.p[self.CVs.index('c1.' + str(alpha))].copy())

        # Innermost compression chamber begins to be tricky
        # By definition innermost compression chamber doesn't make it to the 
        # end of the rotation
        next_theta = self.t + 2 * pi * Nc_max1
        next_p1 = self.p[self.CVs.index('c1.' + str(Nc_max1))].copy()
        next_p1[np.isnan(next_p1)] = 0

        pd1 = self.p[self.CVs.index('d1')].copy()
        pddd = self.p[self.CVs.index('ddd')].copy()

        # Now check if d1 and d2 end before the end of the rotation (they don't 
        # neccessarily)
        if np.isnan(pd1[0]) and np.isnan(pd1[self.Itheta]):
            # d1 & d2 end before the end of the rotation
            # straightforward analysis (just add on pd1 and pd2)
            pd1[np.isnan(pd1)] = 0
            next_p1 += pd1

            # So we know that ddd DOES exist at the beginning/end of the rotation
            # work backwards to find the first place that the ddd does exist
            pdddA = pddd.copy()
            pdddB = pddd.copy()

            i = self.Itheta
            while i > 0:
                if np.isnan(pdddA[i]):
                    i += 1
                    break;
                i -= 1
            pdddA[0:i] = 0  # This is the end of the rotation
            next_p1 += pdddA

            theta = np.append(theta, next_theta)
            p1 = np.append(p1, next_p1)

            i = 0
            while i < len(pdddB):
                if np.isnan(pdddB[i]):
                    break;
                i += 1

            pdddB[i::] = np.nan  # This is the beginning of the next rotation

            theta = np.append(theta, self.t + 2 * pi * (Nc_max1 + 1))
            p1 = np.append(p1, pdddB)

        # Now check if d1 & d2 still exist at the end of the rotation
        elif not np.isnan(pd1[0]) and not np.isnan(pd1[self.Itheta]):
            # d1 & d2 don't end before the end of the rotation
            pd1A = pd1.copy()
            pd1B = pd1.copy()

            i = self.Itheta
            while i > 0:
                if np.isnan(pd1A[i]):
                    i += 1
                    break;
                i -= 1
            pd1A[0:i] = 0  # This is the end of the rotation
            next_p1 += pd1A

            theta = np.append(theta, next_theta)
            p1 = np.append(p1, next_p1)

            last_theta = self.t + 2 * pi * (Nc_max1 + 1)
            last_p1 = pddd.copy()
            last_p2 = pddd.copy()
            last_p1[np.isnan(last_p1)] = 0
            last_p2[np.isnan(last_p2)] = 0

            i = 0
            while i < len(pd1B):
                if np.isnan(pd1B[i]):
                    break;
                i += 1
            if i == len(pd1B) - 1:
                raise ValueError('d1B could not find NaN')

            pd1B[i::] = 0
            last_p1 += pd1B

            theta = np.append(theta, last_theta)
            p1 = np.append(p1, last_p1)

        self.summary.theta1_profile = theta
        self.summary.p1_profile = p1

        #####  ********* PATH #2 *****************
        #####  ********* PATH #2 *****************
        #####  ********* PATH #2 *****************

        theta = self.t

        # s1 path turns into c2.1 at theta = pi
        p2 = self.p[self.CVs.index('s2')].copy()
        p2[theta > pi] = self.p[self.CVs.index('c2.1')][theta > pi].copy()

        if Nc_max2 > 1:
            for alpha in range(1, Nc_max2):
                theta = np.append(theta, self.t + 2 * pi * alpha)
                _p2 = self.p[self.CVs.index('c2.' + str(alpha))].copy()
                _p2[self.t > pi] = self.p[self.CVs.index('c2.' + str(alpha + 1))][self.t > pi]
                p2 = np.append(p2, _p2)

        next_theta = self.t + 2 * pi * max(Nc_max2 - 1, 1)
        next_p2 = self.p[self.CVs.index('c2.' + str(max(Nc_max2 - 1, 1)))].copy()
        next_p2[self.t > self.theta_d] = 0
        next_p2[np.isnan(next_p2)] = 0

        pd2 = self.p[self.CVs.index('d2')].copy()
        pddd = self.p[self.CVs.index('ddd')].copy()

        # Now check if d1 and d2 end before the end of the rotation (they don't 
        # neccessarily)
        if np.isnan(pd2[0]) and np.isnan(pd2[self.Itheta]):
            # d1 & d2 end before the end of the rotation
            # straightforward analysis (just add on pd1 and pd2)
            pd2[np.isnan(pd2)] = 0
            next_p2 += pd2

            # So we know that ddd DOES exist at the beginning/end of the rotation
            # work backwards to find the first place that the ddd does exist
            pdddA = pddd.copy()
            pdddB = pddd.copy()

            i = self.Itheta
            while i > 0:
                if np.isnan(pdddA[i]):
                    i += 1
                    break;
                i -= 1
            pdddA[0:i] = 0  # This is the end of the rotation
            next_p2 += pdddA

            theta = np.append(theta, next_theta)
            p2 = np.append(p2, next_p2)

            i = 0
            while i < len(pdddB):
                if np.isnan(pdddB[i]):
                    break;
                i += 1

            pdddB[i::] = np.nan  # This is the beginning of the next rotation

            theta = np.append(theta, self.t + 2 * pi * (max(Nc_max2 - 1, 1) + 1))
            p2 = np.append(p2, pdddB)

        # Now check if d1 & d2 still exist at the end of the rotation
        elif not np.isnan(pd2[0]) and not np.isnan(pd2[self.Itheta]):
            # d1 & d2 don't end before the end of the rotation
            pd2A = pd2.copy()
            pd2B = pd2.copy()

            i = self.Itheta
            while i > 0:
                if np.isnan(pd2A[i]):
                    i += 1
                    break;
                i -= 1
            pd2A[0:i] = 0  # This is the end of the rotation
            next_p2 += pd2A

            theta = np.append(theta, next_theta)
            p2 = np.append(p2, next_p2)

            last_theta = self.t + 2 * pi * (Nc_max2)
            last_p2 = pddd.copy()
            last_p2[np.isnan(last_p2)] = 0

            i = 0
            while i < len(pd2B):
                if np.isnan(pd2B[i]):
                    break;
                i += 1
            if i == len(pd2B) - 1:
                raise ValueError('d2B could not find NaN')

            pd2B[i::] = 0
            last_p2 += pd2B

            theta = np.append(theta, last_theta)
            p2 = np.append(p2, last_p2)

        self.summary.theta2_profile = theta
        self.summary.p2_profile = p2

#        import matplotlib.pyplot as plt
#        plt.plot(self.summary.theta1_profile, self.summary.p1_profile)
#        plt.plot(self.summary.theta2_profile, self.summary.p2_profile)
#        plt.show()
#        plt.close()

    def build_volume_profile(self):
        """
        Build the volume profile, tracking along s1,c1.x,d1,ddd and
        s2,c2.x,d2,ddd and store them in the variables summary.theta1_profile,
        summary.V1_profile, summary.theta2_profile, summary.V2_profile

        Each path is handled separately

        """

        # Calculate along one path to track one set of pockets through the whole process
        theta = self.t

        Nc_max1, Nc_max2 = self.Nc_max()

        #####  ********* PATH #1 *****************
        #####  ********* PATH #1 *****************
        #####  ********* PATH #1 *****************

        theta = self.t

        # s1 path is treated normally
        V1 = self.V[self.CVs.index('s1')].copy()

        # Outer compression chambers
        for alpha in range(1, Nc_max1):
            theta = np.append(theta, self.t + 2 * pi * alpha)
            V1 = np.append(V1, self.V[self.CVs.index('c1.' + str(alpha))].copy())

        # Innermost compression chamber begins to be tricky
        # By definition innermost compression chamber doesn't make it to the
        # end of the rotation
        next_theta = self.t + 2 * pi * Nc_max1
        next_V1 = self.V[self.CVs.index('c1.' + str(Nc_max1))].copy()
        next_V1[np.isnan(next_V1)] = 0

        Vd1 = self.V[self.CVs.index('d1')].copy()
        Vddd = self.V[self.CVs.index('ddd')].copy()

        # Now check if d1 and d2 end before the end of the rotation (they don't
        # neccessarily)
        if np.isnan(Vd1[0]) and np.isnan(Vd1[self.Itheta]):
            # d1 & d2 end before the end of the rotation
            # straightforward analysis (just add on Vd1 and Vd2)
            Vd1[np.isnan(Vd1)] = 0
            next_V1 += Vd1

            # So we know that ddd DOES exist at the beginning/end of the rotation
            # work backwards to find the first place that the ddd does exist
            VdddA = Vddd.copy()
            VdddB = Vddd.copy()

            i = self.Itheta
            while i > 0:
                if np.isnan(VdddA[i]):
                    i += 1
                    break;
                i -= 1
            VdddA[0:i] = 0  # This is the end of the rotation
            next_V1 += VdddA

            theta = np.append(theta, next_theta)
            V1 = np.append(V1, next_V1)

            i = 0
            while i < len(VdddB):
                if np.isnan(VdddB[i]):
                    break;
                i += 1

            VdddB[i::] = np.nan  # This is the beginning of the next rotation

            theta = np.append(theta, self.t + 2 * pi * (Nc_max1 + 1))
            V1 = np.append(V1, VdddB)

        # Now check if d1 & d2 still exist at the end of the rotation
        elif not np.isnan(Vd1[0]) and not np.isnan(Vd1[self.Itheta]):
            # d1 & d2 don't end before the end of the rotation
            Vd1A = Vd1.copy()
            Vd1B = Vd1.copy()

            i = self.Itheta
            while i > 0:
                if np.isnan(Vd1A[i]):
                    i += 1
                    break;
                i -= 1
            Vd1A[0:i] = 0  # This is the end of the rotation
            next_V1 += Vd1A

            theta = np.append(theta, next_theta)
            V1 = np.append(V1, next_V1)

            last_theta = self.t + 2 * pi * (Nc_max1 + 1)
            last_V1 = Vddd.copy()
            last_V2 = Vddd.copy()
            last_V1[np.isnan(last_V1)] = 0
            last_V2[np.isnan(last_V2)] = 0

            i = 0
            while i < len(Vd1B):
                if np.isnan(Vd1B[i]):
                    break;
                i += 1
            if i == len(Vd1B) - 1:
                raise ValueError('d1B could not find NaN')

            Vd1B[i::] = 0
            last_V1 += Vd1B

            theta = np.append(theta, last_theta)
            V1 = np.append(V1, last_V1)

        self.summary.theta1_profile = theta
        self.summary.V1_profile = V1

        #####  ********* PATH #2 *****************
        #####  ********* PATH #2 *****************
        #####  ********* PATH #2 *****************

        theta = self.t

        # s1 path turns into c2.1 at theta = pi
        V2 = self.V[self.CVs.index('s2')].copy()
        V2[theta > pi] = self.V[self.CVs.index('c2.1')][theta > pi].copy()

        if Nc_max2 > 1:
            for alpha in range(1, Nc_max2):
                theta = np.append(theta, self.t + 2 * pi * alpha)
                _V2 = self.V[self.CVs.index('c2.' + str(alpha))].copy()
                _V2[self.t > pi] = self.V[self.CVs.index('c2.' + str(alpha + 1))][self.t > pi]
                V2 = np.append(V2, _V2)

        next_theta = self.t + 2 * pi * max(Nc_max2 - 1, 1)
        next_V2 = self.V[self.CVs.index('c2.' + str(max(Nc_max2 - 1, 1)))].copy()
        next_V2[self.t > self.theta_d] = 0
        next_V2[np.isnan(next_V2)] = 0

        Vd2 = self.V[self.CVs.index('d2')].copy()
        Vddd = self.V[self.CVs.index('ddd')].copy()

        # Now check if d1 and d2 end before the end of the rotation (they don't
        # neccessarily)
        if np.isnan(Vd2[0]) and np.isnan(Vd2[self.Itheta]):
            # d1 & d2 end before the end of the rotation
            # straightforward analysis (just add on Vd1 and Vd2)
            Vd2[np.isnan(Vd2)] = 0
            next_V2 += Vd2

            # So we know that ddd DOES exist at the beginning/end of the rotation
            # work backwards to find the first place that the ddd does exist
            VdddA = Vddd.copy()
            VdddB = Vddd.copy()

            i = self.Itheta
            while i > 0:
                if np.isnan(VdddA[i]):
                    i += 1
                    break;
                i -= 1
            VdddA[0:i] = 0  # This is the end of the rotation
            next_V2 += VdddA

            theta = np.append(theta, next_theta)
            V2 = np.append(V2, next_V2)

            i = 0
            while i < len(VdddB):
                if np.isnan(VdddB[i]):
                    break;
                i += 1

            VdddB[i::] = np.nan  # This is the beginning of the next rotation

            theta = np.append(theta, self.t + 2 * pi * (max(Nc_max2 - 1, 1) + 1))
            V2 = np.append(V2, VdddB)

        # Now check if d1 & d2 still exist at the end of the rotation
        elif not np.isnan(Vd2[0]) and not np.isnan(Vd2[self.Itheta]):
            # d1 & d2 don't end before the end of the rotation
            Vd2A = Vd2.copy()
            Vd2B = Vd2.copy()

            i = self.Itheta
            while i > 0:
                if np.isnan(Vd2A[i]):
                    i += 1
                    break;
                i -= 1
            Vd2A[0:i] = 0  # This is the end of the rotation
            next_V2 += Vd2A

            theta = np.append(theta, next_theta)
            V2 = np.append(V2, next_V2)

            last_theta = self.t + 2 * pi * (Nc_max2)
            last_V2 = Vddd.copy()
            last_V2[np.isnan(last_V2)] = 0

            i = 0
            while i < len(Vd2B):
                if np.isnan(Vd2B[i]):
                    break;
                i += 1
            if i == len(Vd2B) - 1:
                raise ValueError('d2B could not find NaN')

            Vd2B[i::] = 0
            last_V2 += Vd2B

            theta = np.append(theta, last_theta)
            V2 = np.append(V2, last_V2)

        self.summary.theta2_profile = theta
        self.summary.V2_profile = V2


    def build_temperature_profile(self):
        """
        Build the temperature profile, tracking along s1,c1.x,d1,ddd and
        s2,c2.x,d2,ddd and store them in the variables summary.theta1_profile,
        summary.T1_profile, summary.theta2_profile, summary.T2_profile

        Each path is handled separately

        """

        # Calculate along one path to track one set of pockets through the whole process
        theta = self.t

        Nc_max1, Nc_max2 = self.Nc_max()

        #####  ********* PATH #1 *****************
        #####  ********* PATH #1 *****************
        #####  ********* PATH #1 *****************

        theta = self.t

        # s1 path is treated normally
        T1 = self.T[self.CVs.index('s1')].copy()

        # Outer compression chambers
        for alpha in range(1, Nc_max1):
            theta = np.append(theta, self.t + 2 * pi * alpha)
            T1 = np.append(T1, self.T[self.CVs.index('c1.' + str(alpha))].copy())

        # Innermost compression chamber begins to be tricky
        # By definition innermost compression chamber doesn't make it to the
        # end of the rotation
        next_theta = self.t + 2 * pi * Nc_max1
        next_T1 = self.T[self.CVs.index('c1.' + str(Nc_max1))].copy()
        next_T1[np.isnan(next_T1)] = 0

        Td1 = self.T[self.CVs.index('d1')].copy()
        Tddd = self.T[self.CVs.index('ddd')].copy()

        # Now check if d1 and d2 end before the end of the rotation (they don't
        # neccessarily)
        if np.isnan(Td1[0]) and np.isnan(Td1[self.Itheta]):
            # d1 & d2 end before the end of the rotation
            # straightforward analysis (just add on Td1 and Td2)
            Td1[np.isnan(Td1)] = 0
            next_T1 += Td1

            # So we know that ddd DOES exist at the beginning/end of the rotation
            # work backwards to find the first place that the ddd does exist
            TdddA = Tddd.copy()
            TdddB = Tddd.copy()

            i = self.Itheta
            while i > 0:
                if np.isnan(TdddA[i]):
                    i += 1
                    break;
                i -= 1
            TdddA[0:i] = 0  # This is the end of the rotation
            next_T1 += TdddA

            theta = np.append(theta, next_theta)
            T1 = np.append(T1, next_T1)

            i = 0
            while i < len(TdddB):
                if np.isnan(TdddB[i]):
                    break;
                i += 1

            TdddB[i::] = np.nan  # This is the beginning of the next rotation

            theta = np.append(theta, self.t + 2 * pi * (Nc_max1 + 1))
            T1 = np.append(T1, TdddB)

        # Now check if d1 & d2 still exist at the end of the rotation
        elif not np.isnan(Td1[0]) and not np.isnan(Td1[self.Itheta]):
            # d1 & d2 don't end before the end of the rotation
            Td1A = Td1.copy()
            Td1B = Td1.copy()

            i = self.Itheta
            while i > 0:
                if np.isnan(Td1A[i]):
                    i += 1
                    break;
                i -= 1
            Td1A[0:i] = 0  # This is the end of the rotation
            next_T1 += Td1A

            theta = np.append(theta, next_theta)
            T1 = np.append(T1, next_T1)

            last_theta = self.t + 2 * pi * (Nc_max1 + 1)
            last_T1 = Tddd.copy()
            last_T2 = Tddd.copy()
            last_T1[np.isnan(last_T1)] = 0
            last_T2[np.isnan(last_T2)] = 0

            i = 0
            while i < len(Td1B):
                if np.isnan(Td1B[i]):
                    break;
                i += 1
            if i == len(Td1B) - 1:
                raise ValueError('d1B could not find NaN')

            Td1B[i::] = 0
            last_T1 += Td1B

            theta = np.append(theta, last_theta)
            T1 = np.append(T1, last_T1)

        self.summary.theta1_profile = theta
        self.summary.T1_profile = T1

        #####  ********* PATH #2 *****************
        #####  ********* PATH #2 *****************
        #####  ********* PATH #2 *****************

        theta = self.t

        # s1 path turns into c2.1 at theta = pi
        T2 = self.T[self.CVs.index('s2')].copy()
        T2[theta > pi] = self.T[self.CVs.index('c2.1')][theta > pi].copy()

        if Nc_max2 > 1:
            for alpha in range(1, Nc_max2):
                theta = np.append(theta, self.t + 2 * pi * alpha)
                _T2 = self.T[self.CVs.index('c2.' + str(alpha))].copy()
                _T2[self.t > pi] = self.T[self.CVs.index('c2.' + str(alpha + 1))][self.t > pi]
                T2 = np.append(T2, _T2)

        next_theta = self.t + 2 * pi * max(Nc_max2 - 1, 1)
        next_T2 = self.T[self.CVs.index('c2.' + str(max(Nc_max2 - 1, 1)))].copy()
        next_T2[self.t > self.theta_d] = 0
        next_T2[np.isnan(next_T2)] = 0

        Td2 = self.T[self.CVs.index('d2')].copy()
        Tddd = self.T[self.CVs.index('ddd')].copy()

        # Now check if d1 and d2 end before the end of the rotation (they don't
        # neccessarily)
        if np.isnan(Td2[0]) and np.isnan(Td2[self.Itheta]):
            # d1 & d2 end before the end of the rotation
            # straightforward analysis (just add on Td1 and Td2)
            Td2[np.isnan(Td2)] = 0
            next_T2 += Td2

            # So we know that ddd DOES exist at the beginning/end of the rotation
            # work backwards to find the first place that the ddd does exist
            TdddA = Tddd.copy()
            TdddB = Tddd.copy()

            i = self.Itheta
            while i > 0:
                if np.isnan(TdddA[i]):
                    i += 1
                    break;
                i -= 1
            TdddA[0:i] = 0  # This is the end of the rotation
            next_T2 += TdddA

            theta = np.append(theta, next_theta)
            T2 = np.append(T2, next_T2)

            i = 0
            while i < len(TdddB):
                if np.isnan(TdddB[i]):
                    break;
                i += 1

            TdddB[i::] = np.nan  # This is the beginning of the next rotation

            theta = np.append(theta, self.t + 2 * pi * (max(Nc_max2 - 1, 1) + 1))
            T2 = np.append(T2, TdddB)

        # Now check if d1 & d2 still exist at the end of the rotation
        elif not np.isnan(Td2[0]) and not np.isnan(Td2[self.Itheta]):
            # d1 & d2 don't end before the end of the rotation
            Td2A = Td2.copy()
            Td2B = Td2.copy()

            i = self.Itheta
            while i > 0:
                if np.isnan(Td2A[i]):
                    i += 1
                    break;
                i -= 1
            Td2A[0:i] = 0  # This is the end of the rotation
            next_T2 += Td2A

            theta = np.append(theta, next_theta)
            T2 = np.append(T2, next_T2)

            last_theta = self.t + 2 * pi * (Nc_max2)
            last_T2 = Tddd.copy()
            last_T2[np.isnan(last_T2)] = 0

            i = 0
            while i < len(Td2B):
                if np.isnan(Td2B[i]):
                    break;
                i += 1
            if i == len(Td2B) - 1:
                raise ValueError('d2B could not find NaN')

            Td2B[i::] = 0
            last_T2 += Td2B

            theta = np.append(theta, last_theta)
            T2 = np.append(T2, last_T2)

        self.summary.theta2_profile = theta
        self.summary.T2_profile = T2

    def post_solve(self):
        """
        Overwrite method 'post_solve' from DanfossScroll class to take into account asymmetric scroll for the
        calculation of polytropic coefficients
        TODO: Fix the calculation of polytropic coefficients
        """

        # Run the Scroll method to set pressure profile, etc.
        Scroll.post_solve(self)

        # Build the volume profile
        self.build_volume_profile()

        # Build the temperature profile
        self.build_temperature_profile()

        # Add polytropic coefficients to the summary
        def compute_polytropic_coefficient(p,V):
            from scipy.optimize import curve_fit

            p_init = p[0]
            V_init = V[0]

            def func(V, gamma):
                return p_init*(V_init/V)**gamma

            popt, pcov = curve_fit(func,V,p,1.05)
            return popt[0]

        # p1 = self.summary.p1_profile[(self.summary.theta1_profile>=2*np.pi) & (self.summary.theta1_profile<=self.geo.phi_fie-self.geo.phi_oos-np.pi)]
        # p2 = self.summary.p2_profile[(self.summary.theta2_profile>=2*np.pi) & (self.summary.theta2_profile<=self.geo.phi_fie-self.geo.phi_oos-np.pi)]
        # V1 = self.summary.V1_profile[(self.summary.theta1_profile>=2*np.pi) & (self.summary.theta1_profile<=self.geo.phi_fie-self.geo.phi_oos-np.pi)]
        # V2 = self.summary.V2_profile[(self.summary.theta2_profile>=2*np.pi) & (self.summary.theta2_profile<=self.geo.phi_fie-self.geo.phi_oos-np.pi)]

        self.summary.polytropic_coeff1 = 1. # compute_polytropic_coefficient(p1,V1)
        self.summary.polytropic_coeff2 = 1. # compute_polytropic_coefficient(p2,V2)

def geo_validation_data(sim):
    import numpy as np
    from PDSim.scroll import asymm_scroll_geo, common_scroll_geo as common
    output_str = ''
    entries = [('$sa$', 'sa', sim.V_sa),
               ('$s_1$', 's1', sim.V_s1),
               ('$s_2$', 's2', sim.V_s2),
               ('$d_1$', 'd1', sim.V_d1),
               ('$d_2$', 'd2', sim.V_d2),
               ('$dd$', 'dd', sim.V_dd),
               ]

    for alpha in range(1, asymm_scroll_geo.Nc(0.0, sim.geo, 1) + 1):
        tex = '$c_{1,' + str(alpha) + '}$'
        key = 'c1.' + str(alpha)
        fcn = sim.V_c1
        entries.append((tex, key, fcn))

    output_str += r'$\theta$ & ' + ' & '.join([tex for (tex, key, val) in entries]) + '\\\\ \n'
    output_str += '\hline' + '\n'
    output_str += '\multicolumn{{{N:d}}}{{c}}{{V (analytic - polygon)*1e6}}\\\\\n'.format(N=len(entries) + 1)
    output_str += '\hline' + '\n'
    for th in np.linspace(0, 2 * pi, 9):
        vals = []
        for (tex, key, fcn) in entries:
            xp, yp = asymm_scroll_geo.CVcoords(key, sim.geo, th)
            V_polygon = common.polyarea(xp, yp) * sim.geo.h
            if key.find('.') > -1:
                V_PDSim = fcn(th, int(key.split('.')[1]))[0]
            else:
                V_PDSim = fcn(th)[0]

            vals.append(str(round((V_polygon - V_PDSim) * 1e6, 3)))

        output_str += str(round(th, 3)) + ' & ' + ' & '.join(vals) + '\\\\ \n'

    output_str += r'\multicolumn{{{N:d}}}{{c}}{{$dV/d\theta$ (analytic - numeric)*1e6}}\\\\\n'.format(
        N=len(entries) + 1)
    for th in np.linspace(0, 2 * pi, 11):
        vals = []
        for (tex, key, fcn) in entries:
            xp, yp = asymm_scroll_geo.CVcoords(key, sim.geo, th)

            if key.find('.') > -1:
                dV_PDSim_num = (fcn(th + 1e-7)[0] - fcn(th)[0]) / (1e-7)
                dV_PDSim_ana = fcn(th)[1]
            else:
                dV_PDSim_num = (fcn(th + 1e-7)[0] - fcn(th)[0]) / (1e-7)
                dV_PDSim_ana = fcn(th)[1]

            vals.append(str(round((dV_PDSim_num - dV_PDSim_ana) * 1e6, 3)))

        output_str += str(round(th, 3)) + ' & ' + ' & '.join(vals) + '\\\\ \n'

    print(output_str)


if __name__ == '__main__':
    pass
