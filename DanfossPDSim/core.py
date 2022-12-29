from __future__ import print_function

from sys import modules
from math import pi
from PDSim.core.core import struct
from PDSim.core.containers import ControlVolume
from PDSim.core.containers import Tube
from PDSim.flow import flow_models
from PDSim.flow.flow_models import IsentropicNozzleWrapper
from PDSim.flow.flow import FlowPath
from PDSim.flow.fanno import Fanno_Ma_nondimLength, p_pstar
from PDSim.scroll import scroll_geo
from PDSim.scroll.core import Scroll
try:
    from PDSim.scroll.plots import plotScrollSet
    from PDSim.plot.plots import debug_plots
except ImportError:
    plotScrollSet = None
    debug_plots = None

from PDSim.core.motor import Motor
from PDSim.misc.hdf5 import HDF5Writer
from DanfossPDSim.asymm_scroll_geo import DanfossGeoVals
from PDSim.scroll.common_scroll_geo import sortAnglesCCW

import matplotlib.pyplot as plt

from CoolProp import State
from CoolProp import CoolProp as CP

import numpy as np
import scipy.optimize
from numpy.linalg import norm

import time
from getpass import getuser

class IDVPort(object):

    #: Involute angle of the involute used to locate this point
    phi = 3.14159

    #: The code for the involute used to locate this point: 'i' or 'o'
    involute = 'i'

    #: Distance away from the involute
    offset = 0.003

    #: Diameter of the port
    D = 0.002

    #: The x coordinate of the center of the port
    x0 = None

    #: The y coordinate of the center of the port
    y0 = None

    theta = 0

    area_dict = 0

class DanfossScroll(Scroll):
    """
    A Danfoss-specialized scroll compressor class that implements additional 
    analysis required for Danfoss compressors
    """
    def __init__(self,*args,**kwargs):
        Scroll.__init__(self,*args,**kwargs)

        self.scroll_geo_module = scroll_geo

        # Upgrade the geometry to the Danfoss geometry class
        self.geo = DanfossGeoVals()

        # Version of main modules
        self.version = {"PDSim":modules["PDSim"].__version__+' git branch: '+str(modules["PDSim"].__git_branch__)+' git revision: '+str(modules["PDSim"].__git_revision__),
                        "CoolProp":modules["CoolProp"].__version__+ ' git revision: '+str(modules["CoolProp"].__gitrevision__),
                        "DanfossPDSim":modules["DanfossPDSim"].__version__+' git branch: '+str(modules["DanfossPDSim"].__git_branch__)+' git revision: '+str(modules["DanfossPDSim"].__git_revision__)}

        self.user = getuser()

    def IDVValveWithDynamics(self,FlowPath,**kwargs):
        try:
            FlowPath.A = self.IDV_valve.A()
            mdot=flow_models.IsentropicNozzle(FlowPath.A,FlowPath.State_up,FlowPath.State_down)
            return mdot
        except ZeroDivisionError:
            return 0.0

    def IDVValveNoDynamics(self,FlowPath,**kwargs):
        if FlowPath.key_up=='discharge_plenum':
            ## pressure in discharge plenum higher than the IDV port
            ## valve is closed - no flow
            return 0.0
        else:
            try:
                mdot=flow_models.IsentropicNozzle(FlowPath.A,
                                                  FlowPath.State_up,
                                                  FlowPath.State_down)
                return mdot
            except ZeroDivisionError:
                return 0.0

    def add_IDVs(self, IDVinfo, debug=False):
        """
        Add IDVs to the model, as specified by a JSON data structure


        """
        if debug:
            print(IDVinfo)

        assert(isinstance(IDVinfo, dict))

        # Constant volume discharge shell (the plenum for the IDV)
        # -------------------------------
        # Create a discharge plenum control volume which is connected to all the IDV CV
        # and then flows to the compressor side of the discharge line
        plenum = IDVinfo['plenum']
        def get_plenum_key(key):
            if key not in plenum:
                raise KeyError(f'Key "{key:s}" not included for plenum')
            return plenum[key]

        plenum_key = get_plenum_key('key')
        plenum_volume = get_plenum_key('volume / m^3')
        disc_tube_node = get_plenum_key('tube_node')
        tube_crosssectionarea = get_plenum_key('tube_flowarea / m^2')
        fluid = get_plenum_key('fluid')
        T_K = get_plenum_key('T / K')
        p_kPa = get_plenum_key('p / kPa')
        self.add_CV(
            ControlVolume(key=plenum_key,
                          VdVFcn=self.V_injection, #  Constant volume 
                          VdVFcn_kwargs=dict(V_tube=plenum_volume),
                          initialState=State.State(fluid, dict(T=T_K, P=p_kPa))))
        FP = FlowPath(key1=plenum_key, 
                      key2=disc_tube_node,
                      MdotFcn=IsentropicNozzleWrapper(),
                  )
        FP.A = tube_crosssectionarea
        self.add_flow(FP)

        for IDV in IDVinfo['IDVs']:
            def get_key(key):
                if key not in IDV:
                    raise KeyError(f'Key "{key:s}" not included for IDV')
                return IDV[key]
            
            CVkey = get_key('key')
            volume_m3 = get_key('volume / m^3')
            fluid = get_key('fluid')
            T_K = get_key('T / K')
            p_kPa = get_key('p / kPa')

            # Add the control volume for the IDV control volume
            self.add_CV(
                ControlVolume(key=CVkey,
                              VdVFcn=self.V_injection, #  Constant volume
                              VdVFcn_kwargs=dict(V_tube = volume_m3),
                              initialState=State.State(fluid, dict(T=T_K, P=p_kPa)))
            )

            # Add the flow between the IDV control volume and the plenum
            FP = FlowPath(key2=CVkey,
                          key1=plenum_key,
                          MdotFcn=self.IDVValveNoDynamics,
                          )
            FP.A = get_key('plenum_flowarea / m^2')
            self.add_flow(FP)

            # Loop over the ports to be added for the given IDV
            for port in IDV['ports']:
                def get_key(key):
                    if key not in port:
                        raise KeyError(f'Key "{key:s}" not included for port')
                    return port[key]

                x_m = get_key('x / m')
                y_m = get_key('y / m')
                D_m = get_key('D / m')
                p = IDVPort()
                p.x0 = x_m
                p.y0 = y_m
                p.D = D_m

                self.calculate_IDV_area(p)

                for partner in p.area_dict:
            
                    #  Create a spline interpolator object for the area between port and the partner chamber
                    A_interpolator = scipy.interpolate.splrep(p.theta, p.area_dict[partner], k=1, s=0)
                    
                    #  Add the flow between the IDV control volume and the chamber through the port
                    self.add_flow(FlowPath(key1=CVkey,
                                          key2=partner,
                                          MdotFcn=self.IDV_CV_flow,
                                          MdotFcn_kwargs=dict(X_d=1.0, 
                                                              A_interpolator=A_interpolator)
                                         )
                                )

    def tipseal_Fanno(self, FlowPath, Xd = 1.0):
        w = self.w_slot
        h = self.delta_slot
        D_h = 2*h*w/(h+w)
        f_F = self.fF_slot
        phi0 = (self.geo.phi_i0+self.geo.phi_o0)/2.0
        phie = (self.geo.phi_ie+self.geo.phi_oe)/2.0
        phis = self.geo.phi_is
        if FlowPath.key2 in ['s1','s2']:
            L12 = self.geo.rb*((phie**2-phis**2)/2.0-phi0*(phie-phis))
        elif FlowPath.key2 in ['c1.1','c2.1']:
            L12 = self.geo.rb*((phie**2-(phis+pi)**2)/2.0-phi0*(phie-(phis+pi)))
        elif FlowPath.key2 in ['c1.2','c2.2']:
            L12 = self.geo.rb*((phie**2-(phis+2*pi)**2)/2.0-phi0*(phie-(phis+2*pi)))
        else:
            raise ValueError

        gamma = FlowPath.State_up.cp/FlowPath.State_up.cv
        Lparam = 4*f_F*L12/D_h

        # First we assume it is choked at the outlet of the flow path (Ma2 = 1.0)
        # which means that the length L12 is the sonic length
        Ma1 = Fanno_Ma_nondimLength(Lparam, gamma)
#        p1_pstar = p_pstar(Ma1, gamma)
#        p2_pstar = 1.0
        #Then we check if the actual pressure ratio (p1/p2) is greater than the
        #choked pressure ratio.  If it is, the assumption of choking was correct
#        p1_p2 = FlowPath.State_up.p/FlowPath.State_down.p

        #if p1_p2 >= p1_pstar/p2_pstar:
            # Flow is choked, we are done
        return Xd*FlowPath.State_up.rho*Ma1*FlowPath.State_up.get_speed_sound()*pi*D_h**2/4.0
        #else:
        #    print p1_p2, p1_pstar/p2_pstar
            # Flow is not choked, need to do more work
        #    raise ValueError('Fanno flow not supported for non-choked flow yet')

    def IDV_CV_flow(self, FlowPath, X_d = 1.0, A_interpolator = None, DP_floor = 0.001):
        """
        A generic isentropic nozzle flow model wrapper with the added consideration
        that if the pressure drop is below the floor value, there is no flow.
        This code was originally added to handle the case of the injection line 
        where there is no flow out of the injection which greatly increases the 
        numerical stiffness 
        
        This function also implements the use of spline interpolation to calculate
        the area between the IDV port and the control volume
        
        Parameters
        ----------
        FlowPath : FlowPath instance
            A fully-instantiated flow path model
        A : float
            throat area for isentropic nozzle model [:math:`m^2`]
        DP_floor: float
            The minimum pressure drop [kPa]
            
        Returns
        -------
        mdot : float
            The mass flow through the flow path [kg/s]
        """
        FlowPath.A = scipy.interpolate.splev(self.theta, A_interpolator)
        try:
            if FlowPath.State_up.p-FlowPath.State_down.p > DP_floor:
                mdot = X_d*flow_models.IsentropicNozzle(FlowPath.A,
                                                        FlowPath.State_up,
                                                        FlowPath.State_down)
                return mdot
            else:
                return 0.0
        except ZeroDivisionError:
            return 0.0

    def calculate_IDV_area(self, port):
        """
        For a given IDV port, calculate the area between all other chambers
        """

        #  Make sure it is an IDVPort instance
        assert (isinstance(port,IDVPort))

        if port.x0 is None and port.y0 is None:
            #  Get the reference point on the scroll wrap
            if port.involute == 'i':
                #  Point on the scroll involute
                x, y = scroll_geo.coords_inv(port.phi, self.geo, 0, 'fi')
                #  Unit normal vector components
                nx, ny = scroll_geo.coords_norm(port.phi, self.geo, 0, 'fi')
            elif port.involute == 'o':
                #  Point on the scroll involute
                x, y = scroll_geo.coords_inv(port.phi, self.geo, 0, 'fo')
                #  Unit normal vector components
                nx, ny = scroll_geo.coords_norm(port.phi, self.geo, 0, 'fo')
            else:
                raise ValueError

            #  Normal direction points towards the scroll wrap, take the opposite 
            #  direction to locate the center of the port
            port.x0 = x - port.offset*nx
            port.y0 = y - port.offset*ny

        #  The coordinates for the port
        t = np.linspace(0, 2*pi)
        xport = port.x0 + port.D/2.0*np.cos(t)
        yport = port.y0 + port.D/2.0*np.sin(t)

        #  Actually use the clipper library to carry out the intersection
        #  of the port with all of the control volumes
        theta_area, area_dict = self.poly_intersection_with_cvs(xport, yport, 300, CVcoords = self.scroll_geo_module.CVcoords)

        #  Save the values
        port.area_dict = area_dict
        port.theta = theta_area

    def calculate_IDV_areas(self):
        """ 
        Calculate the area between an IDV port and all of the control volumes 

        This function is essentially a clone of the calculate_port_areas function of PDSim.Scroll,
        except that different data types are used for the Port class
        """

        # Iterate over the IDV ports, calculating the area for each
        for port in self.IDV_ports:
            self.calculate_IDV_area(port)

#            #  Plot them
#            for k, A in area_dict.iteritems():
#                plt.plot(theta_area, A, label = k)
#
#            plt.legend()
#            plt.show()

    def DISC_D2_DUMMY(self, FP, X_d = 1.0, A_cs_dummy = None):
        """
        The flow path function for the flow through the dummy port
        """
        if A_cs_dummy is None:
            raise ValueError('A_cs_dummy is None')

        # Open area of the dummy port
        FP.A = scipy.interpolate.splev(self.theta, self.spline_Adisc_D1)

        # Use the minimum area of the cross-section of the dummy port and the 
        # open area of the dummy port
        FP.A = min(FP.A, A_cs_dummy)

        try:
            return flow_models.IsentropicNozzle(FP.A, FP.State_up, FP.State_down) * X_d
        except ZeroDivisionError:
            return 0.0

    def _Area_d_dd_Picavet2013(self, theta, geo):
        dtheta = geo.phi_oos + pi - geo.phi_fis
        sinu = geo.ro/geo.ra_arc1*np.sin(theta-scroll_geo.theta_d(geo)-dtheta)
        cosu = (1 - sinu**2)**0.5

        x_oos,y_oos = scroll_geo.coords_inv(geo.phi_os,geo,theta,"oo")
        arg = (x_oos-geo.xa_arc1)**2+(y_oos-geo.ya_arc1)**2-geo.ra_arc1**2*sinu**2
        if arg < 0:
            raise ValueError("Desired area in _Area_d_dd_Picavet2013 cannot be obtained")
        dist = geo.ra_arc1*cosu - (arg)**0.5
        return geo.h*(dist**2)**0.5

    def _Area_d_dd_osculation(self, theta, geo):
        """ 
        Inspired by the analysis of Alain Picavet from 2013,
        an even simpler method to calculate the flow area
        is to calculate the distance between the center of the radius 1
        on the fixed scroll to the point at the "os" point on the orbiting
        scroll at the starting angle on the outer involute.

        This approach has the correct limiting value of zero flow area
        at the point of contact at the discharge angle and will
        yield a well-behaved flow area at all angles
        """
        x_oos, y_oos = scroll_geo.coords_inv(geo.phi_os,geo,theta,"oo")
        distsquared = (x_oos-geo.xa_arc1)**2+(y_oos-geo.ya_arc1)**2

        return geo.h * (geo.ra_arc1 - distsquared**0.5)

    def _Area_d_dd_numeric(self, theta, geo, plot=False):
        """ Numerically calculate the minimum distance """

        # The points along the fixed scroll
        x_fi, y_fi = scroll_geo.coords_inv(np.linspace(geo.phi_is, geo.phi_is+np.pi/2,300),geo,theta,"fi")
        t = np.linspace(geo.t2_arc1, geo.t1_arc1, 300)
        x_fa1, y_fa1 = geo.xa_arc1 + geo.ra_arc1*np.cos(t), geo.ya_arc1 + geo.ra_arc1*np.sin(t)
        t = np.linspace(geo.t1_arc2, geo.t2_arc2, 300)
        x_fa2, y_fa2 = geo.xa_arc2 + geo.ra_arc2*np.cos(t), geo.ya_arc2 + geo.ra_arc2*np.sin(t)
        x_f = np.r_[x_fi[::-1], x_fa1, x_fa2]
        y_f = np.r_[y_fi[::-1], y_fa1, y_fa2]

        # The points along the orbiting scroll
        x_oo, y_oo = scroll_geo.coords_inv(np.linspace(geo.phi_os, geo.phi_os+np.pi/2, 300),geo,theta,"oo")
        theta_m = geo.phi_fie - theta - pi/2
        x_oa1, y_oa1 = -x_fa1 + geo.ro*np.cos(theta_m), -y_fa1 + geo.ro*np.sin(theta_m)
        x_oa2, y_oa2 = -x_fa2 + geo.ro*np.cos(theta_m), -y_fa2 + geo.ro*np.sin(theta_m)
        x_o = np.r_[x_oa1, x_oa2, x_oo]
        y_o = np.r_[y_oa1, y_oa2, y_oo]

        x_oos, y_oos = scroll_geo.coords_inv(geo.phi_os, geo, theta, "oo")

        if plot:
            plt.plot(x_o, y_o, color='r')
            plt.plot(x_f, y_f, color='k')
            plt.plot(x_oos, y_oos, '*')
            plt.axis('equal')
            plt.show()

        from scipy.spatial.distance import cdist
        XA = np.c_[x_o, y_o]
        XB = np.c_[x_f, y_f]
        d = cdist(XA, XB, 'euclidean').min()
        
        # d = ((x_oos-x_f)**2 + (y_oos-y_f)**2)**0.5
        return np.min(d)*geo.h

    def D_to_DD(self,FlowPath,X_d =1.0,method='osculation',**kwargs):
        """
        Overwrite method 'D_to_DD' from Scroll class to overwrite method 'Area_d_dd' from scroll_geo module.
        """
        if method == 'osculation':
            FlowPath.A = self._Area_d_dd_osculation(self.theta, self.geo)
        elif method == 'Picavet2013':
            FlowPath.A = self._Area_d_dd_Picavet2013(self.theta, self.geo)
        else:
            raise KeyError(f'{method} is invalid')
        
        try:
            return flow_models.IsentropicNozzle(FlowPath.A,
                                                FlowPath.State_up,
                                                FlowPath.State_down)*X_d
        except ZeroDivisionError:
            return 0.0


    def suction_heating(self):

        #  Calculate the heat transfer between the discharge and suction temperatures
        #  using a global UA factor.  This heat is removed from the discharge line
        #  and added to the suction line   
        Tinlet = self.Tubes['inlet.1'].State1.T
        Toutlet = self.Tubes['outlet.2'].State2.T
        Q = self.UA_suct_disc * (Toutlet - Tinlet)

        if hasattr(self,'motor'):
            # If some fraction of heat from motor losses is going to get added
            # to suction flow
            if 0.0 <= self.motor.suction_fraction <= 1.0:
                for Tube in self.Tubes:
                    # Find the tube that has one of the keys starting with 'inlet'
                    if Tube.key1.startswith('inlet') or Tube.key2.startswith('inlet'):
                        #Add some fraction of the motor losses to the inlet gas
                        Tube.Q_add = self.motor.losses * self.motor.suction_fraction + Q
                    elif Tube.key1.startswith('outlet') or Tube.key2.startswith('outlet'):
                        Tube.Q_add = -Q
                    else:
                        Tube.Q_add = 0.0

    def tip_seal_bypass(self, FP,  A = -1, Ncv_check = -1, path = -1):
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
            if Ncv_check == scroll_geo.getNc(self.theta, self.geo):
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

    def add_Danfoss_flows(self):
        """
        Add the Danfoss-specific mass flow terms, which in this case includes:
        
        * Fanno flow between dd/ddd chamber and s1 and s2 chambers
        * Bypass around the tipseal
        * Nose leakage (disabled for now)
        * Increased flow through the radial leakages (decreased length) to account for the tipseal
        TODO: Double check the flows and leakages with Danfoss Scroll
        """
        import warnings

        for key in ['c1.1','c2.1','c1.2','c2.2']: #'s1','s2'

            if not hasattr(self,'slot_flow_model'):
                self.slot_flow_model = 'Nozzle'
                warnings.warn('self.slot_flow_model was not provided, defaulting to Nozzle')
                if not hasattr(self, 'w_slot'):
                    warnings.warn('self.w_slot was not provided, no slot leakage flow has been added')
                    return

            if self.slot_flow_model == 'Fanno':
                ## Add the tip seal Fanno flow term
                self.add_flow(FlowPath(key1 = 'dd',
                                       key2 = key,
                                       MdotFcn = self.tipseal_Fanno,
                                       MdotFcn_kwargs = dict(Xd = self.Xd_Fanno)
                                       )
                              )
                self.add_flow(FlowPath(key1 = 'ddd',
                                       key2 = key,
                                       MdotFcn = self.tipseal_Fanno,
                                       MdotFcn_kwargs = dict(Xd = self.Xd_Fanno)
                                       )
                              )
            elif self.slot_flow_model == 'Nozzle':
                FP = FlowPath(key1=key,
                              key2='ddd',
                              MdotFcn=IsentropicNozzleWrapper(),
                             )
                FP.A = self.w_slot*self.delta_slot*self.Xd_Fanno
                self.add_flow(FP)

                FP = FlowPath(key1=key,
                              key2='dd',
                              MdotFcn=IsentropicNozzleWrapper(),
                             )
                FP.A = self.w_slot*self.delta_slot*self.Xd_Fanno
                self.add_flow(FP)
            else:
                raise ValueError('slot_flow_model [{m:s}] is invalid'.format(m=slot_flow_model))

        # The bypass term around the tipseal contact
        self.auto_add_flank_leakage(flankFunc = self.tip_seal_bypass,
                                    flankFunc_kwargs = dict(A = self.w_bypass*self.delta_axial*self.Xd_bypass)
                                    )

        # Arc-length to be used for the nose leakage term
        s_nose =self.geo.rb*((self.geo.phi_fis**2-self.geo.phi_fi0**2)/2.0-self.geo.phi_fi0*(self.geo.phi_fis-self.geo.phi_fi0))

#        # Nose bypass terms
#        FP = FlowPath(key1='c1.2',
#                      key2='ddd',
#                      MdotFcn=IsentropicNozzleWrapper(),
#                     )
#        FP.A = self.delta_axial*s_nose
#        self.add_flow(FP)
#
#        FP = FlowPath(key1='c2.2',
#                      key2='ddd',
#                      MdotFcn=IsentropicNozzleWrapper(),
#                      )
#        FP.A = self.delta_axial*s_nose
#        self.add_flow(FP)
#
        # Nose bypass terms
        FP = FlowPath(key1='d1',
                      key2='dd',
                      MdotFcn=IsentropicNozzleWrapper(),
                     )
        FP.A = self.delta_axial*s_nose
        self.add_flow(FP)

        FP = FlowPath(key1='d2',
                      key2='dd',
                      MdotFcn=IsentropicNozzleWrapper(),
                      )
        FP.A = self.delta_axial*s_nose
        self.add_flow(FP)

        # For every flow path that is a radial leakage (has RadialLeakage in the name), 
        # decrease the effective length of the flow path
        for flow in self.Flows:
            try:
                if 'RadialLeakage' in flow.MdotFcn.__name__:
                    flow.MdotFcn.kwargs['t'] = self.geo.t*0.6
                    print('set the effective thickness of the scroll wrap to 0.6*t due to presence of tip seal; kwargs are now: '+str(flow.MdotFcn.kwargs))
            except BaseException:
                pass

    def scroll_involute_axial_force(self, theta, p_backpressure = 0):
        """
        Overwrite method 'scroll_involute_axial_force' from Scroll class to
        - Fix a bug regarding the continuity of the load. The problem came from the value of phi1 for 'd2' pocket which
        wasn't continue with 'c2.Ncmax'.
        - Add the involute axial force of the Fixed Scroll (see paper "Comprehensive analytic solutions for the
        geometry of symmetric constant-wall-thickness scroll machines" (Ian Bell), Table 1 to get the involute angles
        for each scroll wrap)
        """

        def curve_length(phi_max, phi_min, phi_0):
            """
            Return the curve length of a scroll wrap (see Eq. 4.256 from Ian Bell thesis)
             """
            return self.geo.rb*(0.5*(phi_max**2-phi_min**2)-phi_0*(phi_max-phi_min))


        _slice = list(range(len(theta)))

        # Get the break angle (simplified solution)
        phi_s_sa = self.geo.phi_ooe-pi

        # Get the number of compression chambers in existence at each crank angle
        nC = np.array([scroll_geo.getNc(t,self.geo) for t in theta])

        F = np.zeros_like(self.p)
        F = F[:,_slice]

        # Parameters for the SA chamber
        ## Orbiting Scroll
        ds_SA_o = curve_length(phi_max = self.geo.phi_ooe,
                               phi_min = phi_s_sa,
                               phi_0 = self.geo.phi_oo0)
        ## Fixed Scroll
        ds_SA_f = curve_length(phi_max = self.geo.phi_foe,
                               phi_min = phi_s_sa,
                               phi_0 = self.geo.phi_fo0)
        ICV = self.CVs.index('sa')
        F[ICV,:] = (ds_SA_o+ds_SA_f)*self.geo.t/2*(self.p[ICV,_slice]-p_backpressure)

        # Parameters for the S1 chamber
        ## Orbiting Scroll
        ds_S1_o = curve_length(phi_max = phi_s_sa,
                               phi_min = self.geo.phi_ooe-pi-theta,
                               phi_0 = self.geo.phi_oo0)
        ## Fixed Scroll
        ds_S1_f = curve_length(phi_max = self.geo.phi_fie,
                               phi_min = self.geo.phi_fie-theta,
                               phi_0 = self.geo.phi_fi0)
        ICV = self.CVs.index('s1')
        F[ICV,:] = (ds_S1_o+ds_S1_f)*self.geo.t/2*(self.p[ICV,_slice]-p_backpressure)

        # Parameters for the S2 chamber
        ## Orbiting Scroll
        ds_S2_o = curve_length(phi_max = self.geo.phi_oie,
                               phi_min = self.geo.phi_oie-theta,
                               phi_0 = self.geo.phi_oi0)
        ## Fixed Scroll
        ds_S2_f = curve_length(phi_max = phi_s_sa,
                               phi_min = self.geo.phi_foe-pi-theta,
                               phi_0 = self.geo.phi_fo0)
        ICV = self.CVs.index('s2')
        F[ICV,:] = (ds_S2_o+ds_S2_f)*self.geo.t/2*(self.p[ICV,_slice]-p_backpressure)

        # Parameters for the C1.x and C2.x chambers
        for I in range(1, scroll_geo.nC_Max(self.geo)+1):
            ## Orbiting Scroll
            ds_C1_o = curve_length(phi_max = self.geo.phi_ooe-pi-theta-2*pi*(I-1),
                                   phi_min = self.geo.phi_ooe-pi-theta-2*pi*(I),
                                   phi_0 = self.geo.phi_oo0)
            ## Fixed Scroll
            ds_C1_f = curve_length(phi_max = self.geo.phi_fie-theta-2*pi*(I-1),
                                   phi_min = self.geo.phi_fie-theta-2*pi*(I),
                                   phi_0 = self.geo.phi_fi0)
            ICV = self.CVs.index('c1.'+str(I))
            F[ICV,:] = (ds_C1_o+ds_C1_f)*self.geo.t/2*(self.p[ICV, _slice]-p_backpressure)

            ## Orbiting Scroll
            ds_C2_o = curve_length(phi_max = self.geo.phi_oie-theta-2*pi*(I-1),
                                   phi_min = self.geo.phi_oie-theta-2*pi*(I),
                                   phi_0 = self.geo.phi_oi0)
            ## Fixed Scroll
            ds_C2_f = curve_length(phi_max = self.geo.phi_foe-pi-theta-2*pi*(I-1),
                                   phi_min = self.geo.phi_foe-pi-theta-2*pi*(I),
                                   phi_0 = self.geo.phi_fo0)
            ICV = self.CVs.index('c2.'+str(I))
            F[ICV,:] = (ds_C2_o+ds_C2_f)*self.geo.t/2*(self.p[ICV, _slice]-p_backpressure)

        near_theta_d = (np.abs(theta-scroll_geo.theta_d(self.geo))<1.e-8)

        # Parameters for the D1 chamber
        ## Orbiting Scroll
        phi2 = self.geo.phi_ooe-pi-theta-2*pi*(nC)
        phi1 = self.geo.phi_oos
        phi2[near_theta_d] = self.geo.phi_ooe-pi-theta[near_theta_d]-2*pi*(scroll_geo.nC_Max(self.geo)-1)
        ds_D1_o = curve_length(phi_max = phi2,
                               phi_min = phi1,
                               phi_0 = self.geo.phi_oo0)
        ## Fixed Scroll
        phi2 = self.geo.phi_fie-theta-2*pi*(nC)
        phi1 = self.geo.phi_oos+pi
        phi2[near_theta_d] = self.geo.phi_fie-theta[near_theta_d]-2*pi*(scroll_geo.nC_Max(self.geo)-1)
        ds_D1_f = curve_length(phi_max = phi2,
                               phi_min = phi1,
                               phi_0 = self.geo.phi_fi0)
        ICV = self.CVs.index('d1')
        F[ICV,:] = (ds_D1_o+ds_D1_f)*self.geo.t/2*(self.p[ICV,_slice]-p_backpressure)

        # Parameters for the D2 chamber
        ## Orbiting Scroll
        phi2 = self.geo.phi_oie-theta-2*pi*(nC)
        phi1 = self.geo.phi_fos + pi
        phi2[near_theta_d] = self.geo.phi_oie-theta[near_theta_d]-2*pi*(scroll_geo.nC_Max(self.geo)-1)
        ds_D2_o = curve_length(phi_max = phi2,
                               phi_min = phi1,
                               phi_0 = self.geo.phi_oi0)
        ## Fixed Scroll
        phi2 = self.geo.phi_foe-pi-theta-2*pi*(nC)
        phi1 = self.geo.phi_fos
        phi2[near_theta_d] = self.geo.phi_foe-pi-theta[near_theta_d]-2*pi*(scroll_geo.nC_Max(self.geo)-1)
        ds_D2_f = curve_length(phi_max = phi2,
                               phi_min = phi1,
                               phi_0 = self.geo.phi_fo0)
        ICV = self.CVs.index('d2')
        F[ICV,:] = (ds_D2_o+ds_D2_f)*self.geo.t/2*(self.p[ICV,_slice]-p_backpressure)

        # Parameters for the DD chamber
        ## Orbiting Scroll
        ds_DD_o = curve_length(phi_max = self.geo.phi_ois,
                               phi_min = self.geo.phi_oi0,
                               phi_0 = self.geo.phi_oi0)
        ## Fixed Scroll
        ds_DD_f = curve_length(phi_max = self.geo.phi_fis,
                               phi_min = self.geo.phi_fi0,
                               phi_0 = self.geo.phi_fi0)
        ICV = self.CVs.index('dd')
        F[ICV,:] = (ds_DD_o+ds_DD_f)*self.geo.t/2*(self.p[ICV,_slice]-p_backpressure)

        # Parameters for the DDD chamber
        ICV = self.CVs.index('ddd')
        F[ICV,:] = (ds_D1_o+ds_D2_o+ds_DD_o+ds_D1_f+ds_D2_f+ds_DD_f)*self.geo.t/2*(self.p[ICV,_slice]-p_backpressure)

        # Remove all the nan placeholders
        F[np.isnan(F)] = 0

        return F

    def calculate_force_terms(self,
                              orbiting_back_pressure=None):
        """
        Overwrite method 'calculate_force_terms' from Scroll class.

        Fix the calculation of the axial load
        """

        # Call original calculate_force_terms
        Scroll.calculate_force_terms(self, orbiting_back_pressure)

        _slice = list(range(self.Itheta+1))
        t = self.t[_slice]

        # Remove the absolute axial force generated by the gas at the top of the scroll wrap
        # This removes the contribution that was previously calculated in the 
        # calculate_force_terms of the base implementation
        self.forces.summed_Fz -= self.forces.summed_Faxial_involute # The old one

        # Add the net axial force generated by the gas at the top of the scroll wrap
        # This is the corrected implementation needed for the asymmetric scrolls
        self.forces.Faxial_involute = self.scroll_involute_axial_force(t, orbiting_back_pressure)
        self.forces.summed_Faxial_involute = np.sum(self.forces.Faxial_involute, axis = 0)
        self.forces.summed_Fz += self.forces.summed_Faxial_involute # Replace with the new one

        # Recalculate the mean axial force
        self.forces.mean_Fz = np.trapz(self.forces.summed_Fz, t)/(2*pi)

    def detailed_mechanical_analysis(self):
        """
        Overwrite method 'detailed_mechanical_analysis' from Scroll class.

        Add
        - OC_inertial : the inertial forces from the Oldham Coupling
        - max_Fr, max_Ft, max_Fz : the maximum of gas load
        - Fosb, Fumb, Flmb : The load on the OSB, UMB and LMB
        - Aosb, Aumb, Almb : The angle of the load on the OSB, UMB and LMB in the rotating frame (r,v,z)
                - r is the unit vector in the direction of the crankpin
                - v is the unit velocity vector of the OS
                - z is the axial direction from the bottom to the top
        - max_Fosb, max_Fumb, max_Flmb : maximum of loads on journal bearings
        TODO: Check the sign of friction on OC
        TODO: Take into account the repartition of inertia load between OSB and contacts between OS and FS wraps
        TODO: Rework losses from journal bearings
        """

        # Call original detailed_mechanical_analysis
        Scroll.detailed_mechanical_analysis(self)

        # Maximum of gas load
        self.forces.max_Fr = min(self.forces.summed_Fr) # Use of min because Fr<0
        self.forces.max_Ft = max(self.forces.summed_Ft)
        self.forces.max_Fz = max(self.forces.summed_Fz)

        # Recalculate parameters needed for loads on journal bearings
        _slice = list(range(self.Itheta+1))
        theta = self.t[_slice]
        THETA = self.geo.phi_fie - pi / 2 - theta
        muthrust = self.mech.thrust_friction_coefficient
        beta = self.mech.oldham_rotation_beta
        mu3 = mu4 = self.mech.oldham_key_friction_coefficient
        F1, F2, F3, F4 = self.forces.Fkey
        vOS_ybeta = -self.geo.ro * self.omega * (np.sin(THETA)*np.sin(beta) + np.cos(THETA)*np.cos(beta))
        aOR_xbeta =  self.geo.ro * self.omega**2*(-np.cos(THETA)*np.cos(beta)-np.sin(THETA)*np.sin(beta))
        PSI = vOS_ybeta / np.abs(vOS_ybeta)
        self.forces.OC_inertial = self.mech.oldham_mass*aOR_xbeta/1000

        # Calculate load on journal bearings in the rotating frame
        Fosb_r = - self.forces.summed_Fr - (F4-F3)*np.cos(THETA-beta) - PSI*(mu3*F3+mu4*F4)*np.sin(THETA-beta) - self.forces.inertial
        Fosb_v = + self.forces.summed_Ft - (F4-F3)*np.sin(THETA-beta) + PSI*(mu3*F3+mu4*F4)*np.cos(THETA-beta) + muthrust*self.forces.summed_Fz
        Fumb_r = -(Fosb_r+self.forces.inertial)*(1.0+1.0/self.mech.L_ratio_bearings)
        Fumb_v = -(Fosb_v)*(1.0+1.0/self.mech.L_ratio_bearings)
        Flmb_r = (Fosb_r+self.forces.inertial)/self.mech.L_ratio_bearings
        Flmb_v = (Fosb_v)/self.mech.L_ratio_bearings

        # Calculate the norm and the direction of the load on journal bearings
        self.forces.Fosb = np.sqrt(Fosb_r**2 + Fosb_v**2)
        self.forces.Fumb = np.sqrt(Fumb_r**2 + Fumb_v**2)
        self.forces.Flmb = np.sqrt(Flmb_r**2 + Flmb_v**2)
        self.forces.Aosb = np.angle(Fosb_r + Fosb_v*1j, deg=True)
        self.forces.Aumb = np.angle(Fumb_r + Fumb_v*1j, deg=True)
        self.forces.Almb = np.angle(Flmb_r + Flmb_v*1j, deg=True)

        # Mean values
        self.forces.mean_Fosb = np.trapz(self.forces.Fosb, theta) / (2*pi)
        self.forces.mean_Fumb = np.trapz(self.forces.Fumb, theta) / (2*pi)
        self.forces.mean_Flmb = np.trapz(self.forces.Flmb, theta) / (2*pi)
        self.forces.mean_Aosb = np.trapz(self.forces.Aosb, theta) / (2*pi)
        self.forces.mean_Aumb = np.trapz(self.forces.Aumb, theta) / (2*pi)
        self.forces.mean_Almb = np.trapz(self.forces.Almb, theta) / (2*pi)

        # Max values of loads
        self.forces.max_Fosb = max(self.forces.Fosb)
        self.forces.max_Fumb = max(self.forces.Fumb)
        self.forces.max_Flmb = max(self.forces.Flmb)

        # Recalculate F_B with Fosb
        self.forces.F_B = self.forces.Fosb

        # Recalculate the friction coefficient for each bearing
        for i in _slice:
            self.crank_bearing(self.forces.Fosb[i] * 1000)
            self.forces.mu_B[i] = self.losses.crank_bearing_dict['f']
            self.upper_bearing(self.forces.Fumb[i] * 1000)
            self.forces.mu_Bupper[i] = self.losses.upper_bearing_dict['f']
            self.lower_bearing(self.forces.Flmb[i] * 1000)
            self.forces.mu_Blower[i] = self.losses.lower_bearing_dict['f']

        # Update M_B, M_Bupper and M_Blower using Fosb, Fumb and Flmb
        self.forces.M_B      = self.forces.mu_B*self.mech.D_crank_bearing/2*self.forces.Fosb
        self.forces.M_Bupper = self.forces.mu_Bupper*self.mech.D_upper_bearing/2*self.forces.Fumb
        self.forces.M_Blower = self.forces.mu_Blower*self.mech.D_lower_bearing/2*self.forces.Flmb

        # Update Wdot_bearings
        self.forces.Wdot_OS_journal = np.abs(self.omega*self.forces.M_B)*self.mech.journal_tune_factor
        self.forces.Wdot_upper_journal = np.abs(self.omega*self.forces.M_Bupper)*self.mech.journal_tune_factor
        self.forces.Wdot_lower_journal = np.abs(self.omega*self.forces.M_Blower)*self.mech.journal_tune_factor

        # Update Wdot_total
        self.forces.Wdot_total = (self.forces.Wdot_F1
                                  + self.forces.Wdot_F2
                                  + self.forces.Wdot_F3
                                  + self.forces.Wdot_F4
                                  + self.forces.Wdot_OS_journal
                                  + self.forces.Wdot_upper_journal
                                  + self.forces.Wdot_lower_journal
                                  + self.forces.Wdot_thrust)

        self.forces.Wdot_total_mean = np.trapz(self.forces.Wdot_total, theta) / (2 * pi)
        print(self.forces.Wdot_total_mean, 'average mechanical losses')

    def build_volume_profile(self):
        """
        Build the volume profile, tracking along s1,c1.x,d1,ddd and
        s2,c2.x,d2,ddd and store them in the variables summary.theta_profile,
        summary.V1_profile, summary.V2_profile
        """

        # Calculate along one path to track one set of pockets through the whole process
        theta = self.t
        Vcopy = self.V.copy()

        # Suction chambers
        V1 = Vcopy[self.CVs.index('s1')]
        V2 = Vcopy[self.CVs.index('s2')]

        Nc_max1, Nc_max2 = self.Nc_max()

        assert len(theta) == len(V1) == len(V2)

        for path, Nc_max in zip([1,2],[Nc_max1, Nc_max2]):
            if Nc_max > 1:
                for alpha in range(1,Nc_max):
                    # Compression chambers up to the next-to-innermost set are handled
                    # just like the suction chambers
                    if path == 1:
                        theta = np.append(theta, self.t + 2*pi*alpha)
                        V1 = np.append(V1, Vcopy[self.CVs.index('c1.'+str(alpha))])
                    else:
                        V2 = np.append(V2, Vcopy[self.CVs.index('c2.'+str(alpha))])

            # Innermost compression chamber begins to be tricky
            # By definition innermost compression chamber doesn't make it to the
            # end of the rotation
            next_theta = self.t + 2*pi*Nc_max
            if path == 1:
                next_V1 = Vcopy[self.CVs.index('c1.'+str(Nc_max))]
                next_V1[np.isnan(next_V1)] = 0
            else:
                next_V2 = Vcopy[self.CVs.index('c2.'+str(Nc_max))]
                next_V2[np.isnan(next_V2)] = 0

        Vd1 = Vcopy[self.CVs.index('d1')]
        Vd2 = Vcopy[self.CVs.index('d2')]
        Vddd = Vcopy[self.CVs.index('ddd')]

        assert len(theta) == len(V1) == len(V2)

        # Now check if d1 and d2 end before the end of the rotation (they don't
        # neccessarily)
        if np.isnan(Vd1[0]) and np.isnan(Vd1[self.Itheta]):
            # d1 & d2 end before the end of the rotation
            # straightforward analysis (just add on pd1 and pd2)
            Vd1[np.isnan(Vd1)] = 0
            Vd2[np.isnan(Vd2)] = 0
            next_V1 += Vd1
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
            VdddA[0:i] = 0 # This is the end of the rotation
            next_V1 += VdddA
            next_V2 += VdddA

            theta = np.append(theta, next_theta)
            V1 = np.append(V1, next_V1)
            V2 = np.append(V2, next_V2)

            i = 0
            while i < len(VdddB):
                if np.isnan(VdddB[i]):
                    break;
                i += 1

            VdddB[i::] = np.nan # This is the beginning of the next rotation

            theta = np.append(theta, self.t + 2*pi*(Nc_max1 + 1))
            V1 = np.append(V1, VdddB)
            V2 = np.append(V2, VdddB)

        # Now check if d1 & d2 still exist at the end of the rotation
        elif not np.isnan(Vd1[0]) and not np.isnan(Vd1[self.Itheta]):
            # d1 & d2 don't end before the end of the rotation
            Vd1A = Vd1.copy()
            Vd1B = Vd1.copy()
            Vd2A = Vd2.copy()
            Vd2B = Vd2.copy()

            i = self.Itheta
            while i > 0:
                if np.isnan(Vd2A[i]):
                    i += 1
                    break;
                i -= 1
            Vd1A[0:i] = 0 # This is the end of the rotation
            Vd2A[0:i] = 0 # This is the end of the rotation
            next_V1 += Vd1A
            next_V2 += Vd2A

            theta = np.append(theta, next_theta)
            V1 = np.append(V1, next_V1)
            V2 = np.append(V2, next_V2)

            last_theta = self.t + 2*pi*(Nc_max1 + 1)
            last_V1 = Vddd.copy()
            last_V2 = Vddd.copy()
            last_V1[np.isnan(last_V1)] = 0
            last_V2[np.isnan(last_V2)] = 0

            i = 0
            while i < len(Vd1B):
                if np.isnan(Vd1B[i]):
                    break;
                i += 1
            if i == len(Vd1B)-1:
                raise ValueError('d1B could not find NaN')

            Vd1B[i::] = 0
            Vd2B[i::] = 0
            last_V1 += Vd1B
            last_V2 += Vd2B

            theta = np.append(theta, last_theta)
            V1 = np.append(V1, last_V1)
            V2 = np.append(V2, last_V2)

        self.summary.theta_profile = theta
        self.summary.V1_profile = V1
        self.summary.V2_profile = V2

        assert len(theta) == len(V1) == len(V2)

    def build_temperature_profile(self):
        """
        Build the temperature profile, tracking along s1,c1.x,d1,ddd and
        s2,c2.x,d2,ddd and store them in the variables summary.theta_profile,
        summary.T1_profile, summary.T2_profile
        """

        # Calculate along one path to track one set of pockets through the whole process
        theta = self.t
        Tcopy = self.T.copy()

        # Suction chambers
        T1 = Tcopy[self.CVs.index('s1')]
        T2 = Tcopy[self.CVs.index('s2')]

        Nc_max1, Nc_max2 = self.Nc_max()

        assert len(theta) == len(T1) == len(T2)

        for path, Nc_max in zip([1,2],[Nc_max1, Nc_max2]):
            if Nc_max > 1:
                for alpha in range(1,Nc_max):
                    # Compression chambers up to the next-to-innermost set are handled
                    # just like the suction chambers
                    if path == 1:
                        theta = np.append(theta, self.t + 2*pi*alpha)
                        T1 = np.append(T1, Tcopy[self.CVs.index('c1.'+str(alpha))])
                    else:
                        T2 = np.append(T2, Tcopy[self.CVs.index('c2.'+str(alpha))])

            # Innermost compression chamber begins to be tricky
            # By definition innermost compression chamber doesn't make it to the
            # end of the rotation
            next_theta = self.t + 2*pi*Nc_max
            if path == 1:
                next_T1 = Tcopy[self.CVs.index('c1.'+str(Nc_max))]
                next_T1[np.isnan(next_T1)] = 0
            else:
                next_T2 = Tcopy[self.CVs.index('c2.'+str(Nc_max))]
                next_T2[np.isnan(next_T2)] = 0

        Td1 = Tcopy[self.CVs.index('d1')]
        Td2 = Tcopy[self.CVs.index('d2')]
        Tddd = Tcopy[self.CVs.index('ddd')]

        assert len(theta) == len(T1) == len(T2)

        # Now check if d1 and d2 end before the end of the rotation (they don't
        # neccessarily)
        if np.isnan(Td1[0]) and np.isnan(Td1[self.Itheta]):
            # d1 & d2 end before the end of the rotation
            # straightforward analysis (just add on pd1 and pd2)
            Td1[np.isnan(Td1)] = 0
            Td2[np.isnan(Td2)] = 0
            next_T1 += Td1
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
            TdddA[0:i] = 0 # This is the end of the rotation
            next_T1 += TdddA
            next_T2 += TdddA

            theta = np.append(theta, next_theta)
            T1 = np.append(T1, next_T1)
            T2 = np.append(T2, next_T2)

            i = 0
            while i < len(TdddB):
                if np.isnan(TdddB[i]):
                    break;
                i += 1

            TdddB[i::] = np.nan # This is the beginning of the next rotation

            theta = np.append(theta, self.t + 2*pi*(Nc_max1 + 1))
            T1 = np.append(T1, TdddB)
            T2 = np.append(T2, TdddB)

        # Now check if d1 & d2 still exist at the end of the rotation
        elif not np.isnan(Td1[0]) and not np.isnan(Td1[self.Itheta]):
            # d1 & d2 don't end before the end of the rotation
            Td1A = Td1.copy()
            Td1B = Td1.copy()
            Td2A = Td2.copy()
            Td2B = Td2.copy()

            i = self.Itheta
            while i > 0:
                if np.isnan(Td2A[i]):
                    i += 1
                    break;
                i -= 1
            Td1A[0:i] = 0 # This is the end of the rotation
            Td2A[0:i] = 0 # This is the end of the rotation
            next_T1 += Td1A
            next_T2 += Td2A

            theta = np.append(theta, next_theta)
            T1 = np.append(T1, next_T1)
            T2 = np.append(T2, next_T2)

            last_theta = self.t + 2*pi*(Nc_max1 + 1)
            last_T1 = Tddd.copy()
            last_T2 = Tddd.copy()
            last_T1[np.isnan(last_T1)] = 0
            last_T2[np.isnan(last_T2)] = 0

            i = 0
            while i < len(Td1B):
                if np.isnan(Td1B[i]):
                    break;
                i += 1
            if i == len(Td1B)-1:
                raise ValueError('d1B could not find NaN')

            Td1B[i::] = 0
            Td2B[i::] = 0
            last_T1 += Td1B
            last_T2 += Td2B

            theta = np.append(theta, last_theta)
            T1 = np.append(T1, last_T1)
            T2 = np.append(T2, last_T2)

        self.summary.theta_profile = theta
        self.summary.T1_profile = T1
        self.summary.T2_profile = T2

        assert len(theta) == len(T1) == len(T2)

    def post_solve(self):

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

        p1 = self.summary.p1_profile[(self.summary.theta_profile>=2*np.pi) & (self.summary.theta_profile<=self.geo.phi_ie-self.geo.phi_os-np.pi)]
        p2 = self.summary.p2_profile[(self.summary.theta_profile>=2*np.pi) & (self.summary.theta_profile<=self.geo.phi_ie-self.geo.phi_os-np.pi)]
        V1 = self.summary.V1_profile[(self.summary.theta_profile>=2*np.pi) & (self.summary.theta_profile<=self.geo.phi_ie-self.geo.phi_os-np.pi)]
        V2 = self.summary.V2_profile[(self.summary.theta_profile>=2*np.pi) & (self.summary.theta_profile<=self.geo.phi_ie-self.geo.phi_os-np.pi)]

        self.summary.polytropic_coeff1 = compute_polytropic_coefficient(p1,V1)
        self.summary.polytropic_coeff2 = compute_polytropic_coefficient(p2,V2)


    def attach_HDF5_annotations(self, fName):
        """
        Add annotation for Danfoss Scroll terms

        Parameters
        ----------
        fName : string
            The file name for the HDF5 file that is to be used
        TODO: Addannotations for all new features of DanfossScroll (IDVs, dummy ports, etc.)
        """

        # Use the Scroll annotations
        Scroll.attach_HDF5_annotations(self,fName)

        # Add specific annotations for DanfossScroll
        attrs_dict = {
                '/forces/Fosb':'The force applied to the Orbiting Scroll Bearing [kN]',
                '/forces/Fumb':'The force applied to the Upper Main Bearing [kN]',
                '/forces/Flmb':'The force applied to the Lower Main Bearing [kN]',
                '/forces/Aosb':'The direction of the force applied to the Orbiting Scroll Bearing [deg]',
                '/forces/Aumb':'The direction of the force applied to the Upper Main Bearing [deg]',
                '/forces/Almb':'The direction of the force applied to the Lower Main Bearing [deg]',
                '/forces/mean_Fosb':'Mean of Fosb over one rotation [kN]',
                '/forces/mean_Fumb':'Mean of Fumb over one rotation [kN]',
                '/forces/mean_Flmb':'Mean of Flmb over one rotation [kN]',
                '/forces/mean_Aosb':'Mean of Aosb over one rotation [kN]',
                '/forces/mean_Aumb':'Mean of Aumb over one rotation [kN]',
                '/forces/mean_Almb':'Mean of Almb over one rotation [kN]',
                '/forces/max_Fosb':'Max of Fosb over one rotation [kN]',
                '/forces/max_Fumb':'Max of Fumb over one rotation [kN]',
                '/forces/max_Flmb':'Max of Flmb over one rotation [kN]',
                '/forces/max_Fr':'Max of Fr over one rotation [kN]',
                '/forces/max_Ft':'Max of Ft over one rotation [kN]',
                '/forces/max_Fz':'Max of Fz over one rotation [kN]',
                '/forces/OC_inertial': 'Magnitude of inertial force from Oldham Coupling (m*omega^2*r) [kN]',
                '/forces/inertial': 'Magnitude of inertial force from Orbiting Scroll (m*omega^2*r) [kN]',
                '/IDV_ports':'Information relative to Intermediate Discharge Valves'
                }

        import h5py
        hf = h5py.File(fName,'a')

        for k, v in attrs_dict.items():
            dataset = hf.get(k)
            if dataset is not None:
                dataset.attrs['note'] = v
        hf.close()

    def write_mech_csv(self,fName='test.csv'):
        """
        Write data relative to mechanics in a .csv file named fName.
        """
        import textwrap
        from getpass import getuser
        sep = ";"
        pdsim_version = modules["PDSim"].__version__
        danfosspdsim_version = modules["DanfossPDSim"].__version__ +"_"+ modules["DanfossPDSim"].__git_branch__ +"_"+ modules["DanfossPDSim"].__git_revision__
        coolprop_version = modules["CoolProp"].__version__
        # Operating conditions
        inlet_state_Post = self.sim.inlet_state.copy()
        outlet_state_Post = self.sim.outlet_state.copy()
            
        inlet_state_Post.update(dict(P=inlet_state_Post.p, Q=1))
        outlet_state_Post.update(dict(P=outlet_state_Post.p, Q=1))

        s = textwrap.dedent(
             """
                User{sep} {username}
                
                Version
                DanfossPDSim{sep} {danfosspdsim_version}
                PDSim{sep} {pdsim_version}
                CoolProp{sep} {coolprop_version}
                
                Condition Cycle
                Refrigerant{sep} {refrigerant}
                Evap Temperature{sep} {Tevap:.3f}{sep} [C]
                Cond Temperature{sep} {Tcond:.3f}{sep} [C]
                Superheat{sep} {SH:.3}{sep} [K]
                Evap Pressure{sep} {Pevap:.3f}{sep} [bar]
                Cond Pressure{sep} {Pcond:.3f}{sep} [bar]
                Average Polynomial Coefficient{sep} {gamma:.5f}{sep} [-]
                Lub Viscosity{sep} {mu_oil:.3f}{sep} [cP]
                Rotation Speed{sep} {rpm:.3f}{sep} [rpm]
                """.format(sep = sep,
                           username = getuser(),
                           danfosspdsim_version = danfosspdsim_version,
                           pdsim_version = pdsim_version,
                           coolprop_version = coolprop_version,
                           refrigerant = self.inlet_state.Fluid.decode('utf-8'),
                           Pevap = self.inlet_state.p*1e-2,
                           Pcond = self.outlet_state.p*1e-2,
                           Tevap = inlet_state_Post.T - 273.15,
                           Tcond = outlet_state_Post.T - 273.15,
                           gamma = (self.summary.polytropic_coeff1 + self.summary.polytropic_coeff2)/2,
                           SH = self.inlet_state.T - self.inlet_state_Post.T,
                           mu_oil = self.mech.mu_oil*1e3,
                           rpm = self.omega*60/(2*pi))).lstrip()

        phif0 = (self.geo.phi_fi0+self.geo.phi_fo0)/2
        s += textwrap.dedent(
                """
                FS Involute Geometry
                Generating Radius{sep} {GR:.3f}{sep} [mm]
                Thickness{sep} {ITHK:.3f}{sep} [mm]
                Flank height{sep} {h:.3f}{sep} [mm]
                U1{sep} {U1:.5f}{sep} [rad]
                U3{sep} {U3:.5f}{sep} [rad]
                U6{sep} {U6:.5f}{sep} [rad]
                """.format(sep = sep,
                           GR = self.geo.rb*1e3,
                           ITHK = self.geo.t*1e3,
                           h = self.geo.h*1e3,
                           U1 = self.geo.phi_fos-phif0,
                           U3 = self.geo.phi_fis-phif0,
                           U6 = self.geo.phi_fie-phif0))

        phio0 = (self.geo.phi_oi0+self.geo.phi_oo0)/2
        s += textwrap.dedent(
                """
                OS Involute Geometry
                Generating Radius{sep} {GR:.3f}{sep} [mm]
                Thickness{sep} {ITHK:.3f}{sep} [mm]
                Flank height{sep} {h:.3f}{sep} [mm]
                U1{sep} {U1:.5f}{sep} [rad]
                U3{sep} {U3:.5f}{sep} [rad]
                U6{sep} {U6:.5f}{sep} [rad]
                OS mass{sep} {mass:.3f}{sep} [kg]
                Inertial force{sep} {Fios:.3f}{sep} [N]
                """.format(sep=sep,
                           GR = self.geo.rb*1e3,
                           ITHK = self.geo.t*1e3,
                           h = self.geo.h*1e3,
                           U1 = self.geo.phi_oos-phio0,
                           U3 = self.geo.phi_ois-phio0,
                           U6 = self.geo.phi_oie-phio0,
                           mass = self.mech.orbiting_scroll_mass,
                           Fios = self.forces.inertial*1e3))

        vector_labels = [{'Crankshaft angle':'deg'},
                         {'Fr':'N'},
                         {'Ft':'N'},
                         {'Fz':'N'},
                         {'Torque':'N.m'},
                         {'Fosb':'N'},
                         {'Aosb':'N'},
                         {'Fumb':'N'},
                         {'Aumb':'N'},
                         {'Flmb':'N'},
                         {'Almb':'N}'
                         }]
        for k in range(self.CVs.N):
            vector_labels.append({'p_'+self.CVs.keys[k]:'MPa'})


        s += textwrap.dedent(
                """
                {vec_labels}
                {vec_units}
                """.format(vec_labels=sep.join([list(label.keys())[0] for label in vector_labels]),
                           vec_units=sep.join([list(label.values())[0] for label in vector_labels]))
                            )

        m = np.zeros([361,11+self.CVs.N])
        t_ = np.linspace(0.,2*pi,361)
        # Interpolation of vectors on t_
        Fr_   = np.interp(t_, self.forces.THETA[0]-self.forces.THETA,self.forces.summed_Fr)*1e3
        Ft_   = np.interp(t_, self.forces.THETA[0]-self.forces.THETA,self.forces.summed_Ft)*1e3
        Fz_   = np.interp(t_, self.forces.THETA[0]-self.forces.THETA,self.forces.summed_Fz)*1e3
        Mz_   = np.interp(t_, self.forces.THETA[0]-self.forces.THETA,self.forces.summed_Mz)*1e3
        Fosb_ = np.interp(t_, self.forces.THETA[0]-self.forces.THETA,self.forces.Fosb)*1e3
        Fumb_ = np.interp(t_, self.forces.THETA[0]-self.forces.THETA,self.forces.Fumb)*1e3
        Flmb_ = np.interp(t_, self.forces.THETA[0]-self.forces.THETA,self.forces.Flmb)*1e3
        Aosb_ = np.interp(t_, self.forces.THETA[0]-self.forces.THETA,self.forces.Aosb)
        Aumb_ = np.interp(t_, self.forces.THETA[0]-self.forces.THETA,self.forces.Aumb)
        Almb_ = np.interp(t_, self.forces.THETA[0]-self.forces.THETA,self.forces.Almb)
        p_ = np.zeros([361,self.CVs.N])
        for k in range(self.CVs.N):
            p_[:,k] = np.interp(t_, self.forces.THETA[0]-self.forces.THETA,self.p[k,:])*1e-3
        p_[np.isnan(p_)] = 0

        m[:,0] = t_*360./(2*pi)
        m[:,1] = Fr_
        m[:,2] = Ft_
        m[:,3] = Fz_
        m[:,4] = Mz_
        m[:,5] = Fosb_
        m[:,6] = Aosb_
        m[:,7] = Fumb_
        m[:,8] = Aumb_
        m[:,9] = Flmb_
        m[:,10]= Almb_
        m[:,11:] = p_

        with open(fName, 'w') as f:
            f.write(s)
            np.savetxt(f,m,delimiter='; ')


    @staticmethod
    def set_three_arc_disc(geo, r1, r2, alpha):
        """
        Calculate the three-arc discharge geometry as described by A. Picavet.
        This treatment is more general, and allows for a wider range of solutions.

        Parameters
        ----------
        r1 : double
        r2 : double
        alpha : double
        """
        # Point on the scroll involute
        xfis, yfis = scroll_geo.coords_inv(geo.phi_fis, geo, 0, 'fi')
        nxfis, nyfis = scroll_geo.coords_norm(geo.phi_fis, geo, 0, 'fi') # Points towards the scroll; will need to flip direction
        # Center of the first arc
        O1 = np.array([xfis, yfis]) - r1*np.array([nxfis[0], nyfis[0]])
        xO1, yO1 = O1

        # Point on the scroll involute
        xfos, yfos = scroll_geo.coords_inv(geo.phi_fos, geo, 0, 'fo')
        nxfos, nyfos = scroll_geo.coords_norm(geo.phi_fos, geo, 0, 'fo') # Points towards the scroll
        # Center of the second arc
        O2 = np.array([xfos, yfos]) + r2*np.array([nxfos[0], nyfos[0]])
        xO2, yO2 = O2

        # The point M1 on the circle
        M1 = np.array([xO1 + r1*np.cos(alpha), yO1 + r1*np.sin(alpha)])

        # Direction vector from center of arc #1 to point M1
        rM1 = np.array([np.cos(alpha), np.sin(alpha)])

        def objective(c):
            """
            The objective function to be solved numerically by moving the
            center of the arc 3 along the line between M1 and O1.  The 
            multiplication factor c should be positive at the end
            """

            O3 = O1 - c*rM1*r1
            r3 = norm(O3-M1)
            L = norm(O3-O2)
            return r3+r2-L

        c = scipy.optimize.newton(objective, 0)

        O3 = O1 - c*rM1*r1
        r3 = norm(O3-M1)
        xO3, yO3 = O3

        ################
        # Angle checking
        ################

        # The points taken in the order O1, O2, M1 must form an angle less than 
        # 90 degrees and O1->O2->M1 must be traversed in a counter-clockwise 
        # direction

        # Angle between the arcs from M1 to O1 and M1 to O2 (must be less 
        # than 90 degree)
        rM1O2 = O2-M1 # M1 to O1
        rM1O2 /= norm(rM1O2)
        rM1O1 = O1-M1 # M1 to O2
        rM1O1 /= norm(rM1O1)
        angle_between = np.arccos(np.dot(rM1O2, rM1O1))
        if angle_between > np.pi/2:
            raise ValueError("angle between [{0:g} radians] must be less than pi/2 [{1:g}] radians".format(angle_between, np.pi/2))

        # Points O1->O2->M1 must yield positive cross product (correct location)
        rO1O2 = O2-O1
        rO1O2 /= norm(rO1O2)
        rO2M1 = M1-O2
        rO2M1 /= norm(rO2M1)
        good_direction = np.cross(rO1O2,rO2M1) > 0
        if not good_direction:
            raise ValueError("O1->O2->M1 not traversed in CCW direction")

        # Set the arcs in the geo structure
        geo.xa_arc1 = xO1
        geo.ya_arc1 = yO1
        geo.ra_arc1 = r1
        geo.t1_arc1 = np.arctan2(M1[1] - yO1, M1[0] - xO1)
        geo.t2_arc1 = np.arctan2(yfis - yO1, xfis - xO1)
        geo.t1_arc1, geo.t2_arc1 = sortAnglesCCW(geo.t1_arc1, geo.t2_arc1)

        geo.xa_arc2 = xO2
        geo.ya_arc2 = yO2
        geo.ra_arc2 = r2
        geo.t1_arc2 = np.arctan2(yfos - yO2, xfos - xO2)
        geo.t2_arc2 = np.arctan2(yO3 - yO2, xO3 - xO2)
        geo.t1_arc2, geo.t2_arc2 = sortAnglesCCW(geo.t1_arc2, geo.t2_arc2)

        geo.xa_arc3 = xO3
        geo.ya_arc3 = yO3
        geo.ra_arc3 = r3
        geo.t1_arc3 = np.arctan2(M1[1] - yO3, M1[0] - xO3)
        geo.t2_arc3 = np.arctan2(yO2 - yO3, xO2 - xO3)
        geo.t1_arc3, geo.t2_arc3 = sortAnglesCCW(geo.t1_arc3, geo.t2_arc3)

        # Make a dummy line at end of arc1
        geo.b_line = 0
        geo.t1_line = geo.t2_line = geo.xa_arc2 + geo.ra_arc2*np.cos(geo.t1_arc2)
        geo.m_line = (geo.ya_arc2 + geo.ra_arc2*np.sin(geo.t1_arc2))/geo.t1_line

        # A little mini-script for plotting the curves for debugging purposes
        # for i in ['1','2','3']:
        #     t1,t2 = getattr(geo,'t1_arc'+i), getattr(geo,'t2_arc'+i)
        #     xa,ya = getattr(geo,'xa_arc'+i), getattr(geo,'ya_arc'+i)
        #     r = getattr(geo,'ra_arc'+i)
        #     t = np.linspace(t1, t2, 300)
        #     x, y = xa + r*np.cos(t), ya + r*np.sin(t)
        #     plt.plot(x,y)
        # plt.show()


if __name__=='__main__':
    pass
