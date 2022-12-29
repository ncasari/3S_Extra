# -*- coding: utf-8 -*-

import json
import traceback
import pdsim_plugins
from panels import pdsim_panels
import wx, textwrap
from wx.lib.scrolledpanel import ScrolledPanel
import os, sys
from math import pi, atan2
import numpy as np
import matplotlib.pyplot as plt
import copy
import six

from PDSim.scroll.plots import plotScrollSet
from PDSim.scroll import scroll_geo
from PDSim.scroll import common_scroll_geo
from PDSim.misc.datatypes import AnnotatedValue

import CoolProp.CoolProp as CP
from panels.pdsim_panels import PlotPanel, StatePanel
from panels.scroll_panels import InvoluteToCoords

class GeometryInputsPanel(pdsim_panels.PDPanel):
    desc_map = dict(disable_radial_suction=('Disable radial suction leakage flow', '-', True),
                    disable_flank_suction=('Disable flank suction leakage flow', '-', True)
                    )

    def __init__(self, parent):
        pdsim_panels.PDPanel.__init__(self, parent)

        self.keys_for_config = []
        # The sizer for all the objects
        sizer = wx.FlexGridSizer(cols=1, vgap=4, hgap=4)

        caption = textwrap.dedent(u"""
        This panel allows you to use the Danfoss definitions of the geometry 
        and convert them to PDSim compliant values.  This conversion is a 
        one-directional conversion, from this panel to the main geometry
        panel.  The values in the geometry panel will be over-written by the 
        values generated here.
        
        U3 = U1 + pi
        
        To convert to the involute angles of PDSim,
        
        phi_0 = -pi/2  # The initial angle for the centerline of the scroll wrap
        phi_i0 = phi_0 + ts/(rb*2)
        phi_o0 = phi_0 - ts/(rb*2)
        phi_os = U1 - pi/2
        phi_is = U1 + pi/2
        
        """
                                  )

        self.text = wx.StaticText(self, label=caption)
        self.U1_label = wx.StaticText(self, label='U1 [radian]')
        self.U1_value = wx.TextCtrl(self, value='1')
        self.ts_label = wx.StaticText(self, label='ts [m]')
        self.ts_value = wx.TextCtrl(self, value='0.005')
        self.rb_label = wx.StaticText(self, label='rb [m]')
        self.rb_value = wx.TextCtrl(self, value='0.003')

        self.GoButton = wx.Button(self, label='Go!')

        sizer.Add(self.text)
        sizer2 = wx.FlexGridSizer(cols=2, vgap=4, hgap=4)
        sizer2.Add(self.U1_label)
        sizer2.Add(self.U1_value)
        sizer2.Add(self.ts_label)
        sizer2.Add(self.ts_value)
        sizer2.Add(self.rb_label)
        sizer2.Add(self.rb_value)

        sizer.Add(sizer2)
        sizer.Add(self.GoButton)
        self.SetSizer(sizer)
        sizer.Layout()
        self.Refresh()

        self.GoButton.Bind(wx.EVT_BUTTON, self.OnGo)

    def OnGo(self, event=None):
        U1 = float(self.U1_value.GetValue())
        ts = float(self.ts_value.GetValue())
        rb = float(self.rb_value.GetValue())

        phi_0 = -pi / 2  # The initial angle for the centerline of the scroll wrap
        phi_i0 = phi_0 + ts / (rb * 2)
        phi_o0 = phi_0 - ts / (rb * 2)
        phi_os = U1 - pi / 2
        phi_is = U1 + pi / 2

        main = self.GetTopLevelParent()
        main.set_GUI_object_value('phi_fi0', '{p:0.16f}'.format(p=phi_i0))
        main.set_GUI_object_value('phi_fis', '{p:0.16f}'.format(p=phi_is))
        main.set_GUI_object_value('phi_fos', '{p:0.16f}'.format(p=phi_os))
        main.set_GUI_object_value('t', '{p:0.16f}'.format(p=ts))
        main.set_GUI_object_value('ro', '{p:0.16f}'.format(p=rb * pi - ts))

        dlg = wx.MessageDialog(None,
                               'You must still set the volume ratio and the displacement in the main geometry panel')
        dlg.ShowModal();
        dlg.Destroy()

        main.MTB.SetSelection(0)
        main.MTB.InputsTB.SetSelection(0)


class OtherInputsPanel(pdsim_panels.PDPanel):
    desc_map = dict(disable_radial_suction=('Disable radial suction leakage flow', '-', False),
                    disable_flank_suction=('Disable flank suction leakage flow', '-', False)
                    )

    def __init__(self, parent, config):
        pdsim_panels.PDPanel.__init__(self, parent)

        self.keys_for_config = []
        # The sizer for all the objects
        sizer = wx.FlexGridSizer(cols=2, vgap=4, hgap=4)

        annotated_values = self.get_annotated_values(['disable_radial_suction', 'disable_flank_suction'], config)

        # Build the items and return the list of annotated GUI objects
        annotated_GUI_objects = self.construct_items(annotated_values,
                                                     sizer=sizer,
                                                     parent=self)
        self.SetSizer(sizer)
        sizer.Layout()
        self.Refresh()

        self.GetTopLevelParent().register_GUI_objects(annotated_GUI_objects)


class TipSealLeakagePanel(pdsim_panels.PDPanel):
    desc_map = dict(use_tipseal=('Using tip seal model', '', False),
                    delta_axial=('Axial gap between scrolls [m]', 'm', 150e-6),
                    w_bypass=('Width of each bypass flowpath [m]', 'm', 0.001),
                    Xd_bypass=('Flow coefficient bypass [-]', '-', 0.7),
                    Xd_Fanno=('Tip seal slot flow coefficient for Fanno/Nozzle [-]', '-', 0.7),
                    w_slot=('Width of slot for tipseal [-]', '-', 0.003),
                    delta_slot=('Gap between seal and slot floor [m]', 'm', 0.00025),
                    fF_slot=('Fanning friction factor in slot [-]', '-', 0.001),
                    )

    def __init__(self, parent, config):
        pdsim_panels.PDPanel.__init__(self, parent)

        self.keys_for_config = []
        # The sizer for all the objects
        sizer_for_tipseal_inputs = wx.FlexGridSizer(cols=2, vgap=4, hgap=4)

        annotated_values = self.get_annotated_values(
            ['use_tipseal', 'delta_axial', 'w_bypass', 'Xd_bypass', 'Xd_Fanno', 'w_slot', 'delta_slot', 'fF_slot'],
            config)

        # Build the items and return the list of annotated GUI objects
        annotated_GUI_objects = self.construct_items(annotated_values,
                                                     sizer=sizer_for_tipseal_inputs,
                                                     parent=self)

        self.OptionFanno = wx.RadioButton(self, label='Fanno')
        self.OptionNozzle = wx.RadioButton(self, label='Isentropic Nozzle')
        sizer_for_tipseal_inputs.Add(wx.StaticText(self, label='Slot flow model:'))
        sizer_for_tipseal_inputs.Add(self.OptionFanno)
        sizer_for_tipseal_inputs.AddSpacer(10)
        sizer_for_tipseal_inputs.Add(self.OptionNozzle)
        sizer_for_tipseal_inputs.AddSpacer(10)
        self.OptionNozzle.SetValue(True)
        self.SetSizer(sizer_for_tipseal_inputs)
        sizer_for_tipseal_inputs.Layout()
        self.Refresh()

        self.GetTopLevelParent().register_GUI_objects(annotated_GUI_objects)
        self.usingTipSeal = self.GetTopLevelParent().get_GUI_object('use_tipseal').GUI_location
        self.usingTipSeal.Bind(wx.EVT_CHECKBOX, self.OnChangeUsage)
        self.OnChangeUsage()

    def OnChangeUsage(self, event=None):
        isEnabled = self.usingTipSeal.GetValue()

        for term in ['delta_axial', 'w_bypass', 'Xd_bypass', 'Xd_Fanno', 'w_slot', 'delta_slot', 'fF_slot']:
            self.GetTopLevelParent().get_GUI_object(term).GUI_location.Enable(isEnabled)
            self.OptionFanno.Enable(isEnabled)
            self.OptionNozzle.Enable(isEnabled)


class UASuctDiscPanel(pdsim_panels.PDPanel):
    desc_map = dict(use_UAsuct=('Using heat transfer from discharge to suction', '', False),
                    UA_suct_disc=('UA between suction and discharge temperatures [kW/K]', 'kW/K', 0.015))

    def __init__(self, parent, config):
        pdsim_panels.PDPanel.__init__(self, parent)

        self.keys_for_config = []
        # The sizer for all the objects
        sizer_for_UA_inputs = wx.FlexGridSizer(cols=2, vgap=4, hgap=4)

        annotated_values = self.get_annotated_values(['use_UAsuct', 'UA_suct_disc'], config)

        # Build the items and return the list of annotated GUI objects
        annotated_GUI_objects = self.construct_items(annotated_values,
                                                     sizer=sizer_for_UA_inputs,
                                                     parent=self)
        self.SetSizer(sizer_for_UA_inputs)
        sizer_for_UA_inputs.Layout()
        self.Refresh()

        self.GetTopLevelParent().register_GUI_objects(annotated_GUI_objects)
        self.usingUAsuct = self.GetTopLevelParent().get_GUI_object('use_UAsuct').GUI_location
        self.usingUAsuct.Bind(wx.EVT_CHECKBOX, self.OnChangeUsage)
        self.OnChangeUsage()

    def OnChangeUsage(self, event=None):
        isEnabled = self.usingUAsuct.GetValue()

        for term in ['UA_suct_disc']:
            self.GetTopLevelParent().get_GUI_object(term).GUI_location.Enable(isEnabled)


class DummyPortPanel(pdsim_panels.PDPanel):
    desc_map = dict(use_dummy=('Using Dummy Port', '', False),
                    h_dummy_port=('Dummy port depth [m]', 'm', 0.003),
                    X_d_dummy=('Flow coefficient for dummy port [-]', '-', 0.7),
                    )

    def __init__(self, parent, config):
        pdsim_panels.PDPanel.__init__(self, parent)

        self.keys_for_config = []
        # The sizer for all the objects on the disableable panel
        sizer_for_inputs = wx.FlexGridSizer(cols=2, vgap=4, hgap=4)

        annotated_values = self.get_annotated_values(['use_dummy', 'h_dummy_port', 'X_d_dummy'], config)

        # Build the items and return the list of annotated GUI objects
        annotated_GUI_objects = self.construct_items(annotated_values,
                                                     sizer=sizer_for_inputs,
                                                     parent=self)
        self.SetSizer(sizer_for_inputs)
        sizer_for_inputs.Layout()
        self.Refresh()

        self.GetTopLevelParent().register_GUI_objects(annotated_GUI_objects)
        self.usingDummy = self.GetTopLevelParent().get_GUI_object('use_dummy').GUI_location
        self.usingDummy.Bind(wx.EVT_CHECKBOX, self.OnChangeUsage)
        self.OnChangeUsage()

    def OnChangeUsage(self, event=None):
        isEnabled = self.usingDummy.GetValue()

        for term in ['h_dummy_port', 'X_d_dummy']:
            self.GetTopLevelParent().get_GUI_object(term).GUI_location.Enable(isEnabled)


class IntermediateDischargePanel(pdsim_panels.PDPanel):
    desc_map = dict(use_IDV=('Using IDV', '', False),
                    IDIDVTube=('Inner diameter of IDV tube [m]', 'm', 0.008),
                    LIDVTube=('Length of IDV tube [m]', 'm', 0.044),
                    IDIDVTap=('Inner diameter of each IDV tap [m]', 'm', 0.005),
                    Vdisc_plenum=('Volume of discharge plenum [m\xb3]', 'm^3', 0.0005),
                    Xd_IDVCV_plenum=('Flow coefficient between IDV Tube and plenum [-]', '-', 0.75),
                    offset_angle1=('Angle between Oldham slot and outer involute ports [deg]', '-', 60.29),
                    offset_angle2=('Angle between inner involute ports and symmetry [deg]', '-', 9.53),
                    offset_distance=('Centerline distance for pair of ports [m]', '-', 0.00732),
                    offset_outer=('Angle between 1st and 2nd pair of outer ports[deg]', '-', 0),
                    # added by Alain Picavet
                    offset_inner=('Angle between 1st and 2nd pair of inner ports[deg]', '-', 0),
                    # added by Alain Picavet
                    )

    def __init__(self, parent, config):
        pdsim_panels.PDPanel.__init__(self, parent)

        self.keys_for_config = []
        # The sizer for all the objects
        sizer_for_inputs = wx.FlexGridSizer(cols=2, vgap=4, hgap=4)

        annotated_values = self.get_annotated_values(
            ['use_IDV', 'IDIDVTap', 'IDIDVTube', 'LIDVTube', 'Vdisc_plenum', 'Xd_IDVCV_plenum', 'offset_angle1',
             'offset_angle2', 'offset_distance', 'offset_outer', 'offset_inner'], config)  # last parameters added by Alain Picavet

        # Build the items and return the list of annotated GUI objects
        annotated_GUI_objects = self.construct_items(annotated_values,
                                                     sizer=sizer_for_inputs,
                                                     parent=self)

        # The plot of the scroll wraps
        self.PP = PlotPanel(self)
        self.ax = self.PP.figure.add_axes((0, 0, 1, 1))

        sizer_for_inputs.Add(self.PP)
        self.SetSizer(sizer_for_inputs)
        sizer_for_inputs.Layout()
        self.Refresh()

        self.GetTopLevelParent().register_GUI_objects(annotated_GUI_objects)
        self.usingIDV = self.GetTopLevelParent().get_GUI_object('use_IDV').GUI_location
        self.usingIDV.Bind(wx.EVT_CHECKBOX, self.OnChangeUsage)
        self.OnChangeUsage()

    def OnChangeUsage(self, event=None):

        from DanfossPDSim.core import IDVPort

        isEnabled = self.usingIDV.GetValue()

        for term in ['IDIDVTap', 'IDIDVTube', 'LIDVTube', 'Vdisc_plenum', 'Xd_IDVCV_plenum', 'offset_angle1',
                     'offset_angle2', 'offset_distance', 'offset_outer', 'offset_inner']:
            self.GetTopLevelParent().get_GUI_object(term).GUI_location.Enable(isEnabled)

        self.PP.Show(isEnabled)

        if isEnabled:
            def get(term):
                return self.GetTopLevelParent().get_GUI_object_value(term)

            Scroll = self.GrandParent.panels_dict['GeometryPanel'].Scroll

            # Midpoint of outer wraps ports
            phim = get('offset_angle1') / 180 * pi + pi
            dphi = get('offset_distance') / (2 * Scroll.geo.rb * (phim - Scroll.geo.phi_fo0))
            off_out = get('offset_outer') / 180 * pi
            off_in = get('offset_inner') / 180 * pi
            # These ports are connected to c2.1 or c2.2 or d2
            p = IDVPort()
            p.phi = phim - dphi
            p.involute = 'o'
            p.offset = Scroll.geo.t / 2.0
            p.D = get('IDIDVTap')
            ports = [p]
            p = IDVPort()
            p.phi = phim + dphi
            p.involute = 'o'
            p.offset = Scroll.geo.t / 2.0
            p.D = get('IDIDVTap')
            ports.append(p)
            if off_out != 0:  # added by Alain Picavet
                p = IDVPort()
                p.phi = phim + off_out - dphi
                p.involute = 'o'
                p.offset = Scroll.geo.t / 2.0
                p.D = get('IDIDVTap')
                ports.append(p)
                p = IDVPort()
                p.phi = phim + off_out + dphi
                p.involute = 'o'
                p.offset = Scroll.geo.t / 2.0
                p.D = get('IDIDVTap')
                ports.append(p)

            # These two ports are connected to c1.1 or c1.2 or d1
            p = IDVPort()
            p.phi = phim - dphi + pi + get('offset_angle2') / 180 * pi
            p.involute = 'i'
            p.offset = Scroll.geo.t / 2.0
            p.D = get('IDIDVTap')
            ports.append(p)
            p = IDVPort()
            p.phi = phim + dphi + pi + get('offset_angle2') / 180 * pi
            p.involute = 'i'
            p.offset = Scroll.geo.t / 2.0
            p.D = get('IDIDVTap')
            ports.append(p)
            if off_out != 0:  # added by Alain Picavet
                p = IDVPort()
                p.phi = phim + off_in - dphi
                p.involute = 'i'
                p.offset = Scroll.geo.t / 2.0
                p.D = get('IDIDVTap')
                ports.append(p)
                p = IDVPort()
                p.phi = phim + off_in + dphi
                p.involute = 'i'
                p.offset = Scroll.geo.t / 2.0
                p.D = get('IDIDVTap')
                ports.append(p)

            self.ax.cla()

            plotScrollSet(pi / 4.0,
                          axis=self.ax,
                          geo=Scroll.geo,
                          offsetScroll=Scroll.geo.phi_ie_offset > 0)

            for port in ports:
                x, y = scroll_geo.coords_inv(port.phi, Scroll.geo, 0, 'f' + port.involute)
                nx, ny = scroll_geo.coords_norm(port.phi, Scroll.geo, 0, 'f' + port.involute)

                x0 = x - nx * port.offset
                y0 = y - ny * port.offset

                t = np.linspace(0, 2 * pi)
                self.ax.plot(port.D / 2.0 * np.cos(t) + x0, port.D / 2.0 * np.sin(t) + y0, 'b')

        self.PP.canvas.draw()
        self.Layout()

class IDVMainPanel(pdsim_panels.PDPanel):
    def __init__(self, parent, fluid, *, config=None):
        pdsim_panels.PDPanel.__init__(self, parent)

        self.vertsizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.vertsizer)
        self.inputssizer = wx.FlexGridSizer(cols=2, vgap=4, hgap=4)
        self.vertsizer.Add(self.inputssizer)

        self.annotated_items = self.construct_items(
            [
            AnnotatedValue('IDV'+str(id(self))+'volume', 0.0005, "Volume of IDV tube [m\xb3]", "m^3"),
            AnnotatedValue('IDV'+str(id(self))+'area', 0.003, "Flow area with plenum [m\xb2]", "m^2")
            ],
            sizer=self.inputssizer,
            parent=self
        )

        self.state = StatePanel(self, CPState=CP.State(fluid, dict(T=300,P=100)), Fluid_fixed=True)
        self.vertsizer.Add(self.state)

        AddPortButton = wx.Button(self, label='Add a Port')
        self.vertsizer.Add(AddPortButton)
        AddPortButton.Bind(wx.EVT_BUTTON, self.OnAddPort)

        RemoveButton = wx.Button(self, label='Remove This IDV')
        self.vertsizer.Add(RemoveButton)
        RemoveButton.Bind(wx.EVT_BUTTON, self.GrandParent.OnRemoveIDV)

        self.SetBackgroundColour(wx.YELLOW)

        self.Refresh()

        if config:
            self.set_from_struct(config)

    def OnAddPort(self, event=None):
        if event is not None: event.Skip()
        NewPort = GenericPortPanel(self, prefix="IDV")
        self.vertsizer.Add(NewPort)
        self.vertsizer.Layout()
        self.Refresh()
        self.GetParent().Layout()
        self.GrandParent.Refresh()
        self.GrandParent.Layout()
        self.GrandParent.OnReplot()
        return NewPort

    def get_struct(self):
        def get(term):
            for item in self.annotated_items:
                if item.key == term:
                    return float(item.GetValue())
        V_tube = get('IDV'+str(id(self))+'volume')
        Area = get('IDV'+str(id(self))+'area')
        state = self.state.GetState()
        return {
            'key': 'IDV.' + str(id(self)),
            'fluid': state.Fluid.decode('ascii'),
            'T / K': state.T,
            'p / kPa': state.p,
            'volume / m^3': V_tube,
            'plenum_flowarea / m^2': Area,
            'ports': [port.get_struct() for port in self.GetChildren() if isinstance(port, GenericPortPanel)]
        }

    def set_from_struct(self, config):
        """ Take in a JSON-like structure and use it to populate this class """
        def setter(term, val):
            for item in self.annotated_items:
                if item.key == term:
                    item.SetValue(str(val))
                    return
            raise KeyError(term)
        vol = setter('IDV'+str(id(self))+'volume', config['volume / m^3'])
        area = setter('IDV'+str(id(self))+'area', config['plenum_flowarea / m^2'])
        self.state.SetState(CP.State(config['fluid'], dict(T=config['T / K'], P=config['p / kPa'])))
        for port in config['ports']:
            NewPort = self.OnAddPort()
            NewPort.set_from_struct(port)

class IDVPlenumPanel(pdsim_panels.PDPanel):
    def __init__(self, parent, fluid, *, config=None):
        pdsim_panels.PDPanel.__init__(self, parent)

        self.vertsizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.vertsizer)
        self.inputssizer = wx.FlexGridSizer(cols=2, vgap=4, hgap=4)
        self.vertsizer.Add(self.inputssizer)

        self.annotated_items = self.construct_items(
            [
            AnnotatedValue('IDVplenumvolume', 0.005, "Volume of plenum [m\xb3]", "m^3"),
            AnnotatedValue('IDVplenumarea', 0.003, "Flow area with discharge [m\xb2]", "m^2")
            ],
            sizer=self.inputssizer,
            parent=self
        )

        self.state = StatePanel(self, CPState=CP.State(fluid, dict(T=300,P=100)), Fluid_fixed=True)
        self.vertsizer.Add(self.state)

        self.SetBackgroundColour(wx.GREEN)
        self.Refresh()

        if config:
            self.set_from_struct(config)

    def get_struct(self):
        def get(term):
            for item in self.annotated_items:
                if item.key == term:
                    return float(item.GetValue())
        vol = get('IDVplenumvolume')
        area = get('IDVplenumarea')
        state = self.state.GetState()
        return {
            'key': 'discharge_plenum',
            'volume / m^3': vol,
            'tube_node': 'outlet.1',
            'tube_flowarea / m^2': area,
            'fluid': state.Fluid.decode('ascii'),
            'T / K': state.T,
            'p / kPa': state.p
        }

    def set_from_struct(self, config):
        """ Take in a JSON-like structure and use it to populate this class """
        def setter(term, val):
            for item in self.annotated_items:
                if item.key == term:
                    item.SetValue(str(val))
                    return
            raise KeyError(term)
        setter('IDVplenumvolume', config['volume / m^3'])
        setter('IDVplenumarea', config['tube_flowarea / m^2'])
        self.state.SetState(CP.State(config['fluid'], dict(T=config['T / K'], P=config['p / kPa'])))

class GenericPortPanel(pdsim_panels.PDPanel):
    def __init__(self, parent, *, prefix, config=None):
        """
        prefix: a string prefix indicating what sort of a port this is. Could be inj, or IDV
        """
        pdsim_panels.PDPanel.__init__(self, parent)
        self.prefix = prefix

        self.vertsizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.vertsizer)
        self.inputssizer = wx.FlexGridSizer(cols=2, vgap=4, hgap=4)
        self.vertsizer.Add(self.inputssizer)

        self.GeoButton = wx.Button(self, label="From involute")
        self.inputssizer.Add(self.GeoButton)
        self.GeoButton.Bind(wx.EVT_BUTTON, self.OnSelectXY)
        self.inputssizer.AddSpacer(0)

        self.annotated_items = self.construct_items(
            [
            AnnotatedValue(self.prefix+'port'+str(id(self))+'x', -1, "x [m]", "m"),
            AnnotatedValue(self.prefix+'port'+str(id(self))+'y', -1, "y [m]", "m"),
            AnnotatedValue(self.prefix+'port'+str(id(self))+'D', -1, "Diameter [m]", "m")
            ],
            sizer=self.inputssizer,
            parent=self
        )

        self.Layout()
        self.Refresh()
        self.SetBackgroundColour(wx.RED)

        if config:
            self.set_from_struct(config)

    def setter(self, term, val):
            for item in self.annotated_items:
                if item.key == term:
                    item.SetValue(str(val))
                    return
            raise KeyError(term)

    def OnSelectXY(self, event):
        ITB = self.GetTopLevelParent().MTB.InputsTB
        Scroll = ITB.panels_dict['GeometryPanel'].Scroll
        dlg = InvoluteToCoords(None)
        if dlg.ShowModal() == wx.ID_OK:
            inv = dlg.inv.GetStringSelection()
            phi = float(dlg.angle.GetValue())
            offset = float(dlg.offset.GetValue())
            
            key_dict = {'Orbiting Inner': 'oi', 'Orbiting Outer':'oo','Fixed Inner':'fi','Fixed Outer':'fo'}
            xinv, yinv = scroll_geo.coords_inv(phi, Scroll.geo, 0, key_dict[inv])
            nxinv, nyinv = scroll_geo.coords_norm(phi, Scroll.geo, 0, key_dict[inv])
            
            x = xinv - nxinv[0]*offset
            y = yinv - nyinv[0]*offset

            self.setter(self.prefix+'port'+str(id(self))+'x', x)
            self.setter(self.prefix+'port'+str(id(self))+'y', y)
        dlg.Destroy()
        self.GrandParent.Parent.OnReplot()

    def get_struct(self):
        def get(term):
            for item in self.annotated_items:
                if item.key == term:
                    return float(item.GetValue())
        return {'x / m': get(self.prefix+'port'+str(id(self))+'x'), 
                'y / m': get(self.prefix+'port'+str(id(self))+'y'), 
                'D / m': get(self.prefix+'port'+str(id(self))+'D')}

    def set_from_struct(self, config):
        """ Take in a JSON-like structure and use it to populate this class """
        self.setter(self.prefix+'port'+str(id(self))+'x', config['x / m'])
        self.setter(self.prefix+'port'+str(id(self))+'y', config['y / m'])
        self.setter(self.prefix+'port'+str(id(self))+'D', config['D / m'])

class AdvancedIntermediateDischargePanel(pdsim_panels.PDPanel):
    desc_map = dict()

    def __init__(self, parent, config):
        pdsim_panels.PDPanel.__init__(self, parent)

        #Now we are going to put everything into a scrolled window
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        self.scrolled_panel = ScrolledPanel(self, size=(-1,-1),
                                 style = wx.TAB_TRAVERSAL, name="panel1")
        self.scrolled_panel.SetScrollbars(1,1,1,1)
        self.scrolled_panel.SetupScrolling()

        self.keys_for_config = []
        # The sizer for all the objects
        self.sizer_for_inputs = wx.FlexGridSizer(cols=1, vgap=4, hgap=4)

        # The plot of the scroll wraps
        from PDSim.plot.plots import Plot
        self.PP = Plot(self.scrolled_panel)
        self.ax = self.PP.figure.add_axes((0, 0, 1, 1))
        self.PP.canvas.draw()
        self.sizer_for_inputs.Add(self.PP)

        self.AddIDVButton = wx.Button(self.scrolled_panel, label='Add an IDV')
        self.sizer_for_inputs.Add(self.AddIDVButton)
        self.AddIDVButton.Bind(wx.EVT_BUTTON, self.OnAddIDV)

        self.DisplayJSONButton = wx.Button(self.scrolled_panel, label='Print JSON')
        self.sizer_for_inputs.Add(self.DisplayJSONButton)
        self.DisplayJSONButton.Bind(wx.EVT_BUTTON, self.OnDisplayJSON)

        self.sizer_for_inputs.Layout()
        self.Refresh()

        self.scrolled_panel.SetSizer(self.sizer_for_inputs)
        main_sizer.Add(self.scrolled_panel,1,wx.EXPAND)
        self.SetSizer(main_sizer)
        main_sizer.Layout()
        self.main_sizer = main_sizer

        if config:
            try:
                self.set_from_struct(config)
            except BaseException as BE:
                traceback.print_tb(BE.__traceback__)
                print(BE)

    def GetGeo(self):
        ITB = self.GetTopLevelParent().MTB.InputsTB
        Scroll = ITB.panels_dict['GeometryPanel'].Scroll
        return Scroll.geo

    def GetState(self):
        """ 
        Get the state class used in the inputs toolbook 
        """

        # Inputs Toolbook
        ITB = self.GetTopLevelParent().MTB.InputsTB
        CPState = None
        for panel in ITB.panels:
            if panel.Name == 'StatePanel':
                CPState = panel.SuctionStatePanel.GetState()
                break
        if CPState is None:
            raise ValueError('StatePanel not found in Inputs Toolbook')
        
        return CPState

    def OnDisplayJSON(self, event=None):
        j = self.get_struct()
        print(json.dumps(j, indent=2))

    def GetActiveIDVPanels(self):
        return [child for child in self.scrolled_panel.GetChildren() if (isinstance(child, IDVMainPanel) and child.IsShown())]

    def GetActivePlenums(self):
        return [child for child in self.scrolled_panel.GetChildren() if (isinstance(child, IDVPlenumPanel) and child.IsShown())]

    def OnAddIDV(self, event=None):
        """
        Add an IDV to the GUI
        """
        if event is not None: event.Skip()

        # Add a plenum if one is not already present
        num_IDV = len(self.GetActiveIDVPanels())
        if num_IDV == 0:
            NewPlenum = IDVPlenumPanel(self.scrolled_panel, fluid=self.GetState().Fluid)
            self.sizer_for_inputs.Add(NewPlenum)

        NewIDV = IDVMainPanel(self.scrolled_panel, fluid=self.GetState().Fluid)
        self.sizer_for_inputs.Add(NewIDV)
        self.sizer_for_inputs.Layout()
        self.main_sizer.Layout()
        self.OnReplot()
        return NewIDV

    def OnRemoveIDV(self, event=None):
        """
        Remove an IDV from the GUI
        """
        if event is not None: event.Skip()        

        old_num_IDV = len(self.GetActiveIDVPanels())
        IDV = event.GetEventObject().Parent
        assert(isinstance(IDV, IDVMainPanel))
        self.sizer_for_inputs.Detach(IDV)
        IDV.Hide()
        if old_num_IDV == 1:
            plenums = self.GetActivePlenums()
            assert(len(plenums) == 1)
            self.sizer_for_inputs.Detach(plenums[0])
            plenums[0].Hide()
        self.sizer_for_inputs.Layout()
        self.main_sizer.Layout()
        self.OnReplot()

    def get_struct(self):
        """
        Get the nested data structure needed to define the IDVs completely
        """
        IDVs = self.GetActiveIDVPanels()
        if len(IDVs) == 0:
            return {}

        plenum = self.GetActivePlenums()[0]
        return {
            'plenum': plenum.get_struct(), 
            'IDVs': [IDV.get_struct() for IDV in IDVs]
        }

    def set_from_struct(self, config):
        if config and 'IDVs' in config:
            for IDV in config['IDVs']:
                NewIDV = self.OnAddIDV()
                NewIDV.set_from_struct(IDV)
            plenum = self.GetActivePlenums()[0]
            plenum.set_from_struct(config['plenum'])

    def OnReplot(self, event=None):

        self.ax.cla()

        s = self.get_struct()
        if s and 'IDVs' in s:
            geo = self.GetGeo()
            plotScrollSet(pi/4.0,
                          axis=self.ax,
                          geo=geo,
                          offsetScroll=geo.phi_ie_offset > 0)

            for IDV in s['IDVs']:
                for port in IDV['ports']:
                    t = np.linspace(0, 2 * pi)
                    r = port['D / m']/2.0
                    self.ax.plot(r*np.cos(t) + port['x / m'], r*np.sin(t) + port['y / m'], 'b')

        self.PP.canvas.draw()
        self.Layout()


class InjectionMainPanel(pdsim_panels.PDPanel):
    def __init__(self, parent, fluid, *, config=None):
        pdsim_panels.PDPanel.__init__(self, parent)

        self.vertsizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.vertsizer)
        self.inputssizer = wx.FlexGridSizer(cols=2, vgap=4, hgap=4)
        self.vertsizer.Add(self.inputssizer)

        self.annotated_items = self.construct_items(
            [
            AnnotatedValue('Inj'+str(id(self))+'length', 0.1, "Length of injection tube [m]", "m"),
            AnnotatedValue('Inj'+str(id(self))+'ID', 0.008, "Int. dia. of inj. tube [m]", "m")
            ],
            sizer=self.inputssizer,
            parent=self
        )

        self.state = StatePanel(self, CPState=CP.State(fluid, dict(T=300,P=100)), Fluid_fixed=True)
        self.vertsizer.Add(self.state)

        AddPortButton = wx.Button(self, label='Add a Port')
        self.vertsizer.Add(AddPortButton)
        AddPortButton.Bind(wx.EVT_BUTTON, self.OnAddPort)

        RemoveButton = wx.Button(self, label='Remove This line')
        self.vertsizer.Add(RemoveButton)
        RemoveButton.Bind(wx.EVT_BUTTON, self.GrandParent.OnRemoveLine)

        self.SetBackgroundColour(wx.YELLOW)

        self.Refresh()

        if config:
            self.set_from_struct(config)

    def OnAddPort(self, event=None):
        if event is not None: event.Skip()
        NewPort = GenericPortPanel(self, prefix="Inj")
        self.vertsizer.Add(NewPort)
        self.vertsizer.Layout()
        self.Refresh()
        self.GetParent().Layout()
        self.GrandParent.Refresh()
        self.GrandParent.Layout()
        self.GrandParent.OnReplot()
        return NewPort

    def get_struct(self):
        def get(term):
            for item in self.annotated_items:
                if item.key == term:
                    return float(item.GetValue())
        L_tube = get('Inj'+str(id(self))+'length')
        ID_tube = get('Inj'+str(id(self))+'ID')
        state = self.state.GetState()
        return {
            'key': 'Inj.' + str(id(self)),
            'fluid': state.Fluid.decode('ascii'),
            'T / K': state.T,
            'p / kPa': state.p,
            'length / m': L_tube,
            'ID / m': ID_tube,
            'ports': [port.get_struct() for port in self.GetChildren() if isinstance(port, GenericPortPanel)]
        }

    def set_from_struct(self, config):
        """ Take in a JSON-like structure and use it to populate this class """
        def setter(term, val):
            for item in self.annotated_items:
                if item.key == term:
                    item.SetValue(str(val))
                    return
            raise KeyError(term)
        setter('Inj'+str(id(self))+'length', config['length / m'])
        setter('Inj'+str(id(self))+'ID', config['ID / m'])
        self.state.SetState(CP.State(config['fluid'], dict(T=config['T / K'], P=config['p / kPa'])))
        for port in config['ports']:
            NewPort = self.OnAddPort()
            NewPort.set_from_struct(port)

class AdvancedInjectionPanel(pdsim_panels.PDPanel):
    desc_map = dict()

    def __init__(self, parent, config):
        pdsim_panels.PDPanel.__init__(self, parent)

        #Now we are going to put everything into a scrolled window
        main_sizer = wx.BoxSizer(wx.VERTICAL)

        self.scrolled_panel = ScrolledPanel(self, size=(-1,-1),
                                 style = wx.TAB_TRAVERSAL, name="panel1")
        self.scrolled_panel.SetScrollbars(1,1,1,1)
        self.scrolled_panel.SetupScrolling()

        self.keys_for_config = []
        # The sizer for all the objects
        self.sizer_for_inputs = wx.FlexGridSizer(cols=1, vgap=4, hgap=4)

        # The plot of the scroll wraps
        from PDSim.plot.plots import Plot
        self.PP = Plot(self.scrolled_panel)
        self.ax = self.PP.figure.add_axes((0, 0, 1, 1))
        self.PP.canvas.draw()
        self.sizer_for_inputs.Add(self.PP)

        self.AddLineButton = wx.Button(self.scrolled_panel, label='Add an injection')
        self.sizer_for_inputs.Add(self.AddLineButton)
        self.AddLineButton.Bind(wx.EVT_BUTTON, self.OnAddLine)

        self.DisplayJSONButton = wx.Button(self.scrolled_panel, label='Print JSON')
        self.sizer_for_inputs.Add(self.DisplayJSONButton)
        self.DisplayJSONButton.Bind(wx.EVT_BUTTON, self.OnDisplayJSON)

        self.sizer_for_inputs.Layout()
        self.Refresh()

        self.scrolled_panel.SetSizer(self.sizer_for_inputs)
        main_sizer.Add(self.scrolled_panel,1,wx.EXPAND)
        self.SetSizer(main_sizer)
        main_sizer.Layout()
        self.main_sizer = main_sizer

        if config:
            try:
                self.set_from_struct(config)
            except BaseException as BE:
                traceback.print_tb(BE.__traceback__)
                print(BE)

    def GetGeo(self):
        ITB = self.GetTopLevelParent().MTB.InputsTB
        Scroll = ITB.panels_dict['GeometryPanel'].Scroll
        return Scroll.geo

    def GetState(self):
        """ 
        Get the state class used in the inputs toolbook 
        """

        # Inputs Toolbook
        ITB = self.GetTopLevelParent().MTB.InputsTB
        CPState = None
        for panel in ITB.panels:
            if panel.Name == 'StatePanel':
                CPState = panel.SuctionStatePanel.GetState()
                break
        if CPState is None:
            raise ValueError('StatePanel not found in Inputs Toolbook')
        
        return CPState

    def OnDisplayJSON(self, event=None):
        j = self.get_struct()
        print(json.dumps(j, indent=2))

    def GetActiveLinePanels(self):
        return [child for child in self.scrolled_panel.GetChildren() if (isinstance(child, InjectionMainPanel) and child.IsShown())]

    def OnAddLine(self, event=None):
        """
        Add an injection line to the GUI
        """
        if event is not None: event.Skip()

        NewLine = InjectionMainPanel(self.scrolled_panel, fluid=self.GetState().Fluid)
        self.sizer_for_inputs.Add(NewLine)
        self.sizer_for_inputs.Layout()
        self.main_sizer.Layout()
        self.OnReplot()
        return NewLine

    def OnRemoveLine(self, event=None):
        """
        Remove an injection line from the GUI
        """
        if event is not None: event.Skip()

        old_num_IDV = len(self.GetActiveLinePanels())
        line = event.GetEventObject().Parent
        # assert(isinstance(line, InjectionMainPanel))
        self.sizer_for_inputs.Detach(line)
        line.Hide()
        self.sizer_for_inputs.Layout()
        self.main_sizer.Layout()
        self.OnReplot()

    def get_struct(self):
        """
        Get the nested data structure needed to define the injection lines completely
        """
        Lines = self.GetActiveLinePanels()

        if len(Lines) == 0:
            return {}
        
        return { 
            'lines': [line.get_struct() for line in Lines]
        }

    def set_from_struct(self, config):
        if config and 'lines' in config:
            for Line in config['lines']:
                NewLine = self.OnAddLine()
                NewLine.set_from_struct(Line)

    def OnReplot(self, event=None):

        self.ax.cla()

        s = self.get_struct()
        if s and 'lines' in s:
            geo = self.GetGeo()
            plotScrollSet(pi/4.0,
                          axis=self.ax,
                          geo=geo,
                          offsetScroll=geo.phi_ie_offset > 0)

            for line in s['lines']:
                for port in line['ports']:
                    t = np.linspace(0, 2 * pi)
                    r = port['D / m']/2.0
                    self.ax.plot(r*np.cos(t) + port['x / m'], r*np.sin(t) + port['y / m'], 'b')

        self.PP.canvas.draw()
        self.Layout()

class AsymmetricWrapsPanel(pdsim_panels.PDPanel):
    desc_map = dict(use_asymmetric=('Using Asymmetric Geometry', '', False),
                    phi_ie_offset_asymm=('Offset angle for phi [rad]', 'rad', 3.14159),
                    os_extension=('Extension of the orbiting scroll [rad]', 'rad', 0),
                    SA_dead_volume=('Dead volume added to SA chamber [m^3]', 'm^3', 0.000120),
                    d2_dd_porting_delay_degrees=(
                        'Delay beyond discharge angle that the d2-dd gap opens [deg]', 'deg', 0.0),
                    d1_dd_porting_delay_degrees=(
                        'Delay beyond discharge angle that the d1-dd gap opens [deg]', 'deg', 0.0),
                    )

    def __init__(self, parent, config):
        pdsim_panels.PDPanel.__init__(self, parent)

        self.keys_for_config = []
        # The sizer for all the objects
        sizer_for_inputs = wx.FlexGridSizer(cols=2, vgap=4, hgap=4)

        annotated_values = self.get_annotated_values(['use_asymmetric', 'phi_ie_offset_asymm', 'os_extension', 'SA_dead_volume',
                                                      'd1_dd_porting_delay_degrees', 'd2_dd_porting_delay_degrees'],
                                                     config)

        # Build the items and return the list of annotated GUI objects
        annotated_GUI_objects = self.construct_items(annotated_values,
                                                     sizer=sizer_for_inputs,
                                                     parent=self)

        # The plot of the scroll wraps
        self.PP = PlotPanel(self)
        self.ax = self.PP.figure.add_axes((0, 0, 1, 1))

        sizer_for_inputs.Add(self.PP)
        self.SetSizer(sizer_for_inputs)
        sizer_for_inputs.Layout()
        self.Refresh()

        self.GetTopLevelParent().register_GUI_objects(annotated_GUI_objects)
        self.usingAsymmetric = self.GetTopLevelParent().get_GUI_object('use_asymmetric').GUI_location
        self.usingAsymmetric.Bind(wx.EVT_CHECKBOX, self.OnChangeUsage)
        for o in ['phi_ie_offset_asymm', 'os_extension', 'd1_dd_porting_delay_degrees', 'd2_dd_porting_delay_degrees']:
            self.GetTopLevelParent().get_GUI_object(o).GUI_location.Bind(wx.EVT_KILL_FOCUS, self.OnRefresh)
        self.OnChangeUsage()

    def OnRefresh(self, event=None):
        if event is not None: event.Skip()

        def get(term):
            return self.GetTopLevelParent().get_GUI_object_value(term)

        phi_ie_offset = get('phi_ie_offset_asymm')
        os_extension = get('os_extension')
        if phi_ie_offset == str(phi_ie_offset) and phi_ie_offset.lower().strip() == u'pi':
                phi_ie_offset = pi
        if os_extension == str(os_extension) and os_extension.lower().strip() == u'pi':
                os_extension = pi

        self.GrandParent.panels_dict['GeometryPanel'].OnRefresh()
        geo = self.GrandParent.panels_dict['GeometryPanel'].get_geo()

        # The magic starts here
        geo.phi_ie_offset = phi_ie_offset
        geo.phi_fie += geo.phi_ie_offset
        geo.phi_foe += geo.phi_ie_offset
        geo.phi_oie = geo.phi_fie - pi + os_extension
        geo.phi_ooe = geo.phi_foe - pi + os_extension

        self.ax.cla()

        def build_SA_wall(theta, geo):
            # Involute portion
            phi = np.linspace(geo.phi_fie, geo.phi_oie + np.pi)
            x, y = common_scroll_geo.coords_inv(phi, geo, 0, 'fi')

            # Arc portion
            r = (2 * pi * geo.rb - geo.t) / 2.0
            xee, yee = common_scroll_geo.coords_inv(geo.phi_oie + pi, geo, 0.0, 'fi')
            xse, yse = common_scroll_geo.coords_inv(geo.phi_oie - pi, geo, 0.0, 'fo')
            xoie, yoie = common_scroll_geo.coords_inv(geo.phi_oie, geo, theta, 'oi')
            xooe, yooe = common_scroll_geo.coords_inv(geo.phi_ooe, geo, theta, 'oo')
            x0, y0 = (xee + xse) / 2, (yee + yse) / 2

            beta = atan2(yee - y0, xee - x0)
            t = np.linspace(beta, beta + pi, 1000)
            xc, yc = x0 + r * np.cos(t), y0 + r * np.sin(t)
            return np.r_[x, xc], np.r_[y, yc]

        theta = pi / 4.0
        plotScrollSet(theta,
                      axis=self.ax,
                      geo=geo,
                      shaveOn=False,
                      wallOn=False,
                      offsetScroll=False)
        x, y = build_SA_wall(theta, geo)
        self.ax.plot(x, y)

        self.PP.canvas.draw()
        self.Layout()


    def OnChangeUsage(self, event=None):
        isEnabled = self.usingAsymmetric.GetValue()

        for term in ['phi_ie_offset_asymm', 'os_extension', 'SA_dead_volume',
                     'd1_dd_porting_delay_degrees', 'd2_dd_porting_delay_degrees']:
            self.GetTopLevelParent().get_GUI_object(term).GUI_location.Enable(isEnabled)

        self.PP.Show(isEnabled)

        if isEnabled:
            self.OnRefresh()


class ThreeArcDiscPanel(pdsim_panels.PDPanel):
    desc_map = dict(use_threearc=('Using Three-Arc Discharge Geometry', '', False),
                    r1_threearc=('Radius of arc 1 (connected to inner involute) [m]', 'm', 0.001),
                    r2_threearc=('Radius of arc 2 (connected to outer involute) [m]', 'm', 0),
                    alpha_threearc=('Angle of the break point [rad]', 'rad', 0.55))

    def __init__(self, parent, config):
        pdsim_panels.PDPanel.__init__(self, parent)

        self.keys_for_config = []
        # The sizer for all the objects on the disableable panel
        sizer_for_inputs = wx.FlexGridSizer(cols=2, vgap=4, hgap=4)

        annotated_values = self.get_annotated_values(['use_threearc', 'r1_threearc', 'r2_threearc', 'alpha_threearc'],
                                                     config)

        # Build the items and return the list of annotated GUI objects
        annotated_GUI_objects = self.construct_items(annotated_values,
                                                     sizer=sizer_for_inputs,
                                                     parent=self)
        self.SetSizer(sizer_for_inputs)

        # The plot of the scroll wraps
        self.PP = PlotPanel(self)
        self.ax = self.PP.figure.add_axes((0, 0, 1, 1))

        sizer_for_inputs.Add(self.PP)
        sizer_for_inputs.Layout()
        self.Refresh()

        self.GetTopLevelParent().register_GUI_objects(annotated_GUI_objects)
        self.usingThreeArc = self.GetTopLevelParent().get_GUI_object('use_threearc').GUI_location
        self.usingThreeArc.Bind(wx.EVT_CHECKBOX, self.OnChangeUsage)
        self.OnChangeUsage()

    def OnChangeUsage(self, event=None):

        from DanfossPDSim.core import DanfossScroll
        from DanfossPDSim.asymm_scroll_geo import DanfossGeoVals

        isEnabled = self.usingThreeArc.GetValue()

        for term in ['r1_threearc', 'r2_threearc', 'alpha_threearc']:
            self.GetTopLevelParent().get_GUI_object(term).GUI_location.Enable(isEnabled)

        self.PP.Show(isEnabled)

        if isEnabled:
            def get(term):
                return self.GetTopLevelParent().get_GUI_object_value(term)

            r1 = get('r1_threearc')
            r2 = get('r2_threearc')
            alpha = get('alpha_threearc')

            geo = self.GrandParent.panels_dict['GeometryPanel'].get_geo()
            dfgeo = DanfossGeoVals()
            geo.copy_inplace(dfgeo)
            try:
                DanfossScroll.set_three_arc_disc(dfgeo, r1, r2, alpha)
            except ValueError as VE:
                dlg = wx.MessageDialog(None, str(VE))
                dlg.ShowModal();
                dlg.Destroy()

            self.ax.cla()

            theta = pi / 4.0
            plotScrollSet(theta,
                          axis=self.ax,
                          geo=dfgeo,
                          shaveOn=False,
                          wallOn=False,
                          offsetScroll=False)

            for i in ['1', '2', '3']:
                t1, t2 = getattr(dfgeo, 't1_arc' + i), getattr(dfgeo, 't2_arc' + i)
                xa, ya = getattr(dfgeo, 'xa_arc' + i), getattr(dfgeo, 'ya_arc' + i)
                r = getattr(dfgeo, 'ra_arc' + i)
                t = np.linspace(t1, t2, 300)
                self.ax.plot(xa + r * np.cos(t), ya + r * np.sin(t))
                self.ax.plot(xa, ya, 'o')

        self.PP.canvas.draw()
        self.Layout()


class DanfossMainToolBook(wx.Notebook):
    def __init__(self, parent, configdict):
        wx.Notebook.__init__(self, parent, -1, style=wx.NB_TOP)
        if configdict == '':
            configdict = {}

        self.GeometryInputsPanel = GeometryInputsPanel(self)
        self.OtherInputsPanel = OtherInputsPanel(self, configdict.get('Other',{}))
        self.TipSealPanel = TipSealLeakagePanel(self, configdict.get('TipSeal',{}))
        self.UASuctDiscPanel = UASuctDiscPanel(self, configdict.get('UA',{}))
        self.IDVPanel = IntermediateDischargePanel(self, configdict.get('IDV', {}))
        self.AdvIDVPanel = AdvancedIntermediateDischargePanel(self, configdict.get('AdvIDV',{}))
        self.AdvInjectionPanel = AdvancedInjectionPanel(self, configdict.get('AdvInj',{}))
        self.DummyPortPanel = DummyPortPanel(self, configdict.get('Dummy',{}))
        self.AsymmPanel = AsymmetricWrapsPanel(self, configdict.get('Asymmetric',{}))
        self.ThreeArcDiscPanel = ThreeArcDiscPanel(self, configdict.get('ThreeArc',{}))

        self.AddPage(self.GeometryInputsPanel, 'Geometry conversion')
        self.AddPage(self.OtherInputsPanel, 'Other inputs')
        self.AddPage(self.TipSealPanel, 'Tip Seal')
        self.AddPage(self.UASuctDiscPanel, 'UA(suct-disc)')
        self.AddPage(self.IDVPanel, 'IDV')
        self.AddPage(self.AdvIDVPanel, 'AdvIDV')
        self.AddPage(self.AdvInjectionPanel, 'AdvInj.')
        self.AddPage(self.DummyPortPanel, 'Dummy Port')
        self.AddPage(self.AsymmPanel, 'Asymmetric')
        self.AddPage(self.ThreeArcDiscPanel, 'ThreeArcDisc')


IDV_template = """
#  The volume of one of the IDV "tubes" from the surface of the scroll wrap to the valve plate
V_tube = pi*{IDIDVTube:g}**2/4.0*{LIDVTube:g}

#  The pressure of the IDV tube initially
Tsat_IDV = (inletState.Tsat+outletState.Tsat)/2.0
pIDV = CP.PropsSI('P', 'T', Tsat_IDV, 'Q', 1, inletState.Fluid)/1000.0 #[kPa]

for IDVCV in ['IDVCV.1','IDVCV.2']:
    #  The time-variant control volume for the IDV control volume between
    #  the scroll wrap and the valve plate
    sim.add_CV(ControlVolume(key = IDVCV,
                                    VdVFcn = sim.V_injection, #  Constant volume
                                    VdVFcn_kwargs = dict(V_tube = V_tube),
                                    initialState = State.State(inletState.Fluid, dict(T = Tsat_IDV + 10, P = pIDV)
                                                               )
                                    )
                      )
# Constant volume discharge shell
# If we add IDVs, we create a discharge plenum control volume which is
# then connected to the discharge line.  In this way we can then 
# connect both the normal discharge port and the IDV port to this plenum
sim.add_CV(ControlVolume(key ='discharge_plenum',
                         VdVFcn = sim.V_injection, #  Constant volume 
                         VdVFcn_kwargs = dict(V_tube = {Vdisc_plenum:g}), # Consider order of twice the displacement
                         initialState = outletState.copy()
                         )
                         )

# Midpoint of outer wraps ports
phim = {offset_angle1:g}/180*pi + pi
dphi = {offset_distance:g}/(2*sim.geo.rb*(phim-sim.geo.phi_fo0))

# These ports are connected to c2.1 or c2.2 or d2
p = IDVPort()
p.phi = phim-dphi
p.involute = 'o'
p.offset = {IDIDVTap:g}/2.0
p.D = {IDIDVTap:g}
sim.IDV_ports = [p]
p = IDVPort()
p.phi = phim+dphi
p.involute = 'o'
p.offset = {IDIDVTap:g}/2.0
p.D = {IDIDVTap:g}
sim.IDV_ports.append(p)

# These two ports are connected to c1.1 or c1.2 or d1
p = IDVPort()
p.phi = phim-dphi+pi +{offset_angle2:g}/180*pi
p.involute = 'i'
p.offset = {IDIDVTap:g}/2.0
p.D = {IDIDVTap:g}
sim.IDV_ports.append(p)
p = IDVPort()
p.phi = phim+dphi+pi +{offset_angle2:g}/180*pi
p.involute = 'i'
p.offset = {IDIDVTap:g}/2.0
p.D = {IDIDVTap:g}
sim.IDV_ports.append(p)

# Uncomment these lines to see a plot of the locations of the IDV ports
# plotScrollSet(0.0, geo = sim.geo)
# for port in sim.IDV_ports:
#     x,y = scroll_geo.coords_inv(port.phi, sim.geo, 0, 'f'+port.involute)
#     nx,ny = scroll_geo.coords_norm(port.phi, sim.geo, 0, 'f'+port.involute)
#     x0 = x - nx*port.offset
#     y0 = y - ny*port.offset
#     t = np.linspace(0, 2*pi)
#     plt.plot(port.D/2.0*np.cos(t) + x0, port.D/2.0*np.sin(t) + y0,'b')
# plt.show()
        
#  Calculate the areas between each IDV port and every control volume
sim.calculate_IDV_areas()

for IDVCV in ['IDVCV.1','IDVCV.2']:
    #  Add the flow between the IDV control volume and the discharge plenum
    FP = FlowPath(key2 = IDVCV,
                  key1 = 'discharge_plenum',
                  MdotFcn = sim.IDVValveNoDynamics,
                  )
    FP.A = pi*{IDIDVTube:g}**2/4.0*{Xd_IDVCV_plenum:g}
    sim.add_flow(FP)
    
    if IDVCV == 'IDVCV.1':
        ports = sim.IDV_ports[0:2]
        partners = ['c2.1','c2.2','d2']
    elif IDVCV == 'IDVCV.2':
        ports = sim.IDV_ports[2::]
        partners = ['c1.1','c1.2','d1']
    else:
        raise ValueError
        
    for port in ports:
        for partner in port.area_dict:
        
            #  Create a spline interpolator object for the area between port and the partner chamber
            A_interpolator = scipy.interpolate.splrep(port.theta, port.area_dict[partner], k = 2, s = 0)
            
            #  Add the flow between the IDV control volume and the chamber through the port
            sim.add_flow(FlowPath(key1 = IDVCV,
                                         key2 = partner,
                                         MdotFcn = sim.IDV_CV_flow,
                                         MdotFcn_kwargs = dict(X_d = 1, 
                                                               A_interpolator = A_interpolator)
                                         )
                                 )

for flow in sim.Flows:
    if flow.key1 == 'outlet.1': flow.key1 = 'discharge_plenum'
    if flow.key2 == 'outlet.1': flow.key2 = 'discharge_plenum'
    
FP = FlowPath(key1='discharge_plenum', 
              key2='outlet.1', 
              MdotFcn=IsentropicNozzleWrapper(),
          )
FP.A = pi*0.02**2/4
sim.add_flow(FP)
"""

dummy_port_template = """
# Add the dummy port to the model
sim.add_flow(FlowPath(key1 = 'd2',
                      key2 = 'dd',
                      MdotFcn = sim.DISC_D2_DUMMY,
                      MdotFcn_kwargs = dict(X_d = {X_d_dummy:g},
                                            A_cs_dummy = sim.h_dummy_port*sim.geo.ra_arc1*2 # depth * r1 * 2
                                            )
                      )
             )
"""

asymm_post_build_template = """
# Set the geometry for the asymmetric compressor
sim.geo.phi_fie += {phi_ie_offset:14g} # phi_ie_offset
sim.geo.phi_foe += {phi_ie_offset:14g} # phi_ie_offset
os_extension = {os_extension:.14g} # extension of the orbiting scroll
sim.geo.phi_oie = sim.geo.phi_fie - pi + os_extension
sim.geo.phi_ooe = sim.geo.phi_foe - pi + os_extension
sim.SA_dead_volume = {SA_dead_volume} # Additional volume for the SA chamber [m^3]
"""

set_delayed_porting_template = """
# Find the flow path for {key:s}-dd, and increase the port angle delay keyword argument
for flow in sim.Flows:
    if flow.key1 == '{key:s}' and flow.key2 == 'dd' and flow.MdotFcn.__name__ is not None and 'D_to_DD' in flow.MdotFcn.__name__:
        flow.MdotFcn.kwargs['porting_delay_degrees'] = {porting_delay_degrees:g}
        print('set the porting delay to {porting_delay_degrees:g} degrees for the {key:s}-dd flow path; kwargs are now: '+str(flow.MdotFcn.kwargs))
"""

set_three_arc_template = """
# Set the three-arc discharge geometry for the asymmetric compressor
DanfossScroll.set_three_arc_disc(sim.geo,{r1:g},{r2:g},{alpha:g})
"""

injection_template = """           
        
from PDSim.scroll.core import Port

# If scipy is available, use its spline interpolation function, otherwise, 
# use our implementation (for packaging purposes)
try:
    import scipy.interpolate as interp
except ImportError:
    import PDSim.misc.scipylike as interp

sim.fixed_scroll_ports = []

injection_info = {injection_info:s}

# Add the ports
for line in injection_info["lines"]:
    key = line["key"]
    sim.add_tube(Tube(key1=key+'.1',
                      key2=key+'.2',
                      L=line['length / m'],
                      ID=line['ID / m'],
                      mdot=mdot_guess*0.1, 
                      State1=State.State(inletState.Fluid,
                                         dict(P=line['p / kPa'], T=line['T / K'])
                                         ),
                      fixed=1,
                      TubeFcn=sim.TubeCode
                      )     
                  )

    for port in line["ports"]:
        p = Port()
        p.x = port['x / m']
        p.y = port['y / m']
        p.D = port['D / m']
        p.parent = line['key']
        p.X_d = 0.8 # hardcoded for now...
        p.X_d_backflow = 0.8 # hardcoded for now...
        sim.fixed_scroll_ports.append(p)    
    
for port in sim.fixed_scroll_ports:

    #  Calculate the areas between port and every control volume
    sim.calculate_port_areas(port)
    
    for partner in port.area_dict:
    
        #  Create a spline interpolator object for the area between port and the partner chamber
        A_interpolator = interp.splrep(port.theta, port.area_dict[partner], k=1, s = 0)
        
        #  Add the flow between the injection tube and the chamber through the port
        sim.add_flow(FlowPath(key1 = port.parent +'.2',
                              key2 = partner,
                              MdotFcn = sim.INTERPOLATING_NOZZLE_FLOW,
                              MdotFcn_kwargs = dict(X_d = port.X_d,
                                                    X_d_backflow = port.X_d_backflow,
                                                    upstream_key = port.parent + '.2',
                                                    A_interpolator = A_interpolator
                                                    )
                             )
                     )
"""


class DanfossPlugin(pdsim_plugins.PDSimPlugin):
    short_description = "Danfoss plugin"

    def should_enable(self):
        """ Returns True if the plugin should be enabled """
        return True

    def activate(self, event=None, config=''):
        """ Activate the plugin """
        self._activated = not self._activated

        ITB = self.GUI.MTB.InputsTB

        # Append the folder containing this file to the python search path
        sys.path.append(os.path.split(os.path.abspath(__file__))[0])
        print('added', sys.path[-1], 'to the path')

        if self._activated:

            this_directory, fname = os.path.split(__file__)
            ico_path = os.path.join(this_directory, 'Danfoss.png')

            if os.path.exists(ico_path):
                ico = wx.Bitmap(ico_path, wx.BITMAP_TYPE_PNG)
                self.image_index = ITB.il.Add(ico)
            else:
                print('The image file', ico_path, 'was not found')
                self.image_index = ITB.il.Add(wx.EmptyBitmap(32, 32))

            # Add the panel to the inputs panel
            self.DanfossMTB = DanfossMainToolBook(ITB, config)

            ITB.AddPage(self.DanfossMTB, "Danfoss", imageId=self.image_index)

            # self.page_index = ITB.FindPage(self.DanfossMTB)

        else:
            page_names = [ITB.GetPageText(I) for I in range(ITB.GetPageCount())]
            I = page_names.index("Danfoss")
            ITB.RemovePage(I)
            ITB.GetImageList().Remove(self.image_index)
            # Unregister the terms
            keys = ['disable_radial_suction', 'disable_flank_suction', 'delta_axial',
                    'w_bypass', 'Xd_bypass', 'Xd_Fanno', 'w_slot', 'delta_slot', 'fF_slot',
                    'UA_suct_disc']
            self.DanfossMTB.GetTopLevelParent().unregister_GUI_objects(keys)
            self.DanfossMTB.Destroy()
            del self.DanfossMTB

    def get_script_chunks(self):
        def get(term):
            return self.GUI.get_GUI_object_value(term)

        post_import = textwrap.dedent(
            'from DanfossPDSim.core import *\nfrom DanfossPDSim.asymm import AsymmetricScroll\n')

        if self.DanfossMTB.AsymmPanel.usingAsymmetric.GetValue() == True:
            # We have an asymmetric scroll
            pre_build_instantiation = '\n#Over-write the Scroll class used\nScroll = AsymmetricScroll\n'
            post_build = ''
        else:
            # We have a symmetric scroll
            pre_build_instantiation = '\n#Over-write the Scroll class used\nScroll = DanfossScroll\n'
            post_build = ''

        post_build_instantiation = ''

        #  Terms from other inputs
        for term in ['disable_radial_suction', 'disable_flank_suction']:
            val = self.GUI.get_GUI_object_value(term)
            post_build_instantiation += 'sim.{name:s} = {value:s}\n'.format(name=term,
                                                                            value=str(val))

        # Terms from leakage
        if self.DanfossMTB.TipSealPanel.usingTipSeal.GetValue() == True:
            for term in ['delta_axial', 'w_bypass', 'Xd_bypass', 'Xd_Fanno', 'w_slot', 'delta_slot', 'fF_slot']:
                val = self.GUI.get_GUI_object_value(term)
                post_build_instantiation += 'sim.{name:s} = {value:s}\n'.format(name=term,
                                                                                value=str(val))
            post_build += 'sim.add_Danfoss_flows()\n'

        # Terms from UA-suct-disc term
        if self.DanfossMTB.UASuctDiscPanel.usingUAsuct.GetValue() == True:
            for term in ['UA_suct_disc']:
                val = self.GUI.get_GUI_object_value(term)
                post_build_instantiation += 'sim.{name:s} = {value:s}\n'.format(name=term,
                                                                                value=str(val))
        else:
            post_build_instantiation += 'sim.{name:s} = {value:s}\n'.format(name='UA_suct_disc', value='0.0')

        # Terms from dummy port
        if self.DanfossMTB.DummyPortPanel.usingDummy.GetValue() == True:
            for term in ['h_dummy_port', 'X_d_dummy']:
                val = self.GUI.get_GUI_object_value(term)
                post_build_instantiation += 'sim.{name:s} = {value:s}\n'.format(name=term,
                                                                                value=str(val))
            post_build += dummy_port_template.format(X_d_dummy=self.GUI.get_GUI_object_value('X_d_dummy'),
                                                     )

        if self.DanfossMTB.AsymmPanel.usingAsymmetric.GetValue() == True:
            post_build += set_delayed_porting_template.format(
                porting_delay_degrees=self.GUI.get_GUI_object_value('d1_dd_porting_delay_degrees'), key='d1')
            post_build += set_delayed_porting_template.format(
                porting_delay_degrees=self.GUI.get_GUI_object_value('d2_dd_porting_delay_degrees'), key='d2')

        if self.DanfossMTB.ThreeArcDiscPanel.usingThreeArc.GetValue():
            post_build += set_three_arc_template.format(r1=self.GUI.get_GUI_object_value('r1_threearc'),
                                                        r2=self.GUI.get_GUI_object_value('r2_threearc'),
                                                        alpha=self.GUI.get_GUI_object_value('alpha_threearc'))

        # Using the IDV if enabled
        if self.DanfossMTB.IDVPanel.usingIDV.GetValue() == True:
            post_build += IDV_template.format(IDIDVTube=get('IDIDVTube'),
                                              LIDVTube=get('LIDVTube'),
                                              IDIDVTap=get('IDIDVTap'),
                                              Vdisc_plenum=get('Vdisc_plenum'),
                                              Xd_IDVCV_plenum=get('Xd_IDVCV_plenum'),
                                              offset_angle1=get('offset_angle1'),
                                              offset_angle2=get('offset_angle2'),
                                              offset_distance=get('offset_distance'),
                                              offset_outer=get('offset_outer'),  # added by Alain Picavet
                                              offset_inner=get('offset_inner'),  # added by Alain Picavet
                                              )

        if self.DanfossMTB.AdvIDVPanel.GetActiveIDVPanels():
            post_build += 'IDV_info = ' + json.dumps(self.DanfossMTB.AdvIDVPanel.get_struct(), indent=2) + '\nsim.add_IDVs(IDV_info)\n'

        if self.DanfossMTB.AdvInjectionPanel.GetActiveLinePanels():
            post_build += injection_template.format(injection_info = json.dumps(self.DanfossMTB.AdvInjectionPanel.get_struct(), indent=2))

        # The "normal" terms
        core = dict(post_import=post_import,
                    pre_build_instantiation=pre_build_instantiation,
                    post_build_instantiation=post_build_instantiation,
                    post_build=post_build,
                    )

        # Add term for asymmetric terms
        if self.DanfossMTB.AsymmPanel.usingAsymmetric.GetValue():
            vals = dict(phi_ie_offset=self.GUI.get_GUI_object_value('phi_ie_offset_asymm'),
                        os_extension=self.GUI.get_GUI_object_value('os_extension'),
                        SA_dead_volume=self.GUI.get_GUI_object_value('SA_dead_volume'))
            if vals['phi_ie_offset'] == str(vals['phi_ie_offset']) and vals['phi_ie_offset'].lower().strip() == u'pi':
                    vals['phi_ie_offset'] = pi
            if vals['os_extension'] == str(vals['os_extension']) and vals['os_extension'].lower().strip() == u'pi':
                    vals['os_extension'] = pi
            core['plugin_injected_chunks'] = dict(ScrollGeometryPanel_After=asymm_post_build_template.format(**vals))

        if self.DanfossMTB.AsymmPanel.usingAsymmetric.GetValue() == True:
            # We have an asymmetric scroll, so upgrade the geometry class
            core['post_build_instantiation'] += 'sim.geo = DanfossGeoVals()\n'

        return core

    def get_config_chunk(self):
        def getOneValue(self, options):
            tab_val = {}
            for term in options:
                #val = []
                #val.append(term)
                #val.append(self.GUI.get_GUI_object_value(term))
                val = self.GUI.get_GUI_object_value(term)
                tab_val[term] = val
            return tab_val

        chunk = []
        configdict = {}
        #Other options
        opt = getOneValue(self, self.DanfossMTB.OtherInputsPanel.desc_map.keys())
        dict_values = dict(Other = opt)
        #TipSeal
        tip = getOneValue(self, self.DanfossMTB.TipSealPanel.desc_map.keys())
        dict_values["TipSeal"] = tip
        #UA
        ua = getOneValue(self, self.DanfossMTB.UASuctDiscPanel.desc_map.keys())
        dict_values["UA"] = ua
        #Dummy port
        dummy = getOneValue(self, self.DanfossMTB.DummyPortPanel.desc_map.keys())
        dict_values["Dummy"] = dummy
    #     # #Compliance
    #     # compliance = getOneValue(self,['use_compliance','A_compliance_feed','V_compliance_annulus','H_compliance_annulus','compliance_involute_angle','compliance_distance','inner_radius','outer_radius','axial_gap','X_d_axial_leakage'])
    #     # #compliance.append(["Is active", self.DanfossMTB.AxialCompliancePanel.usingCompliance.GetValue()])
    #     # dict_values["Compliance"] = compliance
        #Asymmetric
        asym = getOneValue(self, self.DanfossMTB.AsymmPanel.desc_map.keys())
        dict_values["Asymmetric"] = asym
        #IDV
        IDV  = getOneValue(self, self.DanfossMTB.IDVPanel.desc_map.keys())
    #                                 'LIDVTube',
    #                                 'IDIDVTap',
    #                                 'Vdisc_plenum',
    #                                 'Xd_IDVCV_plenum',
    #                                 'offset_angle1',
    #                                 'offset_angle2',
    #                                 'offset_distance',
    #                                 'offset_outer', # added by Alain Picavet
    #                                 'offset_inner', # added by Alain Picavet
    #                                 'triple_IDV',
    #                                 'single_IDV',
    #                                 'angle_IDV1',
    #                                 'outer_IDVs',
    #                                 'inner_IDVs',
    #                                 'delta_angle_IDV2',
    #                                 'delta_angle_IDV3',
    #                                 'delta_angle_IDV4',
    #                                 'delta_angle_IDV5',
    #                                 'delta_angle_IDV6',
    #                                 'delta_angle_IDV7',
    #                                 ])
    #     IDV.append(["Is active", self.DanfossMTB.IDVPanel.usingIDV.GetValue()])
        dict_values["IDV"] = IDV
        dict_values["AdvIDV"] = self.DanfossMTB.AdvIDVPanel.get_struct()
        dict_values["AdvInj"] = self.DanfossMTB.AdvInjectionPanel.get_struct()
        # Three-Arc
        threearc = getOneValue(self, self.DanfossMTB.ThreeArcDiscPanel.desc_map.keys())
        dict_values["ThreeArc"] = threearc
        chunk.append(dict_values)
        val_dict = {"Plugin:DanfossPlugin":dict_values}
        return val_dict

    def apply(self):
        """
        Doesn't need to do anything at build time of simulation before it is run
        """
        pass

# Add this guard so that the plugin will safely import into GUI
if __name__ == '__main__':
    pass
