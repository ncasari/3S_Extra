from __future__ import print_function


from PDSim.plot.plots import *

from PDSim.plot.plots import PlotNotebook as PlotNotebook_PDSim


class PlotNotebook(PlotNotebook_PDSim):
    def __init__(self, Simulation, parent, id=-1, plot_names=None, family=None):
        PlotNotebook_PDSim.__init__(self, Simulation, parent, id=id, plot_names=plot_names, family=family)

    def build_main_page(self):
        page = wx.Panel(self.nb, -1)
        sizer = wx.FlexGridSizer(cols=2)
        label1 = wx.StaticText(page, label='Click on the buttons below to add plot')
        sizer.Add(label1)
        self.plot_buttons = [('Stepsize', self.stepsize_theta),
                             ('Volume v. crank angle', self.V_theta),
                             ('Derivative of Volume v. crank angle', self.dV_dtheta),
                             ('Temperature v. crank angle', self.T_theta),
                             ('Pressure v. crank angle', self.p_theta),
                             ('Pressure v. volume', self.p_V),
                             ('Density v. crank angle', self.rho_theta),
                             ('Mass v. crank angle', self.m_theta),
                             ('Mass flow v. crank angle', self.mdot_theta),
                             ('Temperature-pressure', self.temperature_pressure),
                             ('Heat transfer v. crank angle', self.heat_transfer),
                             ('Initial temperature history', self.initial_temperature_history),
                             ('Lump residuals v. lump temps', self.lumps_residual_v_lump_temps),
                             ('Discharge residual history', self.discharge_residual_history),
                             ('Valve lift v. crank angle', self.valve_theta)
                             ]
        self.recip_plot_buttons = [('Valve lift v. crank angle', self.valve_theta)]
        self.scroll_plot_buttons = [('Pressure profile', self.pressure_profile),
                                    ('Axial force v. crank angle', self.axial_force),
                                    ('X-direction force v. crank angle', self.x_direction_force),
                                    ('Y-direction force v. crank angle', self.y_direction_force),
                                    ('Crank pin force magnitude v. crank angle', self.magnitude_force),
                                    ('Gas Torque v. crank angle', self.torque),
                                    ('Force trace', self.force_trace),
                                    ('Force component trace', self.force_component_trace),
                                    ('Radial force', self.radial_force),
                                    ('Tangential force', self.tangential_force)
                                    ]
        self.danfoss_plot_buttons = [('Pressure profile', self.pressure_profile),
                                    ('Axial force v. crank angle', self.axial_force),
                                    ('X-direction force v. crank angle', self.x_direction_force),
                                    ('Y-direction force v. crank angle', self.y_direction_force),
                                    ('Crank pin force magnitude v. crank angle', self.magnitude_force),
                                    ('Gas Torque v. crank angle', self.torque),
                                    ('Force trace', self.force_trace),
                                    ('Force component trace', self.force_component_trace),
                                    ('Radial force', self.radial_force),
                                    ('Tangential force', self.tangential_force),
                                    ('Journal Bearing force', self.bearing_force)
                                    ]
        for value, callbackfcn in self.plot_buttons:
            btn = wx.Button(page, label=value)
            sizer.Add(btn)
            btn.Bind(wx.EVT_BUTTON, callbackfcn)

        if self.family is not None:
            if self.family == 'Scroll Compressor':
                more_plot_buttons = self.scroll_plot_buttons
            elif self.family == 'Recip Compressor':
                more_plot_buttons = self.recip_plot_buttons
            elif self.family == 'Danfoss Scroll Compressor':
                more_plot_buttons = self.danfoss_plot_buttons
            else:
                raise ValueError("Invalid family; options are 'Scroll Compressor' or 'Recip Compressor'")
        else:
            more_plot_buttons = None

        if more_plot_buttons is not None:
            for value, callbackfcn in more_plot_buttons:
                btn = wx.Button(page, label=value)
                sizer.Add(btn)
                btn.Bind(wx.EVT_BUTTON, callbackfcn)
        else:
            print('could not add more buttons particular to current family:', self.family)

        page.SetSizer(sizer)
        self.nb.AddPage(page, "Main")


    def bearing_force(self, event=None):
        # Bearing force magnitude

        axes = self.add('Bearing force magnitude').gca()
        if isinstance(self.Sim, h5py.File):
            theta = self.Sim.get('/t')[()]
            Fosb = self.Sim.get('/forces/Fosb')[()]
            Fumb = self.Sim.get('/forces/Fumb')[()]
            Flmb = self.Sim.get('/forces/Flmb')[()]
        else:
            theta = self.Sim.t
            Fosb = self.Sim.forces.Fosb
            Fumb = self.Sim.forces.Fumb
            Flmb = self.Sim.forces.Flmb

        axes.plot(theta, Fosb, 'r-', lw=1.5, label='OSB')
        axes.plot(theta, Fumb, 'b-', lw=1.5, label='UMB')
        axes.plot(theta, Flmb, 'g-', lw=1.5, label='LMB')
        axes.set_ylabel(r'Bearing force [kN]')
        axes.set_xlabel(r'$\theta$ [rad]')

        xmin, xmax = axes.get_xlim()
        axes.set_xlim(xmin, xmin + (xmax - xmin) * 1.5)
        axes.legend(loc='upper right', bbox_to_anchor=(1, 1),
                    ncol=2, fancybox=True, shadow=True)