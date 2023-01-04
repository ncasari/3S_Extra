# -*- coding: utf-8 -*-

from __future__ import absolute_import
import wx, yaml, os
from panels import scroll_panels, pdsim_panels
import numpy as np

family_menu_name = 'Danfoss Scroll Compressor'

# Strings for the script
import_string = 'from PDSim.scroll.core import Scroll\n'
instantiation_string = 'sim = Scroll()\n'
additional_imports_string = ''

def write_to_xlsx(workbook, runs):
    """
    This provides code to the GUI to add additional terms that are interesting for the scroll compressor
    
    workbook : xlsxwriter.Workbook opened with constant_memory = True
    runs : opened HDF5 files from the runs
    """
    
    ######################### PRESSURE PROFILES ###############################
    ######################### PRESSURE PROFILES ###############################
    ######################### PRESSURE PROFILES ###############################
    
    ws = workbook.add_worksheet('Pressure profiles')
    
    # Adjust these to adjust the spacing around each block of data
    row_idx = 3
    col_idx = 1
    
    ## Output the matrices
    ## Data has to be written by row because workbook opened with constant_memory = True for writing speed
    
    # Adjust these to adjust the spacing around each block of data
    row_idx = 4
    col_idx = 1
    
    # 3 is the number of empty columns between runs
    offset = 3 + 3
    
    # Header 
    my_col_idx = col_idx
    for run in runs:
        run_index = run.get('run_index')[()]
        ws.write(row_idx - 3, my_col_idx - 1, 'Run index #'+str(run_index))
        
        if run.get('description'):
            description = run.get('description').asstr()[()]
            ws.write(row_idx - 3, my_col_idx - 1, 'Run index #'+str(run_index)+': '+description)
            
        my_col_idx += offset
    
    # Column headers
    my_col_idx = col_idx
    for run in runs:
        ws.write(row_idx-2, my_col_idx - 1, 'theta')
        ws.write(row_idx-2, my_col_idx , 'p1')
        ws.write(row_idx-2, my_col_idx + 1, 'p2')
                
        my_col_idx += offset
        
    datas = []
    maxlen = 0
    for run in runs:
        # Each are stored as 1D array, convert to 2D column matrix
        theta = np.array(run.get('summary/theta_profile')[()], ndmin = 2)
        p1 = np.array(run.get('summary/p1_profile')[()], ndmin = 2)
        p2 = np.array(run.get('summary/p2_profile')[()], ndmin = 2)
    
        datas.append(np.r_[theta, p1, p2].T)
        if np.prod(theta.shape) > maxlen:
            maxlen = np.prod(theta.shape)

    # Data
    for r in range(maxlen):
        my_col_idx = col_idx
        for data in datas:

            if r >= data.shape[0]:
                my_col_idx += offset
                continue
                
            # Theta            
            ws.write(r+row_idx-1, my_col_idx-1,data[r, 0])
            
            if not np.isnan(data[r,1]):
                ws.write(r+row_idx-1, my_col_idx,data[r, 1])
            if not np.isnan(data[r,2]):
                ws.write(r+row_idx-1, my_col_idx+1,data[r, 2])
                    
            my_col_idx += offset    
            
    ########################### VIRTUAL SENSORS ###############################
    ########################### VIRTUAL SENSORS ###############################
    ########################### VIRTUAL SENSORS ###############################
    
    if run.get('sensors') is not None and run.get('sensors/coords') is not None:
        
        ws = workbook.add_worksheet('Virtual Sensors')
        
        ## Build a sensor structure, which includes a header and the data
        theta = run.get('t')[()]
        
        run_sensors = []
        nrow = 0
        ncol = 0
        for run in runs:
            vals = []
            sensors_grp = run.get('sensors')
            for i in range(len(sensors_grp.get('T'))):
                x = sensors_grp.get('coords/{i:d}/0'.format(i=i))[()]
                y = sensors_grp.get('coords/{i:d}/1'.format(i=i))[()]
                T = sensors_grp.get('T/{i:d}'.format(i=i))[()]
                p = sensors_grp.get('p/{i:d}'.format(i=i))[()]
                rho = sensors_grp.get('rho/{i:d}'.format(i=i))[()]
                if len(p) > nrow:
                    nrow = len(p)
                ncol += 6
                
                vals.append((x, y, T, p, rho))
            run_sensors.append(vals)
        
        ## Make a large list of list of empty strings
        
        buf = [['' for col in range(ncol)] for row in range(nrow+10)]
        
        col_offset = 0
        row_offset = 3
        for run, sensors in zip(runs, run_sensors):
            
            run_index = run.get('run_index')[()]
            s = 'Run index #'+str(run_index)
            
            if run.get('description'):
                description = run.get('description').asstr()[()]
                s = 'Run index #'+str(run_index)+': '+description
            
            buf[row_offset-2][0 + col_offset] = s
            for sensor in sensors:                
                buf[row_offset][0 + col_offset] = 'theta [rad]'
                buf[row_offset][1 + col_offset] = 'T [K]'
                buf[row_offset][2 + col_offset] = 'p [kPa]'
                buf[row_offset][3 + col_offset] = 'rho [kg/m^3]'
                x, y, T, p, rho = sensor
                buf[row_offset-1][0 + col_offset] = 'x={x:g}, y={y:g}'.format(x = x, y = y)
                for row in range(len(T)):
                    buf[row+1+row_offset][0 + col_offset] = theta[row]
                    buf[row+1+row_offset][1 + col_offset] = T[row]
                    buf[row+1+row_offset][2 + col_offset] = p[row]
                    buf[row+1+row_offset][3 + col_offset] = rho[row]
                col_offset += 5

        for r in range(len(buf)):
            for c in range(len(buf[0])):
                if isinstance(buf[r][c],basestring) or not np.isnan(buf[r][c]):
                    ws.write(r, c, buf[r][c])

class InputsToolBook(pdsim_panels.InputsToolBook):
    """
    The toolbook that contains the pages with input values
    """
    def __init__(self, parent, config):
        """
        Parameters
        ----------
        config : yaml config file
            The top-level configuration file
        """
        wx.Listbook.__init__(self, parent, -1, style=wx.NB_LEFT)
        self.il = wx.ImageList(32, 32)
        indices=[]
        for imgfile in ['Geometry.png',
                        'StatePoint.png',
                        'MassFlow.png',
                        'MechanicalLosses.png',
                        'Sensor.png']:
            ico_path = os.path.join('ico',imgfile)
            indices.append(self.il.Add(wx.Image(ico_path,wx.BITMAP_TYPE_PNG).ConvertToBitmap()))
        self.AssignImageList(self.il)
        
        Main = wx.GetTopLevelParent(self)
        # Make the scroll panels.  
        self.panels=(scroll_panels.GeometryPanel(self, config['GeometryPanel'],name='GeometryPanel'),
                     pdsim_panels.StateInputsPanel(self, config['StatePanel'], name='StatePanel'),
                     scroll_panels.MassFlowPanel(self, config['MassFlowPanel'], name='MassFlowPanel'),
                     scroll_panels.MechanicalLossesPanel(self, config['MechanicalLossesPanel'], name='MechanicalLossesPanel'),
                     scroll_panels.VirtualSensorsPanel(self, config['VirtualSensorsPanel'], name='VirtualSensorsPanel')
                     )
        
        for Name, index, panel in zip(['Geometry','States','Flow','Mechanical','Sensors'],indices,self.panels):
            self.AddPage(panel,Name,imageId=index)
            
        #: A dictionary that maps name to panel 
        self.panels_dict = {panel.Name:panel for panel in self.panels}
        
    
    def get_config_chunks(self):
        chunks = {}
        for panel in self.panels:
            chunks[panel.name] = panel.get_config_chunk()
        return chunks
        
danfoss_yaml=(
"""
family : Danfoss Scroll Compressor

GeometryPanel:
  Vdisp : 104.8e-6 # Displacement volume / revolution [m^3/rev]
  Vratio : 2.2 # Built-in volume ratio [-]
  ro : 0.005 # Orbiting radius [m]
  t : 0.004 # Scroll wrap thickness [m]
  use_offset : False
  delta_offset : 1e-3 # Offset scroll portion gap width [m]
  phi_fi0 : 0.0 # Inner wrap initial involute angle [rad]
  phi_fis : 3.141 # Inner wrap starting involute angle [rad]
  phi_fos : 0.3 # Outer wrap starting involute angle [rad]
  delta_flank : 15e-6 # Flank gap width [m]
  delta_radial : 15e-6 # Radial gap width [m]
  d_discharge : 0.01 # Discharge port diameter [m]
  inlet_tube_length : 0.3 # Inlet tube length [m]
  inlet_tube_ID : 0.02 # Inlet tube inner diameter [m]
  outlet_tube_length : 0.3 # Outlet tube length [m]
  outlet_tube_ID : 0.02 # Outlet tube inner diameter [m]
  disc_curves:
      type : 2Arc
      r2 : 0

MassFlowPanel:
  sa-s1:
      model : IsentropicNozzle
      options : {Xd : 0.8}
  sa-s2:
      model : IsentropicNozzle
      options : {Xd : 0.8}
  inlet.2-sa:
      model : IsentropicNozzle
      options : {Xd : 0.8}
  d1-dd:
      model : IsentropicNozzle
      options : {Xd : 0.8}
  d2-dd:
      model : IsentropicNozzle
      options : {Xd : 0.8}

MechanicalLossesPanel:
  eta_motor : 0.95 # Motor efficiency [-]
  h_shell : 0.01 # Shell air-side heat transfer coefficient [kW/m^2/K]
  A_shell : 0.040536 # Shell Area [m^2]
  Tamb : 298.0 # Ambient temperature [K]
  mu_oil : 0.0086 # Oil viscosity [Pa-s]
  D_upper_bearing : 0.025 # Upper bearing diameter [m]
  L_upper_bearing : 0.025 # Upper bearing length [m]
  c_upper_bearing : 20e-6 # Upper bearing gap [m]
  D_crank_bearing : 0.025 # Crank bearing diameter [m]
  L_crank_bearing : 0.025 # Crank bearing length [m]
  c_crank_bearing : 20e-6 # Crank bearing gap [m]
  D_lower_bearing : 0.025 # Lower bearing diameter [m]
  L_lower_bearing : 0.025 # Lower bearing length [m]
  c_lower_bearing : 20e-6 # Lower bearing gap [m]
  thrust_friction_coefficient : 0.03 # Thrust bearing friction coefficient [-]
  thrust_ID : 0.08 # Thrust bearing inner diameter [m]
  thrust_OD : 0.3 # Thrust bearing outer diameter [m]
  L_ratio_bearings : 3.0 # Ratio of lengths for bearings [-]
  HTC : 0.0 # Heat transfer coefficient in scrolls [-]
  journal_tune_factor : 1.0 # Journal loss tune factor [-]
  scroll_plate_thickness : 0.008

StatePanel:
  omega : 377.0 # Rotational speed [rad/s]
  inletState: 
      Fluid : R410A
      T : 283.15 #[K]
      rho : 5.75 #[kg/m^3]
  discharge:
      pratio : 2.0

VirtualSensorsPanel:
  coords: []

ParametricPanel:
  structured : True

SolverInputsPanel:
  cycle_integrator: RK45
  integrator_options: {epsRK45 : 1e-7}
  eps_cycle : 0.002 # Cycle-Cycle convergence tolerance (RSSE) [-]
  eps_energy_balance : 0.05 # Energy balance convergence tolerance (RSSE) [-]
  outlet_temperature_guesss : -1 # <0 means use adiabatic efficiency to guess

Plugin:DanfossPlugin:
  Asymmetric:
    use_asymmetric: False
    phi_ie_offset_asymm: 3.14159
    os_extension: 0
    SA_dead_volume: 0.00012
    d1_dd_porting_delay_degrees: 0.0
    d2_dd_porting_delay_degrees: 0.0
  Compliance:
    A_compliance_feed: 0.002
    H_compliance_annulus: 0.01
    V_compliance_annulus: 2.19911485751e-05
    X_d_axial_leakage: 1.0
    axial_gap: 5.0e-06
    compliance_distance: 0.01
    compliance_involute_angle: 6
    inner_radius: 0.03
    outer_radius: 0.04
  Dummy:
    use_dummy: False
    X_d_dummy: 0.7
    h_dummy_port: 0.003
  IDV:
    use_IDV: False
    IDIDVTap: 0.004
    IDIDVTube: 0.008
    LIDVTube: 0.044
    Vdisc_plenum: 0.0005
    Xd_IDVCV_plenum: 0.75
    angle_IDV1: 0
    delta_angle_IDV2: 15
    delta_angle_IDV3: 15
    delta_angle_IDV4: 15
    delta_angle_IDV5: 15
    delta_angle_IDV6: 15
    delta_angle_IDV7: 15
    inner_IDVs: 0
    offset_angle1: 60.29
    offset_angle2: 9.53
    offset_distance: 0.00732
    offset_inner: 0
    offset_outer: 0
    outer_IDVs: 0
    single_IDV: 0
    triple_IDV: 0
  Other:
    disable_flank_suction: False
    disable_radial_suction: False
  TipSeal:
    use_tipseal: False
    delta_axial: 0.00015
    w_bypass: 0.001
    Xd_bypass: 0.7
    Xd_Fanno: 0.7
    w_slot: 0.003
    delta_slot: 0.00025
    fF_slot: 0.001
  UA:
    use_UAsuct: False
    UA_suct_disc: 0.015
  ThreeArc:
    use_threearc: False
    r1_threearc: 0.001
    r2_threearc: 0.0
    alpha_threearc: 0.55
"""
)

# This block was removed from the default configuration so that injection is not
# enabled by default.  It can be replaced to enable injection by default
"""
Plugin:ScrollInjectionPlugin:
    - Length : 1.1
      ID : 0.01
      inletState: 
          Fluid : R410A
          T : 283.15 #[K]
          rho : 5.75 #[kg/m^3]
      ports:
      - phi : 7.2
        D : 0.0025
        offset : 0.00125
        check_valve : True
        inner_outer : 'i'
        symmetric : None
      - phi : 10.1
        D : 0.0025
        offset : 0.00125
        check_valve : True
        inner_outer : 'o'
        symmetric : 1

"""

def get_defaults():
    return yaml.load(danfoss_yaml,Loader=yaml.FullLoader)