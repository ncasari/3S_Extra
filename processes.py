from __future__ import print_function

import wx, sys, time, os
from multiprocessing import Process, Pipe, freeze_support, cpu_count, allow_connection_pickling
from threading import Thread
from datatypes import InfiniteList
from PDSimGUI import pdsim_home_folder
import wx.lib.agw.pybusyinfo as PBI
from math import pi


class RedirectText2Pipe(object):
    """
    An text output redirector
    """
    def __init__(self, pipe_inlet):
        self.pipe_inlet = pipe_inlet
    def write(self, string):
        self.pipe_inlet.send(string)
    def flush(self):
        return None

class Run1(Process):
    """
    A :class:`multiprocessing.Process` class that actually runs one simulation. It does 
    so by importing the build and run functions from the build script that was
    generated by the GUI
    
    """
    def __init__(self, pipe_std, pipe_abort, pipe_results, script_name):
        """
        Parameters
        ----------
        pipe_std: :class:`Pipe <multiprocessing.Pipe>` instance
            The pipe that standard output and standard error is redirected to
        pipe_abort: :class:`Pipe <multiprocessing.Pipe>` instance
            The pipe that is used to accept a request to abort from the GUI by sending ``True`` into the pipe.  'ACK' is sent back through the port when the request to abort is received
        pipe_results: :class:`Pipe <multiprocessing.Pipe>` instance
            The pipe that is used to send a model back to the GUI when it finishes.  The model must be pickle-able
        script_name: string
            The name of the script file that will be imported and run from.  It should provide ``build()`` and ``run()`` functions, and ``run()`` should return the completed model which finished 
        """
        Process.__init__(self)
        #Keep local variables that point to the pipes
        self.pipe_std = pipe_std
        self.pipe_abort = pipe_abort
        self.pipe_results = pipe_results
        self.script_name = script_name
        #Reset the abort flag at instantiation
        self._want_abort = False

    def run(self):
        # Any stdout or stderr output will be redirected to a pipe for passing
        # back to the GUI.  Pipes must be used because they are picklable and
        # otherwise the text output will not show up anywhere
        redir = RedirectText2Pipe(self.pipe_std)
        sys.stdout = redir
        sys.stderr = redir
        
        # Build happens through a dynamically generated build script while 
        # thankfully removes the requirement for picklability
        
        #Get the module name (file name without the .py)
        script_name = self.script_name.split('.', 1)[0]
        
        print('About to run the script file', os.path.join(pdsim_home_folder,self.script_name))

        #Import the script module
        script_module = __import__(script_name, globals(), locals(), [], 0)
        
        #Build the simulation
        self.sim = script_module.build()
        
        #Save the script in the simulation as a string
        self.sim.build_script = open(os.path.join(pdsim_home_folder,self.script_name), 'r').read()
        
        #Run the simulation
        script_module.run(self.sim, pipe_abort = self.pipe_abort)
        
        # Delete a few items that cannot pickle properly
        if hasattr(self.sim,'pipe_abort'):
            
            del self.sim.pipe_abort
            del self.sim.FlowStorage
            del self.sim.Abort #Can't pickle because it is a pointer to a bound method
        
        if not self.sim._want_abort:
            inlet_state_Post = self.sim.inlet_state.copy()
            outlet_state_Post = self.sim.outlet_state.copy()
            
            inlet_state_Post.update(dict(P=inlet_state_Post.p, Q=1))
            outlet_state_Post.update(dict(P=outlet_state_Post.p, Q=1))
			
            Teva = inlet_state_Post.T
            Tcon = outlet_state_Post.T

            print(Teva)
            print(Tcon)
			
			
            temp_folder = pdsim_home_folder
            try:
                os.mkdir(temp_folder)
            except OSError:
                pass
            except WindowsError:
                pass
            identifier = script_name.split('_',1)[1] + '_Tevap={Tevap:.2f}'.format(Tevap=Teva - 273.15)
            try:
                identifier += '_Tcond={Tcond:.2f}'.format(Tcond=Tcon - 273.15)
            except:
                identifier += '_Pcond={Pcond:.2f}'.format(Pcond=self.sim.outlet_state.p/100.)
            identifier += '_SH={SH:.2f}'.format(SH=self.sim.inlet_state.T-Teva)
            identifier += '_{freq:.2f}Hz'.format(freq=self.sim.omega/(2.*pi))
            # identifier = 'PDSimGUI ' + time.strftime('%Y-%m-%d-%H-%M-%S')+'_t'+script_name.split('_')[1]
            csv_path  = os.path.join(temp_folder, identifier + '.csv')
            hdf5_path = os.path.join(temp_folder, identifier + '.h5')
            
            from PDSim.misc.hdf5 import HDF5Writer
            try:
                self.sim.omega = float(self.sim.omega)
                self.sim.eta_motor = float(self.sim.eta_motor)
            except AttributeError:
                pass
            HDF5 = HDF5Writer()
            HDF5.write_to_file(self.sim, hdf5_path)
            # Prune off undesired keys as provided by get_prune_keys function
            HDF5.prune(hdf5_path, self.sim.get_prune_keys())
            self.sim.attach_HDF5_annotations(hdf5_path)
            print('Wrote hdf5 file to', hdf5_path)
            try:
                self.sim.write_mech_csv(csv_path)
                print('Wrote csv file to', csv_path)
            except AttributeError:
                pass
            # Send simulation result back to calling thread
            self.pipe_results.send(hdf5_path)
            print('Sent simulation back to calling thread',end='')
            # Wait for an acknowledgment of receipt
            while not self.pipe_results.poll():
                time.sleep(0.1)
                #Check that you got the right acknowledgment key back
                ack_key = self.pipe_results.recv()
                if not ack_key == 'ACK':
                    raise KeyError
                else:
                    print('Acknowledgment of receipt accepted')
                    break
        else:
            print('Acknowledging completion of abort')
            self.pipe_abort.send('ACK')
        
class WorkerThreadManager(Thread):
    """
    This manager thread creates all the threads that run.  It checks how many processors are available and runs Ncore-1 processes
    
    Runs are consumed from the simulations list one at a time
    """
    def __init__(self,
                 simulations, 
                 stdout_targets, 
                 done_callback = None, 
                 Ncores = None, 
                 main_stdout = None,
                 delete_scripts = True):
        
        """
        Parameters
        ----------
        simulations: list
            A list of script files that have been generated, one per run
        stdout_targets : list of :class:`wx.TextCtrl`
            A list of :class:`wx.TextCtrl` that will be cycled through to provide logging output for each thread
        done_callback : A function to be called when the run completes
        Ncores : integer
            The maximum number of cores to use, otherwise number of cores in computer minus one
        main_stdout : :class:`wx.TextCtrl`
            TextCtrl target for high-level textual output
        """
        
        Thread.__init__(self)
        self.done_callback = done_callback
        self.simulations = simulations
        self.stdout_targets = stdout_targets
        self.stdout_list = InfiniteList(stdout_targets)
        self.main_stdout = main_stdout
        if Ncores is None:
            self.Ncores = max(cpu_count()-1,1)
        else:
            self.Ncores = Ncores
        
        self.threadsList = []
        wx.CallAfter(self.main_stdout.AppendText, "Want to run "+str(len(self.simulations))+" simulations in batch mode; "+str(self.Ncores)+' cores available for computation\n')
            
    def run(self):
        #While simulations left to be run or computation is not finished
        while self.simulations or self.threadsList:
            
            #Add a new thread if possible (leave one core for main GUI)
            if len(self.threadsList) < self.Ncores and self.simulations:
                
                #Get the next simulation to be run
                simulation = self.simulations.pop(0)
                #Get the next target 
                next_stdout_target = self.stdout_list.pop()
                #Start the worker thread
                t = RedirectedWorkerThread(stdout_target = next_stdout_target,
                                           script_name = simulation,
                                           done_callback = self.done_callback,
                                           main_stdout = self.main_stdout
                                           )
                t.daemon = True
                t.start()
                self.threadsList.append(t)
                wx.CallAfter(self.main_stdout.AppendText, 'Adding thread;' + str(len(self.threadsList)) + ' threads active; ' + str(len(self.simulations)) +' queued \n') 
            
            for _thread in reversed(self.threadsList):
                if not _thread.is_alive():
                    _thread.join()
                    self.threadsList.remove(_thread)
                    #Reclaim the stdout TextCtrl for the next run
                    self.stdout_list.prepend(_thread.stdout_target)
                    wx.CallAfter(self.main_stdout.AppendText, 'Thread finished; now '+str(len(self.threadsList))+ ' threads active; ' + str(len(self.simulations)) +' queued \n')
            
            #Only check every two seconds in order to keep the GUI responsive and not lock up one core
            time.sleep(2.0)
    
    def abort(self):
        """
        Pass the message to quit to all the threads; don't run any that are queued
        """
        dlg = wx.MessageDialog(None,"Are you sure you want to kill the current runs?",caption ="Kill Batch?",style = wx.OK|wx.CANCEL)
        if dlg.ShowModal() == wx.ID_OK:
            #message = "Aborting in progress, please wait..."
            #busy = PBI.PyBusyInfo(message, parent = None, title = "Aborting")
            
            # Empty the list of simulations to run
            self.simulations = []
            
            for _thread in self.threadsList:
                #Send the abort signal
                _thread.abort()
#                #Wait for it to finish up
#                _thread.join()
            #del busy
            
        dlg.Destroy()
        
class RedirectedWorkerThread(Thread):
    """Worker Thread Class."""
    def __init__(self, script_name, stdout_target = None,  kwargs = None, done_callback = None, add_results = None, main_stdout = None):
        """Init Worker Thread Class."""
        Thread.__init__(self)
        self.script_name = script_name
        self.stdout_target = stdout_target
        self.done_callback = done_callback
        self.add_results = add_results
        self.main_stdout = main_stdout
        
        self._want_abort = False
        
    def run(self):
        """
        In this function, actually run the process and pull any output from the 
        pipes while the process runs
        """
        sim = None
        pipe_outlet, pipe_inlet = Pipe(duplex = False)
        pipe_abort_outlet, pipe_abort_inlet = Pipe(duplex = True)
        pipe_results_outlet, pipe_results_inlet = Pipe(duplex = True)

        p = Run1(pipe_inlet, pipe_abort_outlet, pipe_results_inlet, self.script_name)
        p.daemon = True
        p.start()
        
        while p.is_alive():
                
            #If the manager is asked to quit
            if self._want_abort == True:
                # Tell the process to abort, passes message to simulation run
                pipe_abort_inlet.send(True)
                # Wait until it acknowledges the kill by sending back 'ACK'
                while not pipe_abort_inlet.poll():
                    time.sleep(0.1)
                    # Collect all display output from process while you wait
                    while pipe_outlet.poll():
                        wx.CallAfter(self.stdout_target.AppendText, pipe_outlet.recv())
                        
                abort_flag = pipe_abort_inlet.recv()
                if abort_flag == 'ACK':
                    hdf5_path = None
                    break
                else:
                    raise ValueError('abort pipe should have received a value of "ACK"')
                
            # Collect all display output from process
            while pipe_outlet.poll():
                wx.CallAfter(self.stdout_target.AppendText, pipe_outlet.recv())
            time.sleep(1)
            
            # Get back the results from the simulation process if they are waiting
            if pipe_results_outlet.poll():
                hdf5_path = pipe_results_outlet.recv()
                pipe_results_outlet.send('ACK')
                break
            else:
                hdf5_path = None
        
        # Flush out any remaining stuff left in the pipe after process ends
        while pipe_outlet.poll():
            wx.CallAfter(self.stdout_target.AppendText, pipe_outlet.recv())
            time.sleep(0.1)
        
        if self._want_abort == True:
            print(self.name+": Process has aborted successfully")
        else:
            wx.CallAfter(self.stdout_target.AppendText, self.name+": Process is done")
            if hdf5_path is not None:
                "Send the data back to the GUI"
                wx.CallAfter(self.done_callback, hdf5_path)
            else:
                print("Didn't get any simulation data")
        return 1
        
    def abort(self):
        """abort worker thread."""
        wx.CallAfter(self.main_stdout.WriteText, self.name + ': Thread readying for abort\n')
        # Method for use by main thread to signal an abort
        self._want_abort = True