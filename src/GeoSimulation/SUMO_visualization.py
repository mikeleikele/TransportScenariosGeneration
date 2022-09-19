import os 
import xml.etree.cElementTree as ET
from .GeoGraph import *
from .SUMO_network import *
from .SUMO_routes import *
from ..tool.utils_matplot import UtilsMatplot


class SUMO_visualization():

    def __init__(self, sumo_tool_folder, folder_simulationName, name_simulationFile, verbose=False):
        self.folder_simulationName = folder_simulationName
        self.name_simulationFile = name_simulationFile
        self.verbose = verbose
        self.sumo_tool_folder = sumo_tool_folder
        
    
    def plotAttributes(self):
        #https://sumo.dlr.de/docs/Tools/Visualization.html
        raise NotImplementedError()
    

    """
    attr_code::
        t: Time in s
        d: Distance driven (starts with 0 at the first fcd datapoint for each vehicle). Distance is computed based on speed using Euler-integration. Set option --ballistic for ballistic integration.
        a: Acceleration
        s: Speed (m/s)
        i: Vehicle angle (navigational degrees)
        x: X-Position in m
        y: Y-Position in m
        k: Kilometrage (requires --fcd-output.distance)
        g: gap to leader (requires --fcd-output.max-leader-distance)
    """
    def plotTrajectories(self, filename_output, simulObj, attr_code="ts", routeList=None, edgesList=None, verbose=False):
        fcdFile = simulObj.get_fcdFile()

        for char_code in attr_code:
            if char_code not in ['t', 's', 'd', 'a', 'i', 'x', 'y', 'k']:
                raise NotImplementedError()
        sumo_cmd = f'python "{self.sumo_tool_folder}/plot_trajectories.py" {self.folder_simulationName}/{fcdFile} --trajectory-type {attr_code} --output {self.folder_simulationName}/{filename_output}'
        if edgesList is not None:
            sumo_cmd = sumo_cmd +f' --filter-edges {edgesList}'
        elif  routeList is not None:
            sumo_cmd = sumo_cmd +f' --filter-route {routeList}'
        
        if verbose:
            print("\nplot_trajectories\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)
    
    def plotNet(self, filename_output, networkFile, fileinput, key_colors, key_widths, color_map, verbose=True):
        if not UtilsMatplot.isColorMap(color_map):
            raise SUMO_visualization_Exception__ColorMapNotExist(color_map)
        sumo_cmd = f'python "{self.sumo_tool_folder}/visualization\plot_net_dump.py" --net {self.folder_simulationName}/{networkFile} --dump-inputs {self.folder_simulationName}/{fileinput} --measures {key_colors},{key_widths} --colormap {color_map} --min-color-value -.1 --max-color-value .1 --max-width-value .1  --max-width 3 --min-width .5 --output {self.folder_simulationName}/{filename_output}'
        
        if verbose:
            print("\nnetTrajectories\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)



class SUMO_visualization_Exception__ColorMapNotExist(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,msg):
        self.msg = msg
          
    def __str__(self):
        return f"'{self.msg}' is not a color map."