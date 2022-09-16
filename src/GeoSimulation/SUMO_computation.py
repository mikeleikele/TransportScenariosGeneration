import os 
import xml.etree.cElementTree as ET
from .GeoGraph import *
from .SUMO_network import *
from .SUMO_routes import *
from .SUMO_simulation import *


class SUMO_computation():

    def __init__(self, sumo_tool_folder, folder_name, name_file, network_settings, edgeStats_settings=None, simulation_route_mode="random",  verbose=True):
        self.sumo_tool_folder = sumo_tool_folder
        if not os.path.exists(self.sumo_tool_folder):
            raise SUMO_INSTALL_Exception__ToolFolder(self.sumo_tool_folder)
        
        self.folder_name = folder_name
        if not os.path.exists(self.folder_name):
            os.makedirs(self.folder_name)
        self.name_file = name_file
        self.edgeStats_settings = edgeStats_settings

        self.net_simulation = SUMO_network(sumo_tool_folder=self.sumo_tool_folder,folder_simulationName = self.folder_name, name_simulationFile=self.name_file, network_settings=network_settings, verbose=verbose)
        
        if simulation_route_mode not in ["random","demand"]:
            raise SUMO_computation_Exception__ModeNotRecognized(simulation_route_mode)
        else:
            self.simulation_route_mode = simulation_route_mode


    def generate_simulation(self, n_vehicle=200, verbose=False):
        self.network_file = self.net_simulation.network_generation(verbose=verbose)
        self.geometry_file = self.net_simulation.geometry_generation(verbose=verbose)
        
        #TO DO: refactoring !
        if self.simulation_route_mode == "random":
            self.routes_simulation = SUMO_routes(sumo_tool_folder=self.sumo_tool_folder, folder_simulationName = self.folder_name, name_simulationFile=self.name_file, networkObj=self.net_simulation, verbose=verbose)
            self.generate_routes(n_vehicle=n_vehicle,verbose=verbose)
        elif self.simulation_route_mode == "demand":
            return 0
            #self.osm_generate_simulation_demand(verbose)
        else:
            return 0
                
    
    def generate_routes(self, n_vehicle, verbose=False):
        self.flows_file = self.routes_simulation.flows_generation_random(n_vehicle=n_vehicle,verbose=verbose)
        self.routes_file = self.routes_simulation.routes_generation(verbose=verbose)
        self.continuos_reroutes_file = self.routes_simulation.continuous_rerouting_generation(verbose=verbose)

    def esecute_simulation(self, verbose=False):
        self.sumo_simulation = SUMO_simulation(sumo_tool_folder=self.sumo_tool_folder, folder_simulationName = self.folder_name, name_simulationFile=self.name_file, networkObj=self.net_simulation, routesObj=self.routes_simulation, edgeStats_settings=self.edgeStats_settings, verbose=verbose)
        self.sumo_simulation.esecute_simulation(verbose=verbose)
        self.config_file = self.sumo_simulation.get_configFile()

    
    """
    TO DO

    def osm_generate_simulation_demand(self, verbose=False):
        #self.network_file = self.net_simulation.network_generation(verbose=True)
        #self.geometry_file = self.geometry_generation(folder_name=self.folder_name, name_file=self.name_file, network_file=self.network_file, geometric_maps_options=['all'], osm_map_name=self.osm_map_name, verbose=verbose)        
        #activitygen-example.stat.xml \
        #vehicles_generation_cityDemand
        self.stats_file = f"{self.osm_map_name}_statistics_files.xml"
        self.citiesdemand_file = self.vehicles_generation_cityDemand(folder_name=self.folder_name, name_file=self.name_file, network_file=self.network_file, stats_file=self.stats_file, verbose=verbose)
        
    def vehicles_generation_cityDemand(self, folder_name, name_file, network_file, stats_file, verbose=False):
        #https://sumo.dlr.de/docs/Tools/Trip.html
        citydemand_file = f"{name_file}_demand__trips.xml"
        sumo_cmd = f"activitygen --net-file {folder_name}/{network_file} --stat-file {folder_name}/{stats_file} --output-file {folder_name}/{citydemand_file} --random"
        if verbose:
            print("\ndemand generation\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)
        return citydemand_file
    
    """



class SUMO_INSTALL_Exception__ToolFolder(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,msg):
        self.msg = msg
          
    def __str__(self):
        return f"SUMO tool folder '{self.msg}' not found."