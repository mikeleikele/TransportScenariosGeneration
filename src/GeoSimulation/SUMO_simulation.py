import os 
import xml.etree.cElementTree as ET
from .GeoGraph import *
from .SUMO_network import *

class SUMO_simulation():

    def __init__(self, sumo_tool_folder, folder_name, name_file, network_settings, osm_map_name, simulation_route_mode="random",  verbose=True):
        self.sumo_tool_folder = sumo_tool_folder
        if not os.path.exists(self.sumo_tool_folder):
            raise SUMO_INSTALL_Exception__ToolFolder(self.sumo_tool_folder)
        
        self.folder_name = folder_name
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        self.name_file = name_file
        
        self.net_simulation = SUMO_network(folder_simulationName = folder_name, name_simulationFile=name_file, network_settings=network_settings, verbose=verbose)
        
        if simulation_route_mode not in ["random","demand"]:
            raise SUMO_simulation_Exception__ModeNotRecognized(simulation_route_mode)
        else:
            self.simulation_route_mode = simulation_route_mode

    def generate_simulation(self, verbose=False):
        self.network_file = self.net_simulation.network_generation(verbose)
        self.geometry_file = self.net_simulation.geometry_generation(verbose)

        self.simulation_network_mode = "maps"
        if self.simulation_network_mode == "grid":
            self.grid_generate_simulation(verbose)
        elif self.simulation_network_mode == "openstreenmap" or self.simulation_network_mode == "maps":
            if self.simulation_route_mode == "random":
                self.osm_generate_simulation_random(verbose)
            elif self.simulation_route_mode == "demand":
                self.osm_generate_simulation_demand(verbose)

    def grid_generate_simulation(self, verbose=False):
        
        #self.network_file = self.net_simulation.network_generation(verbose=True)
        self.vehicles_file = self.vehicles_generation_random(folder_name=self.folder_name, name_file=self.name_file, network_file=self.network_file,  begin_time=0, end_time=1, period=1, vehicles=200,verbose=verbose)
        self.routes_file = self.routes_generation(folder_name=self.folder_name, name_file=self.name_file, network_file=self.network_file, vehicles_file=self.vehicles_file, begin_time=0, end_time=10000, verbose=verbose)
        #self.geometry_file = None
        self.continuos_reroutes_file = self.continuous_rerouting_generation(folder_name=self.folder_name, name_file=self.name_file, network_file=self.network_file, end_time=10000, verbose=verbose)
        self.write_sumo_config_file(folder_name=self.folder_name, name_file=self.name_file)

    def osm_generate_simulation_random(self, verbose=False):
        #self.network_file = self.net_simulation.network_generation(verbose=True)
        #self.geometry_file = self.geometry_generation(folder_name=self.folder_name, name_file=self.name_file, network_file=self.network_file, geometric_maps_options=['all'], osm_map_name=self.osm_map_name, verbose=verbose)        
        self.vehicles_file = self.vehicles_generation_random(folder_name=self.folder_name, name_file=self.name_file, network_file=self.network_file,  begin_time=0, end_time=1, period=1, vehicles=2000,verbose=verbose)
        self.routes_file = self.routes_generation(folder_name=self.folder_name, name_file=self.name_file, network_file=self.network_file, vehicles_file=self.vehicles_file, begin_time=0, end_time=10000, verbose=verbose)
        self.continuos_reroutes_file = self.continuous_rerouting_generation(folder_name=self.folder_name, name_file=self.name_file, network_file=self.network_file, end_time=10000, verbose=verbose)
        self.write_sumo_config_file(folder_name=self.folder_name, name_file=self.name_file)
   
    def osm_generate_simulation_demand(self, verbose=False):
        #self.network_file = self.net_simulation.network_generation(verbose=True)
        #self.geometry_file = self.geometry_generation(folder_name=self.folder_name, name_file=self.name_file, network_file=self.network_file, geometric_maps_options=['all'], osm_map_name=self.osm_map_name, verbose=verbose)        
        #activitygen-example.stat.xml \
        #vehicles_generation_cityDemand
        self.stats_file = f"{self.osm_map_name}_statistics_files.xml"
        self.citiesdemand_file = self.vehicles_generation_cityDemand(folder_name=self.folder_name, name_file=self.name_file, network_file=self.network_file, stats_file=self.stats_file, verbose=verbose)
        

      
    def vehicles_generation_random(self, folder_name, name_file, network_file, vehicles,begin_time, end_time, period, verbose=False):
        #https://sumo.dlr.de/docs/Tools/Trip.html
        vehicles_file = f"{name_file}_vehicles__beg{begin_time}_end{end_time}_per{period}_veh{vehicles}.xml"
        sumo_cmd = f'python "{self.sumo_tool_folder}/randomTrips.py" -n {folder_name}/{network_file} -o {folder_name}/{vehicles_file} --begin {begin_time} --end {end_time} --period {period} --flows {vehicles}'
        if verbose:
            print("\nvehicles generation\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)
        return vehicles_file

    def vehicles_generation_cityDemand(self, folder_name, name_file, network_file, stats_file, verbose=False):
        #https://sumo.dlr.de/docs/Tools/Trip.html
        citydemand_file = f"{name_file}_demand__trips.xml"
        sumo_cmd = f"activitygen --net-file {folder_name}/{network_file} --stat-file {folder_name}/{stats_file} --output-file {folder_name}/{citydemand_file} --random"
        if verbose:
            print("\ndemand generation\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)
        return citydemand_file

    

    def routes_generation(self, folder_name, name_file, network_file, vehicles_file, begin_time, end_time, verbose=False):
        #https://sumo.dlr.de/docs/jtrrouter.html
        routes_file = f"{name_file}_routes__beg{begin_time}_end{end_time}.xml"
        sumo_cmd = f"jtrrouter --route-files={folder_name}/{vehicles_file} --net-file={folder_name}/{network_file} --output-file={folder_name}/{routes_file} --begin {begin_time}  --end {end_time} --accept-all-destinations"        
        if verbose:
            print("\nroutes generation\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)
        return routes_file
    
    def routes_generation_duarouter(self, folder_name, name_file, network_file, citiesdemand_file, begin_time, end_time, verbose=False):
        #https://sumo.dlr.de/docs/Demand/Activity-based_Demand_Generation.html
        duarouter_file = f"{name_file}_duarouter__routes.xml"
        sumo_cmd = f"duarouter --route-files={folder_name}/{citiesdemand_file} --net-file={folder_name}/{network_file} --output-file={folder_name}/{duarouter_file} --ignore-errors"        
        if verbose:
            print("\nduarouter_file generation\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)
        return duarouter_file

    def continuous_rerouting_generation(self, folder_name, name_file, network_file, end_time, verbose=False):
        #https://sumo.dlr.de/docs/Tools/Misc.html#generatecontinuousrerouterspy
        continuos_reroutes_file = f"{name_file}_continuous_rerouting_generation__end{end_time}.xml"
        sumo_cmd = f'python "{self.sumo_tool_folder}/generateContinuousRerouters.py" -n {folder_name}/{network_file} -o {folder_name}/{continuos_reroutes_file} --end {end_time}'
        if verbose:
            print("\ncontinuous rerouting generation\t>>\t",sumo_cmd)
        os.system(sumo_cmd)
        return continuos_reroutes_file    

    def download_simulation(self,verbose=False):
        zip_cmd = f"zip -r zip_{self.folder_name}.zip {self.folder_name}"
        os.system(zip_cmd)

    def write_sumo_config_file(self, folder_name, name_file):
        filepath = f"{folder_name}/{name_file}.sumocfg"
        
        configuration = ET.Element("configuration")
        k_input = ET.SubElement(configuration, "input")
        ET.SubElement(k_input, "net-file", value=f"{self.network_file}")
        ET.SubElement(k_input, "route-files", value=f"{self.routes_file}")
        if self.geometry_file is None:
            ET.SubElement(k_input, "additional-files", value=f"{self.continuos_reroutes_file}")
        else:
            ET.SubElement(k_input, "additional-files", value=f"{self.continuos_reroutes_file},{self.geometry_file}")

        k_time = ET.SubElement(configuration, "time")
        ET.SubElement(k_time, "begin", value="0")
        ET.SubElement(k_time, "end", value="10000")

        k_output = ET.SubElement(configuration, "output")
        ET.SubElement(k_output, "fcd-output", value=f"{name_file}_simul_output.xml")
        
        tree = ET.ElementTree(configuration)
        ET.indent(tree, space="\t", level=0)
        tree.write(filepath, encoding="utf-8")


class SUMO_INSTALL_Exception__ToolFolder(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,msg):
        self.msg = msg
          
    def __str__(self):
        return f"SUMO tool folder '{self.msg}' not found."