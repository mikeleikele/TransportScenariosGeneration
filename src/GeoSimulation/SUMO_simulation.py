import os 
import xml.etree.cElementTree as ET
from .GeoGraph import *

class SUMO_simulation():

    def __init__(self, sumo_tool_folder, name_file, folder_name, simulation_network_mode="naive",simulation_route_mode="random", osm_map_name=None, osm_map_path=None):
        self.sumo_tool_folder = sumo_tool_folder
        if not os.path.exists(self.sumo_tool_folder):
            raise SUMO_INSTALL_Exception__ToolFolder(self.sumo_tool_folder)
        
        self.folder_name = folder_name
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        self.name_file = name_file
        if simulation_network_mode not in ["naive","openstreenmap","osm"]:
            raise SUMO_simulation_Exception__ModeNotRecognized(simulation_network_mode)
        else:
            self.simulation_network_mode = simulation_network_mode
        
        if simulation_route_mode not in ["random","demand"]:
            raise SUMO_simulation_Exception__ModeNotRecognized(simulation_route_mode)
        else:
            self.simulation_route_mode = simulation_route_mode

        
        
        if self.simulation_network_mode == "openstreenmap" or self.simulation_network_mode == "osm":
            self.osm_map_name = osm_map_name
            if osm_map_path is None:
                self.osm_map_GEO_filepath = f"data\maps\GEO__{osm_map_name}.osm"                
                if not os.path.isfile(self.osm_map_GEO_filepath):
                    raise SUMO_simulation_Exception__FileMapNotFound(self.osm_map_name,self.osm_map_GEO_filepath)
            else:
                self.osm_map_GEO_filepath = f"{osm_map_path}\GEO__{osm_map_name}.osm"
                if not os.path.isfile(self.osm_map_GEO_filepath):
                    raise SUMO_simulation_Exception__FileMapNotFound(self.osm_map_name,self.osm_map_GEO_filepath)
                


    def generate_simulation(self, verbose=False):
        if self.simulation_network_mode == "naive":
            self.naive_generate_simulation(verbose)
        elif self.simulation_network_mode == "openstreenmap" or self.simulation_network_mode == "osm":
            if self.simulation_route_mode == "random":
                self.osm_generate_simulation_random(verbose)
            elif self.simulation_route_mode == "demand":
                self.osm_generate_simulation_demand(verbose)


#netconvert --osm-files data\maps\GEO__bassa.osm

    def naive_generate_simulation(self, verbose=False):
        self.network_file = self.network_generation_grid(folder_name=self.folder_name, name_file=self.name_file, grids=5, lanes=3, length=200, verbose=verbose)
        self.vehicles_file = self.vehicles_generation_random(folder_name=self.folder_name, name_file=self.name_file, network_file=self.network_file,  begin_time=0, end_time=1, period=1, vehicles=200,verbose=verbose)
        self.routes_file = self.routes_generation(folder_name=self.folder_name, name_file=self.name_file, network_file=self.network_file, vehicles_file=self.vehicles_file, begin_time=0, end_time=10000, verbose=verbose)
        self.continuos_reroutes_file = self.continuous_rerouting_generation(folder_name=self.folder_name, name_file=self.name_file, network_file=self.network_file, end_time=10000, verbose=verbose)
        self.write_sumo_config_file(folder_name=self.folder_name, name_file=self.name_file)

    def osm_generate_simulation_random(self, verbose=False):
        self.network_file = self.network_generation_from_osm(folder_name=self.folder_name, name_file=self.name_file, osm_map_path=self.osm_map_GEO_filepath, osm_map_name=self.osm_map_name, verbose=verbose)
        self.vehicles_file = self.vehicles_generation_random(folder_name=self.folder_name, name_file=self.name_file, network_file=self.network_file,  begin_time=0, end_time=1, period=1, vehicles=2000,verbose=verbose)
        self.geometry_file = self.geometry_generation(folder_name=self.folder_name, name_file=self.name_file, network_file=self.network_file, geometric_maps_options=['all'], osm_map_name=self.osm_map_name, verbose=verbose)        
        self.routes_file = self.routes_generation(folder_name=self.folder_name, name_file=self.name_file, network_file=self.network_file, vehicles_file=self.vehicles_file, begin_time=0, end_time=10000, verbose=verbose)
        self.continuos_reroutes_file = self.continuous_rerouting_generation(folder_name=self.folder_name, name_file=self.name_file, network_file=self.network_file, end_time=10000, verbose=verbose)
        self.write_sumo_config_file(folder_name=self.folder_name, name_file=self.name_file)
   
    def osm_generate_simulation_demand(self, verbose=False):
        self.network_file = self.network_generation_from_osm(folder_name=self.folder_name, name_file=self.name_file, osm_map_path=self.osm_map_GEO_filepath, osm_map_name=self.osm_map_name, verbose=verbose)
        #activitygen-example.stat.xml \
        #vehicles_generation_cityDemand
        self.stats_file = f"{self.osm_map_name}_statistics_files.xml"
        self.citiesdemand_file = self.vehicles_generation_cityDemand(folder_name=self.folder_name, name_file=self.name_file, network_file=self.network_file, stats_file=self.stats_file, verbose=verbose)
        self.geometry_file = self.geometry_generation(folder_name=self.folder_name, name_file=self.name_file, network_file=self.network_file, geometric_maps_options=['all'], osm_map_name=self.osm_map_name, verbose=verbose)        


    def network_generation_grid(self, folder_name, name_file, grids,lanes,length, verbose=False):
        network_file = f"{folder_name}/{name_file}_network__grid{grids}_lanes{lanes}_length{length}.xml"
        sumo_cmd = f"netgenerate --grid --grid.number={grids} -L={lanes} --grid.length={length} --output-file={network_file}"
        if verbose:
            print("\nnetwork generation\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)
        return network_file

    def network_generation_from_osm(self, folder_name, name_file, osm_map_path, osm_map_name, remove_geom=True, verbose=False):
        network_file = f"{name_file}_network__osm{osm_map_name}.xml"
        if remove_geom:
            sumo_cmd = f"netconvert --osm-files {osm_map_path} --output-file={folder_name}/{network_file} --geometry.remove --remove-edges.isolated --roundabouts.guess  --ramps.guess --junctions.join --tls.guess-signals --tls.join --tls.default-type actuated"
        else:
            sumo_cmd = f"netconvert --osm-files {osm_map_path} --output-file={folder_name}/{network_file}  --ramps.guess --junctions.join --tls.guess-signals --tls.discard-simple --tls.join --tls.default-type actuated"
        if verbose:
            print("\nnetwork convertion\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)
        return network_file
      
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

    def geometry_generation(self, folder_name, name_file, network_file, osm_map_name, geometric_maps_options = None, verbose=False):
        geometric_maps_filelist = [] 
        if geometric_maps_options is None:
            geometric_maps_options = OSMgeometry.getOptions()
        else:
            poi_options = OSMgeometry.getOptions()
            for key in geometric_maps_options:
                if key not in poi_options:
                    print("not found geometry option to do exc")
        for key in geometric_maps_options:
            print(f"POI__{key}__{osm_map_name}")
            geometric_maps_filelist.append(self.geometry_keymap_generation(folder_name, name_file, network_file, f"POI__{key}__{osm_map_name}.osm",False, False))
        return ','.join(geometric_maps_filelist) 

    def geometry_keymap_generation(self, folder_name, name_file, network_file,  osm_map_name_key, force, verbose=False):
        geometry_file = f"{name_file}_geometry__{osm_map_name_key}.xml"
        if not os.path.isfile(f"{folder_name}/{geometry_file}") or force:
            sumo_cmd = f"\ -n {folder_name}/{network_file} --output-file={folder_name}/{geometry_file} --osm-files data/maps/{osm_map_name_key}  --all-attributes"
            #--ignore-errors
            if verbose:
                print("\ngeometry generation\t>>\t",sumo_cmd,"")
            os.system(sumo_cmd)
        return geometry_file

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

class SUMO_simulation_Exception__ModeNotRecognized(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,mode):
        self.mode = mode
          
    def __str__(self):
        return f"SUMO simultion mode {self.mode} not recognized."


class SUMO_simulation_Exception__FileMapNotFound(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,maps_filename,maps_path):
        self.maps_filename = maps_filename
        self.maps_path = maps_path

    def __str__(self):
        return f"Openstreet map '{self.maps_filename}' not found on path '{self.maps_path}'."