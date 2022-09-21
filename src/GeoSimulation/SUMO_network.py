import os 
from .GeoGraph import *
from tqdm.auto import tqdm

class SUMO_network():

    """
    network_type : generate, maps
    generate::
        geometry : grid,spider,rand
        grid::
            grid.number
            grid.length
        spider::
            arm-number
            circle-number
            space-radius
            omit-center
        random
            rand.iterations
            bidi-probability
            rand.connectivity
            random-lanenumber
    maps::
        osm_maps_name
        remove_geometry
    """
    def __init__(self, sumo_tool_folder, folder_simulationName, name_simulationFile, network_settings, verbose=False):
        self.folder_simulationName = folder_simulationName
        self.name_simulationFile = name_simulationFile
        self.verbose = verbose
        self.sumo_tool_folder = sumo_tool_folder
        
        self.network_type = network_settings['network_type']
        network_type_settings = self.check_isKey(network_settings,self.network_type)
        
        if  self.network_type == "generate":
            self.geometry_settings =None
            self.geometry = self.check_isKey(network_type_settings,"geometry")
            geometry_settings = self.check_isKey(network_type_settings,self.geometry)
            
            if self.geometry == "grid":
                self.number = self.check_isKey(geometry_settings,"number")
                self.length = self.check_isKey(geometry_settings,"length")
            elif self.geometry == "spider":
                self.arm_number = self.check_isKey(geometry_settings,"arm-number")
                self.circle_number = self.check_isKey(geometry_settings,"circle-number")
                self.space_radius = self.check_isKey(geometry_settings,"space-radius")
                self.omit_center = self.check_isKey(geometry_settings,"omit-center")
            elif self.geometry == "rand":
                self.iterations = self.check_isKey(geometry_settings,"iterations")
                self.bidi_probability = self.check_isKey(geometry_settings,"bidi-probability")
                self.rand_connectivity = self.check_isKey(geometry_settings,"connectivity")
            else:
                raise SUMO_network_Exception__NetworkTypeNotFound(f"{self.network_type}::{self.geometry}")
        elif self.network_type == "maps":
            self.osm_maps_name = self.check_isKey(network_type_settings,"osm_maps_name")
            self.remove_geometry = self.check_isKey(network_type_settings,"remove_geometry")
            self.osm_maps_folder = self.check_isKey(network_type_settings,"osm_maps_folder", False)
            
            self.geometry_settings = self.check_isKey(network_type_settings,"geometry_settings", False)
            
            if self.osm_maps_folder is None:
                self.osm_maps_folder = f"data\maps"
        else:
            raise SUMO_network_Exception__NetworkTypeNotFound(self.network_type)

    def check_isKey(self, dict_in, key, raiseExc=True):
        if key in dict_in:
            return dict_in[key]
        else:
            if raiseExc:
                raise SUMO_network_Exception__NetworkKeyNotFound(self.network_type,key)
            else:
                return None


    def get_networkFile(self):
        if self.network_file is None:
            raise SUMO_network_Exception__FileNotInit("Network")
        return self.network_file

    def get_geometrykFile(self,raiseExc):
        if self.geometry_file is None and raiseExc==True:
            raise SUMO_network_Exception__FileNotInit("Geometry")
        return self.geometry_file


    def network_generation(self, verbose=False):           
        if  self.network_type == "generate": 
            self.network_file = f"{self.name_simulationFile}.net.xml"
            if self.geometry == "grid":
                sumo_cmd = f"netgenerate --grid --grid.number={self.number} --grid.length={self.length} --output-file={self.folder_simulationName}/{self.network_file}"
            elif self.geometry == "spider":
                sumo_cmd = f"netgenerate --spider  --spider.arm-number={self.arm_number} --spider.circle-number={self.circle_number} --spider.space-radius={self.space_radius} --spider.omit-center={self.omit_center}  --output-file={self.folder_simulationName}/{self.network_file}"
            elif self.geometry == "rand":
                sumo_cmd = f"netgenerate --rand  --rand.iterations={self.iterations} --bidi-probability={self.bidi_probability} --rand.connectivity={self.rand_connectivity}  --output-file={self.folder_simulationName}/{self.network_file}"
            
            if verbose:
                print("\nnetwork generation\t>>\t",sumo_cmd,"")
            
            p_bar = tqdm(range(10))
            p_bar.update(1)
            p_bar.refresh()
            os.system(sumo_cmd)
            p_bar.update(10)
            p_bar.refresh()
        elif self.network_type == "maps" or self.network_type == "map":
            if not os.path.isfile(f"{self.osm_maps_folder}\GEO__{self.osm_maps_name}.osm"):
                raise SUMO_simulation_Exception__FileMapNotFound(self.osm_maps_name,self.osm_maps_GEO_filepath)
            
            self.network_file = f"{self.name_simulationFile}.net.xml"
            if self.remove_geometry:
                sumo_cmd = f"netconvert --osm-files {self.osm_maps_folder}\GEO__{self.osm_maps_name}.osm --output-file={self.folder_simulationName}/{self.network_file} --geometry.remove --remove-edges.isolated --roundabouts.guess  --ramps.guess --junctions.join --tls.guess-signals --tls.join --tls.default-type actuated"
            else:
                sumo_cmd = f"netconvert --osm-files {self.osm_maps_folder}\GEO__{self.osm_maps_name}.osm --output-file={self.folder_simulationName}/{self.network_file}  --ramps.guess --junctions.join --tls.guess-signals --tls.discard-simple --tls.join --tls.default-type actuated"
            
            if verbose:
                print("\nnetwork convertion\t>>\t",sumo_cmd,"")
            else:
                sumo_cmd = sumo_cmd + f" --no-warnings"
            p_bar = tqdm(range(10))
            p_bar.update(1)
            p_bar.refresh()
            os.system(sumo_cmd)
            p_bar.update(10)
            p_bar.refresh()
        
        return self.network_file

    def geometry_generation(self, force_overwrite=False, verbose=False):        
        if self.geometry_settings is None:
            self.geometry_file = None
            return self.geometry_file
        else:
            geometric_lists = [] 
            #geometric_maps_options = OSMgeometry.getOptions()
            poi_options = OSMgeometry.getOptions()
            tqdm_bar = tqdm(range(len(self.geometry_settings)))
            for i in tqdm_bar:
                key = self.geometry_settings[i]
                if key not in poi_options:
                    raise SUMO_simulation_Exception__GeometryNotExist(key)
                
                
                geometric_lists.append(self.geometry_keymap_generation(osm_map_name= f"{key}", force_overwrite=force_overwrite, verbose=verbose))
                
            self.geometry_file = ','.join(geometric_lists) 
            return self.geometry_file
        

    def geometry_keymap_generation(self, osm_map_name, force_overwrite, verbose=False):
        key_geometry_file = f"{self.name_simulationFile}.geometry.{osm_map_name}.xml"
        if not os.path.isfile(f"{self.folder_simulationName}/{key_geometry_file}") or force_overwrite:
            sumo_cmd = f"polyconvert --net-file {self.folder_simulationName}/{self.network_file} --output-file={self.folder_simulationName}/{key_geometry_file} --osm-files {self.osm_maps_folder}/{osm_map_name}.osm  --all-attributes"
            #--ignore-errors
            if verbose:
                print("\ngeometry generation\t>>\t",sumo_cmd,"")
            os.system(sumo_cmd)
        return key_geometry_file


class SUMO_network_Exception__NetworkTypeNotFound(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,network_type):
        self.network_type = network_type
    def __str__(self):
        return f"'{self.network_type}' is not a network type recognized."

class SUMO_network_Exception__NetworkKeyNotFound(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,network_type,key):
        self.network_type = network_type
        self.key = key

    def __str__(self):
        return f"A '{self.network_type}' network require a '{self.key}' key in the network's settings dict."




class SUMO_network_Exception__FileNotInit(Exception):
    def __init__(self,key):
        self.key = key

    def __str__(self):
        return f"{key} not inizialized."


class SUMO_simulation_Exception__FileMapNotFound(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,maps_filename,maps_path):
        self.maps_filename = maps_filename
        self.maps_path = maps_path

    def __str__(self):
        return f"Openstreet map '{self.maps_filename}' not found on path '{self.maps_path}'."

class SUMO_simulation_Exception__GeometryNotExist(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,key):
        self.key = key

    def __str__(self):
        return f"'{self.key}' unrecognized geometry type."