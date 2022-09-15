import os 
from .SUMO_network import *

class SUMO_routes():

    def __init__(self, sumo_tool_folder, folder_simulationName, name_simulationFile, networkObj, verbose=False):
        self.folder_simulationName = folder_simulationName
        self.name_simulationFile = name_simulationFile
        self.verbose = verbose
        self.sumo_tool_folder = sumo_tool_folder

        if isinstance(networkObj, SUMO_network):
            self.network_file = networkObj.get_networkFile()
        else:
            raise SampleTrips_Exception__SUMO_networkInstance(networkObj)
    
    def get_routesFile(self):
        if self.routes_file is None:
            raise SUMO_routes_Exception__FileNotInit("Routes")
        return self.routes_file

    def get_continuous_reroutingFile(self):
        if self.continuos_reroutes_file is None:
            raise SUMO_routes_Exception__FileNotInit("Continuous_rerouting")
        return self.continuos_reroutes_file

    def flows_generation_random(self, n_vehicle, begin_time=0, end_time=3600, period=1, verbose=False):
        #https://sumo.dlr.de/docs/Tools/Trip.html
        self.n_vehicle = n_vehicle
        self.flows_file = f"{self.name_simulationFile}_flows__beg{begin_time}_end{end_time}_per{period}_veh{self.n_vehicle}.rou.xml"
        
        sumo_cmd = f'python "{self.sumo_tool_folder}/randomTrips.py" --net-file {self.folder_simulationName}\\{self.network_file} --output-trip-file {self.folder_simulationName}\\{self.flows_file} --begin {begin_time} --end {end_time} --period {period} --flows {self.n_vehicle}'
        if verbose:
            print("\nflows generation\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)
        return self.flows_file
        
    def routes_generation(self, begin_time=0, end_time=3600, verbose=False):
        #https://sumo.dlr.de/docs/jtrrouter.html
        self.routes_file = f"{self.name_simulationFile}_routes__beg{begin_time}_end{end_time}.xml"
        sumo_cmd = f"jtrrouter --route-files={self.folder_simulationName}\\{self.flows_file} --net-file={self.folder_simulationName}\\{self.network_file} --output-file={self.folder_simulationName}\\{self.routes_file} --begin {begin_time}  --end {end_time} --accept-all-destinations"        
        if verbose:
            print("\nroutes generation\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)
        return self.routes_file
    
    def routes_generation_duarouter(self, citiesdemand_file, begin_time=0, end_time=3600, verbose=False):
        #https://sumo.dlr.de/docs/Demand/Activity-based_Demand_Generation.html
        self.routes_file = f"{self.name_simulationFile}_duarouter__routes.xml"
        sumo_cmd = f"duarouter --route-files={self.folder_simulationName}\\{citiesdemand_file} --net-file={self.folder_simulationName}\\{self.network_file} --output-file={self.folder_simulationName}\\{self.routes_file} --ignore-errors"        
        if verbose:
            print("\nduarouter_file generation\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)
        return self.routes_file

    def continuous_rerouting_generation(self, begin_time=0, end_time=3600, verbose=False):
        #https://sumo.dlr.de/docs/Tools/Misc.html#generatecontinuousrerouterspy
        self.continuos_reroutes_file = f"{self.name_simulationFile}_continuous_rerouting_generation__beg{begin_time}_end{end_time}.xml"
        sumo_cmd = f'python "{self.sumo_tool_folder}/generateContinuousRerouters.py" --net-file {self.folder_simulationName}\\{self.network_file} --output-file {self.folder_simulationName}\\{self.continuos_reroutes_file} --begin {begin_time} --end {end_time}'
        if verbose:
            print("\ncontinuous rerouting generation\t>>\t",sumo_cmd)
        os.system(sumo_cmd)
        return self.continuos_reroutes_file      

class SUMO_routes_Exception__SUMO_networkInstance(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,instance):
        self.instance = instance
          
    def __str__(self):
        return f"SUMO_routes module require an instance 'SUMO_network' but receive an '{str(type(self.instance_type))}' object."

class SUMO_routes_Exception__FileNotInit(Exception):
    def __init__(self,key):
        self.key = key

    def __str__(self):
        return f"{key} not inizialized."