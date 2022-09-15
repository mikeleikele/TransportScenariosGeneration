import os 
import xml.etree.cElementTree as ET
from .GeoGraph import *
from .SUMO_network import *
from .SUMO_routes import *

class SUMO_simulation():

    def __init__(self, sumo_tool_folder, folder_simulationName, name_simulationFile, networkObj, routesObj, verbose=False):
        self.folder_simulationName = folder_simulationName
        self.name_simulationFile = name_simulationFile
        self.verbose = verbose
        self.sumo_tool_folder = sumo_tool_folder

        if isinstance(networkObj, SUMO_network):
            self.network_file = networkObj.get_networkFile()
            self.geometry_file = networkObj.get_geometrykFile(raiseExc=False)
        else:
            raise SUMO_simulation_Exception__SUMO_Instance(networkObj,SUMO_network)
        
        if isinstance(routesObj, SUMO_routes):
            self.routes_file = routesObj.get_routesFile()
            self.continuos_reroutes_file = routesObj.get_continuous_reroutingFile()
        else:
            raise SUMO_simulation_Exception__SUMO_Instance(routesObj,SUMO_routes)

    def get_configFile(self):
        if self.sumoConfig_file is None:
            raise SUMO_simulation_Exception__FileNotInit("sumoConfig")
        return self.sumoConfig_file

    def get_traceFile(self):
        if self.sumoTrace_file is None:
            raise SUMO_simulation_Exception__FileNotInit("sumoTrace")
        return self.sumoTrace_file

    def get_dumpFile(self):
        if self.sumoDump_file is None:
            raise SUMO_simulation_Exception__FileNotInit("sumoDump")
        return self.sumoDump_file
    
    def get_emissionFile(self):
        if self.sumoEmission_file is None:
            raise SUMO_simulation_Exception__FileNotInit("sumoEmission")
        return self.sumoEmission_file
    
    def get_laneChangeFile(self):
        if self.sumoLaneChange_file is None:
            raise SUMO_simulation_Exception__FileNotInit("sumoLaneChange")
        return self.sumoLaneChange_file
    

    def esecute_simulation(self, trace=False, dump=False, emission=False, lanechange=False, vtk=False, verbose=False):
        self.write_sumoconfig(verbose=verbose)
        self.exe_sumo_simulation(trace=trace, dump=dump, verbose=verbose)
        
    def exe_sumo_simulation(self, trace, dump, verbose=False):
        sumo_cmd = f"sumo -c {self.folder_simulationName}\\{self.sumoConfig_file}"
        if trace:
            self.sumoTrace_file = f"{self.name_simulationFile}.out_trace.xml"
            sumo_cmd = sumo_cmd+ f" --fcd-output {self.folder_simulationName}\\{self.sumoTrace_file}"
        if dump:
            self.sumoDump_file = f"{self.name_simulationFile}.out_dump.xml"
            sumo_cmd = sumo_cmd+ f" --netstate-dump {self.folder_simulationName}\\{self.sumoDump_file}"
        if dump:
            self.sumoEmission_file = f"{self.name_simulationFile}.out_emission.xml"
            sumo_cmd = sumo_cmd+ f" --emission-output {self.folder_simulationName}\\{self.sumoEmission_file}"
        if lanechange:
            self.sumoLaneChange_file =  f"{self.name_simulationFile}.out_laneChange.xml"
            sumo_cmd = sumo_cmd+ f" --lanechange-output {self.folder_simulationName}\\{self.sumoLaneChange_file}"
        if vtk:
            self.sumoVTK_file =  f"{self.name_simulationFile}.out_vtk.xml"
            sumo_cmd = sumo_cmd+ f" --vtk-output {self.folder_simulationName}\\{self.sumoVTK_file}"
        if verbose:
            print("\nflows generation\t>>\t",sumo_cmd,"")
        os.system(sumo_cmd)
        

    def write_sumoconfig(self, trace=False, dump=False, emission=False, verbose=False):
        self.sumoConfig_file = f"{self.name_simulationFile}.sumocfg"
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
        if trace:
            ET.SubElement(k_output, "fcd-output", value=f"{self.name_simulationFile}.out_trace.xml")
        if dump:
            ET.SubElement(k_output, "netstate-dump", value=f"{self.name_simulationFile}.out_dump.xml")
        if emission:
            ET.SubElement(k_output, "emission", value=f"{self.name_simulationFile}.out_emission.xml")
        
        tree = ET.ElementTree(configuration)
        ET.indent(tree, space="\t", level=0)
        tree.write(f"{self.folder_simulationName}\\{self.sumoConfig_file}", encoding="utf-8")


class SUMO_simulation_Exception__SUMO_Instance(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,instance,type_class):
        self.instance = instance
        self.type_class = type_class
          
    def __str__(self):
        return f"SUMO_simulation module require an instance '{type_class}' but receive an '{str(type(self.instance_type))}' object."


class SUMO_simulation_Exception__FileNotInit(Exception):
    def __init__(self,key):
        self.key = key

    def __str__(self):
        return f"{key} not inizialized."