import os 
import xml.etree.cElementTree as ET
from .GeoGraph import *
from .SUMO_network import *
from .SUMO_routes import *

class SUMO_visualization():

    def __init__(self, sumo_tool_folder, folder_simulationName, name_simulationFile, verbose=False):
        self.folder_simulationName = folder_simulationName
        self.name_simulationFile = name_simulationFile
        self.verbose = verbose
        self.sumo_tool_folder = sumo_tool_folder