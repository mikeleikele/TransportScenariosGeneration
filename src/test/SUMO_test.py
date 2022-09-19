from src.GeoSimulation.SUMO_computation import *
from src.GeoSimulation.SUMO_visualization import *


def start_test():
    sumo_tool_folder = 'C:/Program Files (x86)/Eclipse/Sumo/tools'
    sumo_obj = SUMO_simulation(sumo_tool_folder= sumo_tool_folder, name_file="bassa",folder_name="data\sumo_simulation_files\\bassa", simulation_network_mode="osm",simulation_route_mode="random",osm_map_name="bassa")
    sumo_obj.generate_simulation(verbose=True)
    #sumo_obj.download_simulation()

def start_test_grid():
    #https://sumo.dlr.de/docs/netgenerate.html
    sumo_tool_folder = 'C:/Program Files (x86)/Eclipse/Sumo/tools'
    name_simulationFile = "Bassa_MEASURE"
    folder_simulationName = "data\sumo_simulation_files\\Bassa_MEASURE"

    net_grid_settings={
        "network_type":"generate",
        "generate":{
            "geometry":"grid", "grid":{"number":5,"length":200}
        }
    }
    net_spider_settings={
        "network_type":"generate",
        "generate":{
            "geometry":"spider", "spider":{"arm-number":5,"circle-number":4,"space-radius":100,"omit-center":False}
        }
    }
    net_rand_settings={
        "network_type":"generate",
        "generate":{
            "geometry":"rand", "rand":{"iterations":2000,"bidi-probability":1,"connectivity":0.95}
        }
    }
    net_maps_settings={
        "network_type":"maps",
        "maps":{
            "osm_maps_name":"bassa", "osm_maps_folder":None, "remove_geometry":True,
            "geometry_settings":['all']
        }
    }

    network_settings=net_maps_settings

    edgeStats_settings={
        "stats":[
            {"id":"all","type":"emissions"},
            {"id":"all","type":"harmonoise"},
            {"id":"all"},
        ]
    }

    sumo_obj = SUMO_computation(sumo_tool_folder= sumo_tool_folder, folder_simulationName=folder_simulationName, name_simulationFile=name_simulationFile, 
    network_settings=network_settings, simulation_route_mode="random", edgeStats_settings=edgeStats_settings)
    sumo_obj.generate_simulation(verbose=True)
    simulObj = sumo_obj.esecute_simulation(stop=True, fcd=True, verbose=True)

    sumo_viz = SUMO_visualization(sumo_tool_folder= sumo_tool_folder, folder_simulationName=folder_simulationName, name_simulationFile=name_simulationFile)
    

    #sumo_viz.plotTrajectories(filename_output="plot_time-speed.png", simulObj=simulObj)
    
    filein = f"{name_simulationFile}.all.out.edgeData.xml"
    networkFile = f"{name_simulationFile}_network__osm_bassa.net.xml"

    sumo_viz.plotNet(filename_output="plot_netSspeed.png", networkFile=networkFile, fileinput=filein, key_colors="speed", key_widths="density", color_map="viridis")
    