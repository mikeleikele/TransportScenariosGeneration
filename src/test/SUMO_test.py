from src.GeoSimulation.SUMO_simulation import *

def start_test():
    sumo_tool_folder = 'C:/Program Files (x86)/Eclipse/Sumo/tools'
    sumo_obj = SUMO_simulation(sumo_tool_folder= sumo_tool_folder, name_file="bassa",folder_name="data\sumo_simulation_files\\bassa", simulation_network_mode="osm",simulation_route_mode="random",osm_map_name="bassa")
    sumo_obj.generate_simulation(verbose=True)
    #sumo_obj.download_simulation()

def start_test_grid():
    #https://sumo.dlr.de/docs/netgenerate.html
    sumo_tool_folder = 'C:/Program Files (x86)/Eclipse/Sumo/tools'
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
    sumo_obj = SUMO_simulation(sumo_tool_folder= sumo_tool_folder, 
    folder_name="data\sumo_simulation_files\\maps_bassa", name_file="maps_bassa",
    network_settings=net_maps_settings, simulation_route_mode="random",osm_map_name="bassa")
    sumo_obj.generate_simulation(verbose=False)
    #sumo_obj.download_simulation()