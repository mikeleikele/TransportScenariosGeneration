from src.test import GeoGraph_test,SUMO_test, Orienteering_test, ML
from src.GeoSimulation.SUMO_roadstats import SUMO_roadstats
from src.GeoSimulation.SUMO_mapsstats import SUMO_mapsstats
import sys

def geo_test():
    GeoGraph_test.start_test()

def geo_simul_test(name_simulationFile):
    GeoGraph_test.start_test()
    SUMO_test.start_test_grid(name_simulationFile=name_simulationFile)

def simul_test(name_simulationFile):
    SUMO_test.start_test_grid(name_simulationFile=name_simulationFile)
def simul_vis_test(name_simulationFile):
    SUMO_test.start_test_grid(name_simulationFile=name_simulationFile)

def orient_test():
    Orienteering_test.start_test()

def ml_test():
    ML.start_test()
    
def statsMaps(maps_name):
    mapsstats = SUMO_mapsstats(maps_name)
    mapsstats.compute_mapsstats()

def statsRoads(simulation_name):
    roadstats = SUMO_roadstats(simulation_name, is_osm=False)
    roadstats.compute_roadstats()
    
if __name__ == "__main__":
    args = sys.argv[1:]
    
    if args[0] ==  "--orienteering" or args[0] == "--o":
        orient_test()
        print(1)
    elif args[0] == "--geo" or args[0] == "--g":        
        geo_test()    
        print(2)
    
    elif args[0] == "--simulation" or args[0] == "--s":        
        name_simulationFile = args[1]
        simul_test(name_simulationFile)
        print(3)
    elif args[0] == "--visualsimulation" or args[0] == "--sv":
        name_simulationFile = args[1]
        simul_vis_test(name_simulationFile)
        print(4)
    elif args[0] ==  "--geosimulation" or args[0] == "--gs":
        name_simulationFile = args[1]
        print(name_simulationFile)
        geo_simul_test(name_simulationFile)
        print(5)
    elif args[0] ==  "--prediction" or args[0] == "--p":
        ml_test()
        print(6)
    elif args[0] ==  "--statsMaps" or args[0] == "--sm":
        maps_name = args[1]
        #data\maps\GEO__bassa.osm
        statsMaps(maps_name)
        print(7)
    elif args[0] ==  "--statsRoad" or args[0] == "--sr":
        #python test.py --sr grid5
        simulation_name = args[1]
        statsRoads(simulation_name)
        print(8)
    else:
        print(0)