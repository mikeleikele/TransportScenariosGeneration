from src.test import GeoGraph_test,SUMO_test, Orienteering_test, ML
import sys
from src.tool import utils_maps as mu

def geo_simul_test():
    GeoGraph_test.start_test()
    SUMO_test.start_test()

def simul_test(name_simulationFile):
    SUMO_test.start_test_grid(name_simulationFile=name_simulationFile)
def simul_vis_test(name_simulationFile):
    SUMO_test.start_test_grid(name_simulationFile=name_simulationFile)

def orient_test():
    Orienteering_test.start_test()

def ml_test():
    ML.start_test()
    
def statsNet(name_netFile):
    netreader = mu.StatsOSM(name_netFile)
    netreader.statsFromOSM()
    netreader.showStats()



if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == "--simulation" or args[0] == "--s":
        name_simulationFile = args[1]
        simul_test(name_simulationFile)
    if args[0] == "--simulation" or args[0] == "--sv":
        name_simulationFile = args[1]
        simul_vis_test(name_simulationFile)

    elif args[0] ==  "--geosimulation" or args[0] == "--gs":
        geo_simul_test()
    elif args[0] ==  "--orienteering" or args[0] == "--o":
        orient_test()
    elif args[0] ==  "--prediction" or args[0] == "--p":
        ml_test()
    elif args[0] ==  "--statsNetwork" or args[0] == "--stats":
        name_netFile = args[1]
        #data\maps\GEO__bassa.osm
        statsNet(name_netFile)
    else:
        print(0)