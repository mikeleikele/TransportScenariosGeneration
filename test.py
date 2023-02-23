from src.test import GeoGraph_test,SUMO_test, Orienteering_test, ML
from src.GeoSimulation.SUMO_roadstats import SUMO_roadstats
from src.GeoSimulation.SUMO_mapsstats import SUMO_mapsstats
import sys
from src.SamplesGeneration.FlowSampling  import FlowSampling
from src.SamplesGeneration.FlowVisualization  import FlowVisualization

def geo_test():
    GeoGraph_test.start_test()

def geo_simul_test(name_simulationFile):
    GeoGraph_test.start_test()
    SUMO_test.start_SUMO_simulation(name_simulationFile=name_simulationFile)

def simul_test(name_simulationFile):
    SUMO_test.start_SUMO_simulation(name_simulationFile=name_simulationFile)
def simul_vis_test(name_simulationFile):
    SUMO_test.start_SUMO_simulation(name_simulationFile=name_simulationFile)

def orient_test():
    Orienteering_test.start_test()

def ml_test():
    ML.start_test()
    
def statsMaps(maps_name):
    mapsstats = SUMO_mapsstats(maps_name)
    mapsstats.compute_mapsstats()

def statsRoads(simulation_name):
    roadstats = SUMO_roadstats(simulation_name, is_osm=True)
    roadstats.compute_roadstats()

def flowgen(simulation_name, number_samples):
    flows = FlowSampling(is_simulation=True, simulation_name=simulation_name)
    flows.generate_samples(number_samples, draw_graph=True, save_flows=True)

def flowview(simulation_name, number_sample):
    flowviewer = FlowVisualization(simulation_name, "randomgraph", number_sample, load_data=True)
    #flowviewer.draw_sampledgraph("travel_time")
    #flowviewer.draw_sampledgraph("weighted_mean")
    flowviewer.draw_sampledgraph("vehicles_id")

if __name__ == "__main__":
    args = sys.argv[1:]
    print(f" ")
    print(f"      Welcome - OSG      ")
    print(f"|------------------------")
    print(f"| Process: {args[0]}")
    print(f"| Maps   : {args[1]}")
    print(f"|------------------------")
    print(f" ")
    
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
    elif args[0] ==  "--flowgen" or args[0] == "--fg":
        #python test.py --sr grid5
        simulation_name = args[1]
        if len(args)==2:
            number_samples = 1            
        else:
            try:
                number_samples = int(args[2])
            except ValueError:
                print("number_samples require a number.")
                number_samples=1            
        flowgen(simulation_name,number_samples)
        print(9)
    elif args[0] ==  "--flowView" or args[0] == "--fv":
        simulation_name = args[1]
        number_sample = args[2]                   
        flowview(simulation_name,number_sample)
        print(10)
    else:
        print(0," no opt recognized")