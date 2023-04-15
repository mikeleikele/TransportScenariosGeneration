from src.test import GeoGraph_test,SUMO_test, Orienteering_test, ML
from src.GeoSimulation.SUMO_roadstats import SUMO_roadstats
from src.GeoSimulation.SUMO_mapsstats import SUMO_mapsstats
import sys
from src.SamplesGeneration.FlowSampling  import FlowSampling
from src.SamplesGeneration.FlowVisualization  import FlowVisualization
from pathlib import Path

def geographic_cities(par,args):
    if par is None:
        municipalities = {'A': ['Acquasparta, Italy', 'Allerona, Italy', 'Alviano, Italy', 'Amelia, Italy', 'Arrone, Italy', 'Assisi, Italy', 'Attigliano, Italy', 'Avigliano Umbro, Italy'], 'B': ['Baschi, Italy', 'Bastia Umbra, Italy', 'Bettona, Italy', 'Bevagna, Italy'], 'C': ["Calvi dell'Umbria, Italy", 'Campello sul Clitunno, Italy', 'Cannara, Italy', 'Cascia, Italy', 'Castel Giorgio, Italy', 'Castel Ritaldi, Italy', 'Castel Viscardo, Italy', 'Castiglione del Lago, Italy', 'Cerreto di Spoleto, Italy', 'Citerna, Italy', 'Città della Pieve, Italy', 'Città di Castello, Italy', 'Collazzone, Italy', 'Corciano, Italy', 'Costacciaro, Italy'], 'D': ['Deruta, Italy'], 'F': ['Fabro, Italy', 'Ferentillo, Italy', 'Ficulle, Italy', 'Foligno, Italy', 'Fossato di Vico, Italy', 'Fratta Todina, Italy'], 'G': ["Giano dell'Umbria, Italy", 'Giove, Italy', 'Gualdo Cattaneo, Italy', 'Gualdo Tadino, Italy', 'Guardea, Italy', 'Gubbio, Italy'], 'L': ['Lisciano Niccone, Italy', 'Lugnano in Teverina, Italy'], 'M': ['Magione, Italy', 'Marsciano, Italy', 'Massa Martana, Italy', 'Monte Castello di Vibio, Italy', 'Monte Santa Maria Tiberina, Italy', 'Montecastrilli, Italy', 'Montecchio, Italy', 'Montefalco, Italy', 'Montefranco, Italy', 'Montegabbione, Italy', 'Monteleone di Spoleto, Italy', "Monteleone d'Orvieto, Italy", 'Montone, Italy'], 'N': ['Narni, Italy', 'Nocera Umbra, Italy', 'Norcia, Italy'], 'O': ['Orvieto, Italy', 'Otricoli, Italy'], 'P': ['Paciano, Italy', 'Panicale, Italy', 'Parrano, Italy', 'Passignano sul Trasimeno, Italy', 'Penna in Teverina, Italy', 'Piegaro, Italy', 'Pietralunga, Italy', 'Poggiodomo, Italy', 'Polino, Italy', 'Porano, Italy', 'Preci, Italy'], 'S': ['San Gemini, Italy', 'San Giustino, Italy', 'San Venanzo, Italy', "Sant'Anatolia di Narco, Italy", 'Scheggia e Pascelupo, Italy', 'Scheggino, Italy', 'Sellano, Italy', 'Sigillo, Italy', 'Spello, Italy', 'Spoleto, Italy', 'Stroncone, Italy'], 'T': ['Terni, Italy', 'Todi, Italy', 'Torgiano, Italy', 'Trevi, Italy', 'Tuoro sul Trasimeno, Italy'], 'U': ['Umbertide, Italy'], 'V': ['Valfabbrica, Italy', 'Vallo di Nera, Italy', 'Valtopina, Italy']}
        for key in municipalities:
            cities_list = municipalities[key]
            GeoGraph_test.downloadmaps_cities(case="perugia",list_municipalities=cities_list, map_name_file=f"{key}")    
            
    elif par=="bejing":
        if len(args)>=4:
            min_range = int(args[2])
            max_range = int(args[3])            
        else:
            min_range = 1
            max_range = 10
        print("Bejing range taxi:\t", min_range, max_range)
        users_list_range = range(min_range, max_range)
        GeoGraph_test.BejingDataset(users_list=users_list_range, line_break=None)
    else:
        GeoGraph_test.downloadmaps_cities(case=par)

def geographic_point():
    GeoGraph_test.downloadmaps_points(draw_maps=True)

def geo_simul_test(name_simulationFile):
    GeoGraph_test.start_test()
    SUMO_test.start_SUMO_simulation(name_simulationFile=name_simulationFile)

def simul_test(name_simulationFile):
    SUMO_test.start_SUMO_pointsimulation(name_simulationFile=name_simulationFile)

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
    sampled_dir = Path("data","sumo_simulation_files",self.simulation_name,"randomgraph")
    flowviewer = FlowVisualization(simulation_name, sampled_dir, number_sample, load_data=True)
    #flowviewer.draw_sampledgraph("travel_time")
    #flowviewer.draw_sampledgraph("weighted_mean")
    flowviewer.draw_sampledgraph("vehicles_id")

if __name__ == "__main__":
    args = sys.argv[1:]
    print(f" ")
    print(f"      Welcome - OSG      ")
    print(f"|------------------------")
    print(f"| Process: {args[0]}")
    if len(args)>1:
        print(f"| Maps   : {args[1]}")
    print(f"|------------------------")
    print(f" ")
    
    if args[0] ==  "--orienteering" or args[0] == "--o":
        orient_test()
    
    #geographic maps download
    elif args[0] == "--geo" or args[0] == "--g":      
        if len(args)>1:
            par = args[1]
            
        else:
            par=None
        geographic_cities(par,args)
    elif args[0] == "--geopoint" or args[0] == "--gp":        
        geo_test_point()

    
    elif args[0] == "--simulation" or args[0] == "--s":        
        name_simulationFile = args[1]
        simul_test(name_simulationFile)
        print(3)
    
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