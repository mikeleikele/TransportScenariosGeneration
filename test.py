from src.test import GeoGraph_test,SUMO_test, Orienteering_test, ML
from src.GeoSimulation.SUMO_roadstats import SUMO_roadstats
from src.GeoSimulation.SUMO_mapsstats import SUMO_mapsstats
from src.SamplesGeneration.FlowSampling  import FlowSampling
from src.SamplesGeneration.FlowVisualization  import FlowVisualization

from src.NeuroCorrelation.InstancesGeneration import InstancesGeneration

from pathlib import Path
import sys
import argparse
from termcolor import cprint
from colorama import init, Style

def str2bool(v):
    if v in ['yes', 'true', 't', 'y', '1']:
        return True
    else:
        return False
    
def geographic_cities(par,args):
    if par is None:
        #UMBRIA
        municipalities = {'A': ['Acquasparta, Italy', 'Allerona, Italy', 'Alviano, Italy', 'Amelia, Italy', 'Arrone, Italy', 'Assisi, Italy', 'Attigliano, Italy', 'Avigliano Umbro, Italy'], 'B': ['Baschi, Italy', 'Bastia Umbra, Italy', 'Bettona, Italy', 'Bevagna, Italy'], 'C': ["Calvi dell'Umbria, Italy", 'Campello sul Clitunno, Italy', 'Cannara, Italy', 'Cascia, Italy', 'Castel Giorgio, Italy', 'Castel Ritaldi, Italy', 'Castel Viscardo, Italy', 'Castiglione del Lago, Italy', 'Cerreto di Spoleto, Italy', 'Citerna, Italy', 'Città della Pieve, Italy', 'Città di Castello, Italy', 'Collazzone, Italy', 'Corciano, Italy', 'Costacciaro, Italy'], 'D': ['Deruta, Italy'], 'F': ['Fabro, Italy', 'Ferentillo, Italy', 'Ficulle, Italy', 'Foligno, Italy', 'Fossato di Vico, Italy', 'Fratta Todina, Italy'], 'G': ["Giano dell'Umbria, Italy", 'Giove, Italy', 'Gualdo Cattaneo, Italy', 'Gualdo Tadino, Italy', 'Guardea, Italy', 'Gubbio, Italy'], 'L': ['Lisciano Niccone, Italy', 'Lugnano in Teverina, Italy'], 'M': ['Magione, Italy', 'Marsciano, Italy', 'Massa Martana, Italy', 'Monte Castello di Vibio, Italy', 'Monte Santa Maria Tiberina, Italy', 'Montecastrilli, Italy', 'Montecchio, Italy', 'Montefalco, Italy', 'Montefranco, Italy', 'Montegabbione, Italy', 'Monteleone di Spoleto, Italy', "Monteleone d'Orvieto, Italy", 'Montone, Italy'], 'N': ['Narni, Italy', 'Nocera Umbra, Italy', 'Norcia, Italy'], 'O': ['Orvieto, Italy', 'Otricoli, Italy'], 'P': ['Paciano, Italy', 'Panicale, Italy', 'Parrano, Italy', 'Passignano sul Trasimeno, Italy', 'Penna in Teverina, Italy', 'Piegaro, Italy', 'Pietralunga, Italy', 'Poggiodomo, Italy', 'Polino, Italy', 'Porano, Italy', 'Preci, Italy'], 'S': ['San Gemini, Italy', 'San Giustino, Italy', 'San Venanzo, Italy', "Sant'Anatolia di Narco, Italy", 'Scheggia e Pascelupo, Italy', 'Scheggino, Italy', 'Sellano, Italy', 'Sigillo, Italy', 'Spello, Italy', 'Spoleto, Italy', 'Stroncone, Italy'], 'T': ['Terni, Italy', 'Todi, Italy', 'Torgiano, Italy', 'Trevi, Italy', 'Tuoro sul Trasimeno, Italy'], 'U': ['Umbertide, Italy'], 'V': ['Valfabbrica, Italy', 'Vallo di Nera, Italy', 'Valtopina, Italy']}
        
        #ABRUZZO
        municipalities = {'A': ['Abbateggio, Italy', 'Acciano, Italy', 'Aielli, Italy', 'Alanno, Italy', 'Alba Adriatica, Italy', 'Alfedena, Italy', 'Altino, Italy', 'Ancarano, Italy', 'Anversa degli Abruzzi, Italy', 'Archi, Italy', 'Ari, Italy', 'Arielli, Italy', 'Arsita, Italy', 'Ateleta, Italy', 'Atessa, Italy', 'Atri, Italy', 'Avezzano, Italy'], 'B': ['Balsorano, Italy', 'Barete, Italy', 'Barisciano, Italy', 'Barrea, Italy', 'Basciano, Italy', 'Bellante, Italy', 'Bisegna, Italy', 'Bisenti, Italy', 'Bolognano, Italy', 'Bomba, Italy', 'Borrello, Italy', 'Brittoli, Italy', 'Bucchianico, Italy', 'Bugnara, Italy', 'Bussi sul Tirino, Italy'], 'C': ['Cagnano Amiterno, Italy', 'Calascio, Italy', 'Campli, Italy', 'Campo di Giove, Italy', 'Campotosto, Italy', 'Canistro, Italy', 'Canosa Sannita, Italy', 'Cansano, Italy', 'Capestrano, Italy', 'Capistrello, Italy', 'Capitignano, Italy', 'Caporciano, Italy', 'Cappadocia, Italy', 'Cappelle sul Tavo, Italy', 'Caramanico Terme, Italy', 'Carapelle Calvisio, Italy', 'Carpineto della Nora, Italy', 'Carpineto Sinello, Italy', 'Carsoli, Italy', 'Carunchio, Italy', 'Casacanditella, Italy', 'Casalanguida, Italy', 'Casalbordino, Italy', 'Casalincontrada, Italy', 'Casoli, Italy', 'Castel Castagna, Italy', 'Castel del Monte, Italy', 'Castel di Ieri, Italy', 'Castel di Sangro, Italy', 'Castel Frentano], Italy', 'Castelguidone, Italy', 'Castellafiume, Italy', 'Castellalto, Italy', 'Castelli , Italy', 'Castelvecchio Calvisio, Italy', 'Castelvecchio Subequo, Italy', 'Castiglione a Casauria, Italy', 'Castiglione Messer Marino, Italy', 'Castiglione Messer Raimondo, Italy', 'Castilenti, Italy', 'Catignano, Italy', 'Celano, Italy', 'Celenza sul Trigno, Italy', 'Cellino Attanasio, Italy', 'Cepagatti, Italy', 'Cerchio, Italy', 'Cermignano, Italy', 'Chieti, Italy', "Città Sant'Angelo, Italy", "Civita d'Antino, Italy", 'Civitaluparella, Italy', 'Civitaquana, Italy', 'Civitella Alfedena, Italy', 'Civitella Casanova, Italy', 'Civitella del Tronto, Italy', 'Civitella Messer Raimondo, Italy', 'Civitella Roveto, Italy', 'Cocullo, Italy', 'Collarmele, Italy', 'Collecorvino, Italy', 'Colledara, Italy', 'Colledimacine, Italy', 'Colledimezzo, Italy', 'Collelongo, Italy', 'Collepietro, Italy', 'Colonnella, Italy', 'Controguerra, Italy', 'Corfinio, Italy', 'Corropoli, Italy', 'Cortino, Italy', 'Corvara, Italy', 'Crecchio, Italy', 'Crognaleto, Italy', 'Cugnoli, Italy', 'Cupello, Italy'], 'f': ['f, Italy'], 'D': ['Dogliola, Italy'], 'E': ['Elice, Italy'], 'F': ['Fagnano Alto, Italy', 'Fallo , Italy', 'Fano Adriano, Italy', 'Fara Filiorum Petri, Italy', 'Fara San Martino, Italy', 'Farindola, Italy', 'Filetto, Italy', 'Fontecchio, Italy', 'Fossa , Italy', 'Fossacesia, Italy', 'Fraine, Italy', 'Francavilla al Mare, Italy', 'Fresagrandinaria, Italy', 'Frisa, Italy', 'Furci, Italy'], 'G': ['Gagliano Aterno, Italy', 'Gamberale, Italy', 'Gessopalena, Italy', 'Gioia dei Marsi, Italy', 'Gissi, Italy', 'Giuliano Teatino, Italy', 'Giulianova, Italy', 'Goriano Sicoli, Italy', 'Guardiagrele, Italy', 'Guilmi, Italy'], 'I': ['Introdacqua, Italy', "Isola del Gran Sasso d'Italia, Italy"], 'L': ['Lama dei Peligni, Italy', 'Lanciano, Italy', "L'Aquila, Italy", 'Lecce nei Marsi, Italy', 'Lentella, Italy', 'Lettomanoppello, Italy', 'Lettopalena, Italy', 'Liscia, Italy', 'Loreto Aprutino, Italy', 'Luco dei Marsi, Italy', 'Lucoli, Italy'], 'M': ["Magliano de' Marsi, Italy", 'Manoppello, Italy', 'Martinsicuro, Italy', "Massa d'Albe, Italy", 'Miglianico, Italy', 'Molina Aterno, Italy', 'Montazzoli, Italy', 'Montebello di Bertona, Italy', 'Montebello sul Sangro, Italy', 'Monteferrante, Italy', 'Montefino, Italy', 'Montelapiano, Italy', 'Montenerodomo, Italy', 'Monteodorisio, Italy', 'Montereale, Italy', 'Montesilvano, Italy', 'Montorio al Vomano, Italy', 'Morino, Italy', "Morro d'Oro, Italy", "Mosciano Sant'Angelo, Italy", 'Moscufo, Italy', 'Mozzagrogna, Italy'], 'N': ['Navelli, Italy', 'Nereto, Italy', 'Nocciano, Italy', 'Notaresco, Italy'], 'O': ['Ocre, Italy', 'Ofena, Italy', 'Opi , Italy', 'Oricola, Italy', 'Orsogna, Italy', 'Ortona dei Marsi, Italy', 'Ortona, Italy', 'Ortucchio, Italy', 'Ovindoli, Italy'], 'P': ['Pacentro, Italy', 'Paglieta, Italy', 'Palena, Italy', 'Palmoli, Italy', 'Palombaro , Italy', "Penna Sant'Andrea, Italy", 'Pennadomo, Italy', 'Pennapiedimonte, Italy', 'Penne, Italy', 'Perano, Italy', 'Pereto, Italy', 'Pescara, Italy', 'Pescasseroli, Italy', 'Pescina, Italy', 'Pescocostanzo, Italy', 'Pescosansonesco, Italy', 'Pettorano sul Gizio, Italy', 'Pianella, Italy', 'Picciano, Italy', 'Pietracamela, Italy', 'Pietraferrazzana, Italy', 'Pietranico, Italy', 'Pineto, Italy', 'Pizzoferrato, Italy', 'Pizzoli, Italy', 'Poggio Picenze, Italy', 'Poggiofiorito, Italy', 'Pollutri, Italy', 'Popoli, Italy', "Prata d'Ansidonia, Italy", 'Pratola Peligna, Italy', 'Pretoro, Italy', 'Prezza, Italy'], 'Q': ['Quadri, Italy'], 'R': ['Raiano, Italy', 'Rapino, Italy', 'Ripa Teatina, Italy', 'Rivisondoli, Italy', 'Rocca di Botte, Italy', 'Rocca di Cambio, Italy', 'Rocca di Mezzo, Italy', 'Rocca Pia, Italy', 'Rocca San Giovanni, Italy', 'Rocca Santa Maria, Italy', 'Roccacasale, Italy', 'Roccamontepiano, Italy', 'Roccamorice, Italy', 'Roccaraso, Italy', 'Roccascalegna, Italy', 'Roccaspinalveti, Italy', 'Roio del Sangro, Italy', 'Rosciano, Italy', 'Rosello, Italy', 'Roseto degli Abruzzi, Italy'], 'S': ['Salle, Italy', 'San Benedetto dei Marsi, Italy', 'San Benedetto in Perillis, Italy', 'San Buono, Italy', "San Demetrio ne' Vestini, Italy", 'San Giovanni Lipioni, Italy', 'San Giovanni Teatino, Italy', 'San Martino sulla Marrucina, Italy', 'San Pio delle Camere, Italy', 'San Salvo, Italy', 'San Valentino in Abruzzo Citeriore, Italy', 'San Vincenzo Valle Roveto, Italy', 'San Vito Chietino, Italy', 'Santa Maria Imbaro, Italy', 'Sante Marie, Italy', "Sant'Egidio alla Vibrata, Italy", "Sant'Eufemia a Maiella, Italy", "Sant'Eusanio del Sangro, Italy", "Sant'Eusanio Forconese, Italy", 'Santo Stefano di Sessanio, Italy', "Sant'Omero, Italy", 'Scafa, Italy', 'Scanno, Italy', 'Scerni, Italy', 'Schiavi di Abruzzo, Italy', 'Scontrone, Italy', 'Scoppito, Italy', 'Scurcola Marsicana, Italy', 'Secinaro, Italy', 'Serramonacesca, Italy', 'Silvi, Italy', 'Spoltore, Italy', 'Sulmona, Italy'], 'T': ['Tagliacozzo, Italy', 'Taranta Peligna, Italy', 'Teramo, Italy', 'Tione degli Abruzzi, Italy', 'Tocco da Casauria, Italy', 'Torni, Italy', 'Torano Nuovo, Italy', 'Torino di Sangro, Italy', 'Tornareccio, Italy', 'Tornimparte, Italy', "Torre de' Passeri, Italy", 'Torrebruna, Italy', 'Torrevecchia Teatina, Italy', 'Torricella Peligna, Italy', 'Torricella Sicura, Italy', 'Tortoreto, Italy', 'Tossicia, Italy', 'Trasacco, Italy', 'Treglio, Italy', 'Tufillo, Italy', 'Turrivalignani, Italy'], 'V': ['Vacri, Italy', 'Valle Castellana, Italy', 'Vasto, Italy', 'Vicoli, Italy', 'Villa Celiera, Italy', 'Villa Santa Lucia degli Abruzzi, Italy', 'Villa Santa Maria, Italy', "Villa Sant'Angelo, Italy", 'Villalago, Italy', 'Villalfonsina, Italy', 'Villamagna, Italy', 'Villavallelonga, Italy', 'Villetta Barrea, Italy', 'Vittorito, Italy']}
        
        for key in municipalities:
            print(key)
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
    args = sys.argv
    parser = argparse.ArgumentParser(description="Description of your program")
    parser.add_argument('--exp', type=str, required=True, help='Experiment')
    parser.add_argument('--map_name', type=str, required=False, help='Map name')
    parser.add_argument('--simulation_name', type=str, required=False, help='Simulation name')
    parser.add_argument('--number_samples', type=int, required=False, help='Number of samples')
    parser.add_argument('--par', type=str, required=False, help='par')
    
    
    parser.add_argument('--num_case', type=str, required=True, help='Description for num_case')
    parser.add_argument('--experiment_name_suffix', type=str, required=True, help='Description for experiment_name_suffix')
    parser.add_argument('--main_folder', type=str, required=True, help='Description for main_folder')
    parser.add_argument("--repeation_b", type=str, required=True, help='Description for repeation - begin')  
    parser.add_argument("--repeation_e", type=str, required=True, help='Description for repeation - end')  
    parser.add_argument('--optimization', type=str2bool, required=True, help='Enable or disable optimization')
    parser.add_argument('--load_model', type=str2bool, required=True, help='Description for load_model')
    parser.add_argument('--train_models', type=str2bool, required=True, help='Description for train_models')
    parser.add_argument('--time_slot', type=str, required=False, default=None, help='Description time slot')
        
    parsed_args = parser.parse_args(sys.argv[1:])
    
    print(f"\n")
    cprint(Style.BRIGHT + "|------------------------|" + Style.RESET_ALL, 'cyan', attrs=["bold"])
    cprint(Style.BRIGHT + "|        Traffic         |" + Style.RESET_ALL, 'cyan', attrs=["bold"])
    cprint(Style.BRIGHT + "|  Scenarios Generation  |" + Style.RESET_ALL, 'cyan', attrs=["bold"])
    cprint(Style.BRIGHT + "|------------------------|" + Style.RESET_ALL, 'cyan', attrs=["bold"])
    cprint(Style.BRIGHT +f"| Process: {parsed_args.exp}" + Style.RESET_ALL, 'cyan')
    cprint(Style.BRIGHT + "|------------------------|" + Style.RESET_ALL, 'cyan')
    print(f"\n")
    
    if parsed_args.exp ==  "orienteering" or parsed_args.exp == "o":
        orient_test()
    
    #geographic maps download
    elif parsed_args.exp == "geo" or parsed_args.exp == "g":      
        if len(args)>1:
            par = parsed_args.par
            
        else:
            par=None
        geographic_cities(par,args)
    elif parsed_args.exp == "geopoint" or parsed_args.exp == "gp":        
        geo_test_point()

    elif parsed_args.exp == "simulation" or parsed_args.exp == "s":        
        name_simulationFile = parsed_args.simulation_name
        simul_test(name_simulationFile)
        print(3)
    
    elif parsed_args.exp ==  "geosimulation" or parsed_args.exp == "gs":
        name_simulationFile = parsed_args.simulation_name
        print(name_simulationFile)
        geo_simul_test(name_simulationFile)
        print(5)
    elif parsed_args.exp ==  "prediction" or parsed_args.exp == "p":
        ml_test()
        print(6)
    elif parsed_args.exp ==  "statsMaps" or parsed_args.exp == "sm":
        maps_name = parsed_args.map_name
        #data\maps\GEO__bassa.osm
        statsMaps(maps_name)
        print(7)
    elif parsed_args.exp ==  "statsRoad" or parsed_args.exp == "sr":
        #python test.py sr grid5
        simulation_name = parsed_args.simulation_name
        statsRoads(simulation_name)
        print(8)
    elif parsed_args.exp ==  "flowgen" or parsed_args.exp == "fg":
        #python test.py sr grid5
        simulation_name = parsed_args.simulation_name
        if len(args)==2:
            number_samples = 1            
        else:
            try:
                number_samples = parsed_args.number_sample
            except ValueError:
                print("number_samples require a number.")
                number_samples=1            
        flowgen(simulation_name,number_samples)
        print(9)
    elif parsed_args.exp ==  "flowView" or parsed_args.exp == "fv":
        simulation_name = parsed_args.simulation_name
        number_sample = parsed_args.number_sample
        flowview(simulation_name,number_sample)
        print(10)
    
    
    elif parsed_args.exp ==  "neuroD":        
    
        neuroExp = InstancesGeneration(sys.argv[1:])
        cprint(Style.BRIGHT + f"Folder :\t{parsed_args.main_folder}" + Style.RESET_ALL, 'red', attrs=["bold"])
    else:
        print(0," no opt recognized")
        
