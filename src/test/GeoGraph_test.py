from src.GeoSimulation.GeoGraph import *
import matplotlib.pyplot as plt

def start_test():
    places = ['Cervignano del Friuli,Italy', "Terzo d'Aquileia,Italy",'Aquileia,Italy','Ruda,Italy',"Grado,italy",'Aiello del Friuli,Italy',"San giorgio di Nogaro,Italy",
        "campolongo tapogliano, Italy", "Gonars,Italy", "Visco,Italy", "San Vito al Torre,Italy", "Bagnaria arsa,Italy","Fiumicello Villa Vicentina,Italy", "Torviscosa,Italy",
        "Palmanova,Italy", ]
    bassa_friuli = GeoGraph(places, maps_name="bassa", save=True,file_folder= 'data/maps/', file_name="bassa")
    #https://wiki.openstreetmap.org/wiki/Map_features   
    fig, ax = plt.subplots(figsize=(25,18))
    POI_bassa_friuli = bassa_friuli.getPOI()
    GEO_bassa_friuli = bassa_friuli.getGEO()
    POI_bassa_friuli.plot(ax=ax, facecolor='khaki', alpha=0.7)
    ox.plot_graph(GEO_bassa_friuli, ax=ax, node_size=0, edge_linewidth=0.5,show=True)
