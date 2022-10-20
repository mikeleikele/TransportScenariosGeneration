from src.GeoSimulation.GeoGraph import *
import matplotlib.pyplot as plt

def start_test(POI_maps=False, draw_maps=False):
    """places = ['Cervignano del Friuli,Italy', "Terzo d'Aquileia,Italy",'Aquileia,Italy','Ruda,Italy',"Grado,italy",'Aiello del Friuli,Italy',"San giorgio di Nogaro,Italy",
        "campolongo tapogliano, Italy", "Gonars,Italy", "Visco,Italy", "San Vito al Torre,Italy", "Bagnaria arsa,Italy","Fiumicello Villa Vicentina,Italy", "Torviscosa,Italy",
        "Palmanova,Italy", ]
    maps_name="bassa"""

    places = ['Milano,Italy']
    maps_name="Milan"

    geo_maps_settings={
        "osm_maps_name":"Milan",
        "file_folder":None,
        "map_folder":None,
        "options":{
            "places":['Milano,Italy'], 
            "simplification":True,
            "poi_geometry":False, 
            "poi_option":{
                "filter":[
                    "nature","landuse","artificial_1","artificial_2",
                    "public_1","public_2","highway","military"]
            }
        }
    }
    bassa_friuli = GeoGraph(geo_maps_settings=geo_maps_settings)
    #https://wiki.openstreetmap.org/wiki/Map_features   
    fig, ax = plt.subplots(figsize=(25,18))
    if draw_maps:
        if POI_maps:
            POI_bassa_friuli = bassa_friuli.getPOI()
        GEO_bassa_friuli = bassa_friuli.getGEO()    
        if POI_maps:
            POI_bassa_friuli.plot(ax=ax, facecolor='khaki', alpha=0.7)
        ox.plot_graph(GEO_bassa_friuli, ax=ax, node_size=0, edge_linewidth=0.5,show=True)
