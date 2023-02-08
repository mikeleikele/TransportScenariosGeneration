from src.GeoSimulation.GeoGraph import *
import matplotlib.pyplot as plt

def start_test(POI_maps=False, draw_maps=False):
    """places = ['Cervignano del Friuli,Italy', "Terzo d'Aquileia,Italy",'Aquileia,Italy','Ruda,Italy',"Grado,italy",'Aiello del Friuli,Italy',"San giorgio di Nogaro,Italy",
        "campolongo tapogliano, Italy", "Gonars,Italy", "Visco,Italy", "San Vito al Torre,Italy", "Bagnaria arsa,Italy","Fiumicello Villa Vicentina,Italy", "Torviscosa,Italy",
        "Palmanova,Italy", ]
    maps_name='bassa'
    """
    places = ['Cervignano del Friuli,Italy']
    maps_name='cervi'
    """

    places = ['Milano,Italy']
    maps_name='Milan'
    """
    geo_maps_settings={
        "osm_maps_name":maps_name,
        "map_folder":None,
        "poi_folder":None,
        "options":{
            "places":places, 
            "simplification":True,
            "poi_geometry":True, 
            "poi_option":{
                "filter":[
                    "nature","landuse","artificial_1","artificial_2",
                    "public_1","public_2","highway","military"]
            }
        }
    }
    geo_settings = GeoGraph(geo_maps_settings=geo_maps_settings)
    #https://wiki.openstreetmap.org/wiki/Map_features   
    fig, ax = plt.subplots(figsize=(25,18))
    if draw_maps:
        if POI_maps:
            POI_geo = geo_settings.getPOI()
        GEO_geo = bassa_friuli.getGEO()    
        if POI_maps:
            POI_geo.plot(ax=ax, facecolor='khaki', alpha=0.7)
        ox.plot_graph(GEO_geo, ax=ax, node_size=0, edge_linewidth=0.5,show=True)
