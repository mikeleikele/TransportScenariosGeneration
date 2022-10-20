import osmnx as ox
import os 
import geopandas
import pandas as pd
from tqdm.auto import tqdm
import xml.etree.ElementTree as ET

class OSMgeometry:
    options = ["nature","landuse","artificial_1","artificial_2","public_1","public_2","highway","military","all"]

    
    def getOptions():
        option = ["nature","landuse","artificial_1","artificial_2","public_1","public_2","highway","military","all"]
        return option
    
    def isFilter(name_filter):
        option = OSMgeometry.getOptions()
        filters = OSMgeometry.getFilter()
        if name_filter in option:
            key =  f"POI__{name_filter}__"
            value = (key,filters[key])
        else:
            value = None
        return value

    def getFilter():
        poi_filter = {
            "POI__nature__":{'natural':True,'geological':True},#'place':True,
            "POI__landuse__":{'landuse':True},
            "POI__artificial_1__":{'aeroway':True,'aerialway':True,'cycleway':True,'bus_bay':True},
            "POI__artificial_2__":{'public_transport':True,'railway':True,'bridge':True,'tunnel':True,'waterway':True},
            "POI__public_1__":{'amenity':True,'building':True,'emergency':True,'lifeguard':True,'historic':True},
            "POI__public_2__":{'leisure':True,'office':True,'shop':True,'sport':True,'tourism':True},
            "POI__highway__":{'highway':True,'route':True},
            "POI__military__":{'military':True,},            
        }
        return poi_filter

class GeoGraph:
    def __init__(self, geo_maps_settings):
        self.maps_name = geo_maps_settings["osm_maps_name"] 

        if "file_folder" in geo_maps_settings:
            if geo_maps_settings["file_folder"] is None:
                self.file_folder = "data/maps"
            else:
                self.file_folder = geo_maps_settings["file_folder"]
        else:
            self.file_folder = "data/maps"
        if not os.path.exists(self.file_folder):
            os.makedirs(self.file_folder)
        
        if "map_folder" in geo_maps_settings:
            if geo_maps_settings["map_folder"] is None:
                self.file_folder = "data/osm_maps"
            else:
                self.file_folder = geo_maps_settings["map_folder"]
        else:
            self.file_folder = "data/osm_maps"
        if not os.path.exists(self.file_folder):
            os.makedirs(self.file_folder)



        self.places = geo_maps_settings["options"]["places"]
        simplification = geo_maps_settings["options"]["simplification"]
        self.geograph = self.request_graph(self.places, simplification)
        #deprecated
        """if file_name is None:
            self.maps_name = ""
            for i in range(len(self.places)):
                _plane_cod =  self.places[i].lower().replace(',', '_').replace(' ', '-')
                if i == 0:
                  self.maps_name = _plane_cod
                else:
                    self.maps_name += "--"+ _plane_cod
        else:
            self.maps_name = file_name"""
        
        if geo_maps_settings["options"]["poi_geometry"]:
            list_filter = geo_maps_settings["options"]["poi_geometry"]["poi_option"]["filter"]
            if list_filter == "all":
                self.filter_poi = OSMgeometry.getFilter()
            else:
                self.filter_poi = dict()
                for filter_name in list_filter:
                    filter_tuple = OSMgeometry.isFilter(filter_name)   
                    self.filter_poi[filter_tuple[0]] = filter_tuple[1]
            self.poigraph = self.geometries_from_place(self.places,save=True)


    def getGEO(self):
        return self.geograph

    def getPOI(self):
        return self.poigraph

    def request_graph(self,query,simplification=False):
        """
        Negotiates a query to OSMNX if no local stored file else loads local file
        :return: result:
        """
        if "GEO__"+self.maps_name+ '.graphml' in os.listdir(self.file_folder):
            geograph = ox.load_graphml(self.file_folder + '/' + "GEO__"+self.maps_name+ '.graphml' ) 
        else:
            geograph = self.graph_from_place(query,simplification)
        return geograph

    def graph_from_place(self, place, simplify=True,network_type='all', simplification=False, save=True):
        """
        Request a graph from OSMNX
        network_type : "all_private", "all", "bike", "drive", "drive_service", "walk"
        :return: G: OSMN Graph object
        """
        # query graph from place
        G = None
        try:
            p_bar = tqdm(range(10))
            p_bar.update(1)
            p_bar.refresh()
            G = ox.graph_from_place(place, simplify=simplify, network_type=network_type)
            p_bar.update(10)
            p_bar.refresh()
            if simplification:
                G = ox.simplification.simplify_graph(G, strict=True, remove_rings=True)
        
            if save:
                geo_filepath = f"{self.file_folder}/GEO__{self.maps_name}"
                ox.save_graphml(G, filepath=f"{geo_filepath}.graphml")
                ox.save_graph_xml(G, filepath=f"{geo_filepath}.osm")
        except Exception:
            raise GeoGraph_Exception__Param(place)
        return G


    def geometries_from_place(self, places, tags={'natural':True,'place':True},save=False):
        POI_list_all = []
        for key in self.filter_poi:
            tags = filter_poi[key]
            POI_list = []
            for i in tqdm(range(len(places))):
                _place = places[i]
                place = _place.lower().replace(',', '_').replace(' ', '-')
                if f"{self.map_folder}/{key}{place}.geojson" in os.listdir(self.map_folder):
                    _poifile = geopandas.read_file(f"{self.map_folder}/{key}{place}.geojson")
                    POI_list.append(_poifile)
                    POI_list_all.append(_poifile)
                else:
                    try:
                        ox.config(timeout=10000)
                        _poifile = ox.geometries_from_place(_place,tags=tags)
                        POI_list.append(_poifile)
                        POI_list_all.append(_poifile)
                        if save:
                            with open(f"{self.map_folder}/{key}{place}.geojson", "w") as f:
                                f.write(_poifile.to_json())
                    except Exception:
                        raise GeoGraph_Exception__POI(_place)
            poi_geodf = geopandas.GeoDataFrame(pd.concat(POI_list,ignore_index=True))
            if save:
                global_poi_filepath = f"{self.file_folder}/{key}{self.maps_name}"
                with open(f"{global_poi_filepath}.geojson", "w") as f:
                    f.write(poi_geodf.to_json())
                
                sumo_cmd = f"ogr2osm {global_poi_filepath}.geojson --output={global_poi_filepath}.osm --force"
                os.system(sumo_cmd)
        if save:
            poi_all_geodf = geopandas.GeoDataFrame(pd.concat(POI_list_all,ignore_index=True))
            global_poi_filepath_all = f"{self.file_folder}/POI__all__{self.maps_name}"
            with open(f"{global_poi_filepath_all}.geojson", "w") as f:
                f.write(poi_all_geodf.to_json())                
            sumo_cmd = f"ogr2osm {global_poi_filepath_all}.geojson --output={global_poi_filepath_all}.osm --force"
            os.system(sumo_cmd)
        return poi_geodf

    def graph_axis(self,show=False):
        """
        projects graph geometry and plots figure, retrieving an axis
        :return: self.fig, self.axis, ax, graph
        """
        # project and plot
        graph = ox.project_graph(self.geograph)
        fig, ax = ox.plot_graph(graph, node_size=0, edge_linewidth=0.5,
                                show=show,
                                bgcolor='#FFFFFF')
        # set the axis title and grab the dimensions of the figure
        self.fig = fig
        ax.set_title(self.maps_name)
        self.axis = ax.axis()
        return ax, graph


class GeoGraph_Exception__Param(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,instance):
        self.instance = instance
          
    def __str__(self):
        return f"No graph found for '{self.instance}' location. Please try a geo-codable place from OpenStreetMaps."

class GeoGraph_Exception__POI(Exception):
    """Exception raised for error no training modality recognized"""
    def __init__(self,instance):
        self.instance = instance
          
    def __str__(self):
        return f"No POI-DATA found for '{self.instance}' location. Please try a geo-codable place from OpenStreetMaps."