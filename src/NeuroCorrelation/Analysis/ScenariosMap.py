import pandas as pd
import numpy as np
import folium
from folium import plugins
from colormap import rgb2hex, rgb2hls, hls2rgb
import matplotlib
from pathlib import Path
import os

class ScenariosMap():
    
    def __init__(self, data_range, vc_mapping, path_folder_map, path_folder, instance_file, label="x_output"):
        self.case_map = pd.read_csv(path_folder_map)
        self.label = label
        self.vc_mapping = vc_mapping
        points_list = list()
        for xs in self.case_map['points']:
            points = list()
            xxs = (xs[2:-2].split("), ("))
            for item in xxs:
                x = [float(x) for x in item.split(", ")]
                x_tuple = (x[0], x[1])
                points.append(x_tuple)
            points_list.append(points)
        self.case_map['points'] = points_list
        
        self.path_folder = Path(path_folder,"scenarios_map")
        if not os.path.exists(self.path_folder):
            os.makedirs(self.path_folder)
            
        
        path_folder_istances = Path(path_folder,f"{instance_file}.csv")
        self.scenarios = pd.read_csv(path_folder_istances)
        self.max_val = data_range["max_val"]
        self.min_val = data_range["min_val"]
        self.center = (np.mean([x[0] for xs in self.case_map['points'] for x in xs]), np.mean([x[1] for xs in self.case_map['points'] for x in xs]))
        
        
        
    def darken_color(self, r, g, b, factor=0.3):
        return self.adjust_color_lightness(r, g, b, 1 - factor)

    def adjust_color_lightness(self, r, g, b, factor):
        h, l, s = rgb2hls(r, g, b)
        l = max(min(l * factor, 1.0), 0.0)
        r, g, b = hls2rgb(h, l, s)
        return rgb2hex(int(r*255), int(g*255), int(b*255))

    def get_route_color(self, value):
        cmap = matplotlib.colormaps['RdYlGn']
        value_x = (value - self.min_val)/(self.max_val - self.min_val)
        if value_x<=0:
            value_x=0
        elif value_x>=1:
            value_x = 1
        rgba = cmap(value_x)
        lightness_factor = (0.5/1) * float(1)
        cl = self.darken_color(rgba[0], rgba[1], rgba[2], lightness_factor)
        return cl
    

    def draw_scenario(n_scenario, save_html=True):
        scenario_list = [float(x) for x in self.scenarios[self.label][n_scenario][1:-1].split(", ")]
        scenario = dict()
        for road, value in zip(self.vc_mapping, scenario_list):
            scenario[road] = value
  
        maps_scenario = folium.Map(self.center,tiles="cartodbdark_matter",  zoom_start=11)
        folium.TileLayer('cartodbdark_matter').add_to(maps_scenario)
        folium.TileLayer('openstreetmap').add_to(maps_scenario)
  
        folium.LayerControl().add_to(maps_scenario)
        for index, row in map.iterrows():
            value = scenario[row["name_road"]]
            route_color = self.get_route_color(value)
            route = folium.PolyLine(row["points"], color=route_color, weight=5, opacity=0.85).add_to(maps_scenario)
            route.add_to(maps_scenario)
        if save_html:
            folder_html_file = Path(self.path_folder , f"scenario_map_{n_scenario}.html")
            maps_scenario.save(folder_html_file)
    
    def draw_scenarios(self, list_scenarios=None):
        if list_scenarios is None:
            list_scenarios = [i for i in range(len(scenarios))]
        for i in range(list_scenarios):
            self.draw_scenario(n_scenario=i, save_html=True)