import xml.etree.ElementTree as ET
from xml.dom import minidom
import ast
from dict2xml import dict2xml
import math
import statistics
from tqdm.auto import tqdm


class SUMO_roadstats():
    def __init__(self, simulation_name):
        self.this = True
        self.netFile = f"data\sumo_simulation_files\{simulation_name}\{simulation_name}.net.xml"
        self.fcdFile = f"data\sumo_simulation_files\{simulation_name}\{simulation_name}.out.fcd.xml"
        self.outFile = f"data\sumo_simulation_files\{simulation_name}\{simulation_name}.out.roadstats.xml"
    
    
    """
    segmentsMode :
        "fixed" - define a fixed distance, i.e every segment rappresnet 200meters of road
        "relative" - define a number of segment, i.e. all road have 5 segments
    """
    def compute_roadstats(self, segmentsMode="relative", segmentsValue=5, save=True):
        self.segmentsMode = segmentsMode
        self.segmentsValue = segmentsValue
        self.net2dict()
        self.fcd2dict()
        if save:
            self.writeFile()

    def net2dict(self):
        treeNET = ET.parse(self.netFile)
        self.rootNET = treeNET.getroot()
        self.lanes = dict()

        for item in self.rootNET:
            if item.tag == "edge":
                if 'function' not in item.attrib:                    
                    
                    roadInfo_from = item.attrib['from']
                    roadInfo_to = item.attrib['to']
                    roadInfo_priority = item.attrib['priority']
                    roadInfo_stret = item.attrib['id']
                    for sub_item in item:
                        roadInfo = dict()
                        roadInfo['from'] = roadInfo_from
                        roadInfo['to'] = roadInfo_to
                        roadInfo['priority'] = roadInfo_priority
                        roadInfo['street'] = roadInfo_stret
                        roadInfo['index'] = sub_item.attrib['index']
                        roadInfo['speed'] = sub_item.attrib['speed']
                        roadInfo['length'] = float(sub_item.attrib['length'])
                        
                        if self.segmentsMode == "relative":
                            segment_length = math.ceil(roadInfo['length']/self.segmentsValue)
                            segment_number = self.segmentsValue
                        elif self.segmentsMode == "fixed":
                            segment_length = self.segmentsValue
                            segmentsValue = math.ceil(roadInfo['length']/self.segmentsValue)
                        else:
                            raise NotImplementedError()

                        roadInfo['segment_length'] = segment_length
                        roadInfo['segment_number'] = segment_number
                        roadInfo['vehicles_trafic'] = dict()
                        for i in range(0, segment_number):
                            roadInfo['vehicles_trafic'][str(i)] = list()
                        lane_id = sub_item.attrib['id']
                        self.lanes[lane_id] = roadInfo

    def fcd2dict(self):
        treeFCD = ET.parse(self.fcdFile)
        self.rootFCD = treeFCD.getroot()
        
        for item in tqdm(self.rootFCD):    
            if item.tag == "timestep":
                timestep = item.attrib['time']
                for sub_item in item:
                    lane_id = sub_item.attrib['lane']
                    if lane_id in self.lanes:
                        pos_float = float(sub_item.attrib['pos'])
                        if self.lanes[lane_id]['segment_length'] == 0:
                            segment = 0
                        else:
                            segment = math.floor(pos_float/self.lanes[lane_id]['segment_length'])
                        if segment == self.lanes[lane_id]['segment_number']:
                            segment -= 1
                        veh_info = {'vehicle_id':sub_item.attrib['id'], 'speed':sub_item.attrib['speed'], 'position':sub_item.attrib['pos'], 'timestep':timestep}
                        self.lanes[lane_id]['vehicles_trafic'][str(segment)].append(veh_info)

    def writeFile(self, write_segments=True, write_vehicle=True):
        roads_root = ET.Element("roads")
        for lane in self.lanes:
            info_lane = self.lanes[lane]
            road = ET.SubElement(roads_root, "lane") 
            road.set("lane_id",lane)
          
            road.set("from", info_lane['from']) 
            road.set("to", info_lane['to'])
            road.set("priority", info_lane['priority']) 
            road.set("street", info_lane['street']) 
            road.set("index", info_lane['index']) 
            road.set("speed", info_lane['speed']) 
            road.set("length", str(info_lane['length']))
            if write_segments:
                road.set("segments_length", str(info_lane['segment_length']))
                road.set("segments_number", str(info_lane['segment_number']))
            road_speed_list = list()

            for i in range(info_lane['segment_number']):
                if write_segments:
                    traffic_segment = ET.SubElement(road, "traffic", segment=str(i))
                traffic_vehicles = info_lane['vehicles_trafic'][str(i)]
                segment_speed_list = list()
                for traffic_vehicle_point in traffic_vehicles:
                    if write_vehicle:
                        vehicle_point = ET.SubElement(traffic_segment, "vehicle")
                        vehicle_point.set("vehicle_id", traffic_vehicle_point['vehicle_id']) 
                        vehicle_point.set("speed", traffic_vehicle_point['speed'])
                        vehicle_point.set("pos", str(traffic_vehicle_point['position']) )
                        vehicle_point.set("timestep", str(traffic_vehicle_point['timestep']))
                    if write_segments:
                        segment_speed_list.append(float(traffic_vehicle_point['speed']))
                    road_speed_list.append(float(traffic_vehicle_point['speed']))
                if write_segments:
                    if len(segment_speed_list) > 0:
                        segment_speed_mean = statistics.mean(segment_speed_list)
                    else:
                        segment_speed_mean = 'Nan'    
                    if len(segment_speed_list) > 1:
                        segment_speed_std = statistics.stdev(segment_speed_list)
                    else:                        
                        segment_speed_std = 'Nan'
                    
                    traffic_segment.set("speed_mean", str(segment_speed_mean))
                    traffic_segment.set("speed_std", str(segment_speed_std))            
            if len(road_speed_list) > 0:
                road_speed_mean = statistics.mean(road_speed_list)
            else:
                road_speed_mean = 'Nan'    
            if len(road_speed_list) > 1:
                road_speed_std = statistics.stdev(road_speed_list)
            else:                        
                road_speed_std = 'Nan'
                    


            road.set("speed_mean", str(road_speed_mean))
            road.set("speed_std", str(road_speed_std))

        xmlstr = minidom.parseString(ET.tostring(roads_root)).toprettyxml(indent="   ")
        with open(f"{self.outFile}", "w") as f:
            f.write(xmlstr)