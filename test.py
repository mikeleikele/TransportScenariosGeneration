from src.test import GeoGraph_test,SUMO_test, Orienteering_test, ML
import sys

def geo_simul_test():
    GeoGraph_test.start_test()
    SUMO_test.start_test()

def orient_test():
    Orienteering_test.start_test()

def ml_test():
    ML.start_test()
    
if __name__ == "__main__":
    args = sys.argv[1:]
    if args[0] == "--simulation" or args[0] == "--s":
        geo_simul_test()
    elif args[0] ==  "--orienteering" or args[0] == "--o":
        orient_test()
    
    elif args[0] ==  "--prediction" or args[0] == "--p":
        ml_test()
