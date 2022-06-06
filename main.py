import csv
import json
from matplotlib import pyplot as plt
import engine as eng

def main():
    """
    It opens the json file, reads the contents, and then runs an tests the engine for each run in the json file.
    """
    json_file_path = 'runs/test.json'
    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())
    for run in contents['runs']:
        engine = eng.Engine(run)
        engine.dotrain()
        engine.dotest()
    j.close()

if __name__ == '__main__':
    main()