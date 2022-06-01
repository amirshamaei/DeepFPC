import csv
import json

from matplotlib import pyplot as plt

import engine as eng
def main():
    file = open('training_report.csv', 'w+')
    writer = csv.writer(file)
    json_file_path = 'runs/test.json'
    with open(json_file_path, 'r') as j:
        contents = json.loads(j.read())
    for run in contents['runs']:
        # if run["child_root"] == "exp1/":
        # if run["version"] == "dSRFPC/":
            # try:
        #     print(":)")
        # else:
        #     run["max_epoch"] = 100
        #     run["numOfSample"] = 100000
        engine = eng.Engine(run)
            # engine.dotrain()
            # writer.writerows([run["child_root"],run["version"], "%100"])
            # plt.close('all')
        engine.dotest()

        # engine.test(128, 1, 40, run['test_load'], True, 64, 0)
        # except:
        # writer.writerows([run["child_root"],run["version"], "Failed"])

    file.close()
if __name__ == '__main__':
    main()