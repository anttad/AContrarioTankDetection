import argparse
import json
import glob
import os

def main(input_folder):
    
    files = glob.glob(os.path.join(input_folder, '*.json'))
    
    for f in files:
        with open(f) as json_file:
            data = json.load(json_file)
            if len(str(data)) >=4:
                print(data)
                print(f)
                print('           ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=('study clusters'))
    parser.add_argument('--input', help=('folder that all the input images are stored'))
    args = parser.parse_args()
    main(args.input)
