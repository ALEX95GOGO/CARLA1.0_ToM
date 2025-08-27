#!/usr/bin/env python

import argparse
import os
import carla

def main():

    argparser = argparse.ArgumentParser(
        description=__doc__)
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-i', '--input_folder',
        metavar='I',
        default="input_folder",
        help='input folder containing recorder files')
    argparser.add_argument(
        '-o', '--output_folder',
        metavar='O',
        default="output_folder",
        help='output folder for saving results')
    argparser.add_argument(
        '-a', '--show_all',
        action='store_true',
        help='show detailed info about all frames content')

    args = argparser.parse_args()

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)

        # Ensure the output folder exists
        os.makedirs(args.output_folder, exist_ok=True)

        # Process all files in the input folder
        for filename in os.listdir(args.input_folder):
            if filename.endswith(".bin") or filename.endswith(".rec"):
                input_path = os.path.join(args.input_folder, filename)
                if args.show_all:
                    print(f"Processing file: {input_path}")
                result = client.show_recorder_file_info(input_path, args.show_all)
                
                # Save the result to the output folder
                output_filename = os.path.splitext(filename)[0] + "_result.txt"
                output_path = os.path.join(args.output_folder, output_filename)
                with open(output_path, "w+") as output_file:
                    output_file.write(result)
                    if args.show_all:
                        print(f"Saved result to: {output_path}")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        pass

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
    finally:
        print('\ndone.')
