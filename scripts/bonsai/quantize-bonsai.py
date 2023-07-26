#!/usr/bin/env python3

"""Script to execute the "quantize" script on a given set of models."""

import subprocess
import argparse
import glob
import sys
import os


def main():
    """Update the quantize binary name depending on the platform and parse
    the command line arguments and execute the script.
    """

    if "linux" in sys.platform or "darwin" in sys.platform:
        quantize_script_binary = "quantize-bonsai"

    elif "win32" in sys.platform or "cygwin" in sys.platform:
        quantize_script_binary = "quantize-bonsai.exe"

    else:
        print("WARNING: Unknown platform. Assuming a UNIX-like OS.\n")
        quantize_script_binary = "quantize-bonsai"

    parser = argparse.ArgumentParser(
        prog='python3 quantize-bonsai.py',
        description='This script quantizes the given models by applying the '
        f'"{quantize_script_binary}" script on them.'
    )
    parser.add_argument('model_path')

    parser.add_argument(
        '-r', '--remove-orig', action='store_true', dest='remove_orig',
        help='Remove the old model after quantizing it.'
    )
    
    parser.add_argument(
        '-q', '--quantize-script-path', dest='quantize_script_path',
        default=os.path.join(os.getcwd(), quantize_script_binary),
        help='Specify the path to the "quantize" script.'
    )

    args = parser.parse_args()
    args.model_path = os.path.abspath(args.model_path)
    
    if not os.path.isfile(args.quantize_script_path):
        print(
            f'The "{quantize_script_binary}" script was not found in the '
            "current location.\nIf you want to use it from another location, "
            "set the --quantize-script-path argument from the command line."
        )
        sys.exit(1)
    
    model_path_base = args.model_path

    if not os.path.isfile(model_path_base):
        print(f'The file %s was not found' % model_path_base)
        sys.exit(1)

    model_parts_paths = map(
        lambda filename: os.path.join(model_path_base, filename),
        glob.glob(f"{model_path_base}*")
    )

    for model_part_path in model_parts_paths:
        if not os.path.isfile(model_part_path):
            print(
                f"The original model {os.path.basename(model_part_path)} "
                f"was not found in {args.models_path}{os.path.sep}{model}"
                ". If you want to use it from another location, set the "
                "--models-path argument from the command line."
            )
            sys.exit(1)

        __run_quantize_script(
            args.quantize_script_path, model_part_path
        )

        if args.remove_orig:
            os.remove(model_part_path)

def __run_quantize_script(script_path, model_part_path):
    """Run the quantize script specifying the path to it and the path to the
    original model to quantize.
    """
    
    quant = "q8_0" #"q4_1" #"q8_0" #"q4_0"
    quant_id = "7" #"3" #"8" #"2"

    # Replace f32 or f16, whichever is present
    new_quantized_model_path = model_part_path.replace("f32", quant).replace("f16", quant)
    subprocess.run(
        [script_path, model_part_path, new_quantized_model_path, quant_id],
        check=True
    )


if __name__ == "__main__":
    try:
        main()

    except subprocess.CalledProcessError:
        print("\nAn error ocurred while trying to quantize the models.")
        sys.exit(1)

    except KeyboardInterrupt:
        sys.exit(0)

    else:
        print("\nSuccesfully quantized all models.")
