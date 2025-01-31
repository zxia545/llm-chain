from utils import get_training_data
import argparse




def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", type=str, required=True)
    parser.add_argument("-o", type=str, required=True)
    args = parser.parse_args()

    get_training_data(args.input_file, args.output_file)
    
if __name__ == "__main__":
    main()