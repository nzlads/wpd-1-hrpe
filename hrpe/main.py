import os


from hrpe.data import *
from hrpe.data.load import load_hh_data, load_maxmin_data, load_minute_data
from hrpe.data.clean import load_hh_data, load_maxmin_data, load_minute_data


def main():
    
    print(os.getcwd())

    # Set Vars
    substation = 'staplegrove'
    

    # Load data using load.py function for staplegrove
    # Load data function

    hh_data = load_hh_data(substation=substation)
    maxmin_data = load_maxmin_data(substation=substation)
    minute_data = load_minute_data(substation=substation)


    # Data clean
    hh_data     = clean_hh_data(    data = hh_data)
    maxmin_data = clean_maxmin_data(data = maxmin_data)
    minute_data = clean_minute_data(data = minute_data)

    # Build features / differences

    min2hh_data = minute_data_to_hh_data(maxmin_data)







# Model
# Naive fit
# naive predict


# Scoring
# RMSE score


# Debugging plots


##
# standards:
# minmax = truth
# hh = halfhour


if __name__ == '__main__':
    # execute only if run as the entry point into the program
    main()
