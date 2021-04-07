import numpy as np
import pandas as pd

def main():
    df = pd.read_csv('./data/training.csv',header=None)    
    price = df[3]


    
    pass



if __name__ == '__main__':
    main()