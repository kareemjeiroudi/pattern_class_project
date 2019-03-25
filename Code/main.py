from scipy.io import arff

def main():
    with open(r'data/train/1.music.arff', 'r') as f:
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.loadarff.html
        data, meta = arff.loadarff(f)
        print(meta)
        print(data)
        ## TODO: Sanity check: each file should contain around 5 x 3,600 = 18,000 examples.
        

if __name__ == '__main__':
    main()