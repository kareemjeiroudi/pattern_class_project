from scipy.io import arff

def main():
    with open(r'C:\Users\Stefan\Documents\Dokumente\Studium\6.Semester\Machine Learning and Pattern Classification\Project\train\1.music.arff', 'r') as f:
        #https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.arff.loadarff.html
        data, meta = arff.loadarff(f)
        print(meta)

if __name__ == '__main__':
    main()