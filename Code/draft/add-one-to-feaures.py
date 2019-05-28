import pandas as pd
import re

def add_one_to_indices(indices):
	return ["f{:06}".format(int(re.findall(r"\d+", feature)[0])+1) for feature in indices]
	
