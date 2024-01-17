# Just makes random datapoints for now for use

import random
import numpy as np

# Generate a thousand points with 10 coordiantes
randNums = np.random.rand(1000, 10)

# With coordiantes ranging from 0 ~ 999
randNums = randNums * 1000
intRandNums = [[int(i) for i in numArray] for numArray in randNums]
print(randNums)
print(intRandNums)

f = open("data.txt", "w")
for vector in intRandNums:
    for coord in vector:
        f.write(str(coord) + " ")
    f.write("\n")
f.close()