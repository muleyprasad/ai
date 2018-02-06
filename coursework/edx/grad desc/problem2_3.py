# Data structure: age, weight, height

# Data Preparation and Normalization:
# 1. add the vector 1 (intercept) ahead of your matrix
# 2. data matrix columns are 
#     * intercept, age(years), weight(kilograms), height(meters) *
#     age-    feature 1
#     weight- feature 2
#     height- label
# 3. determine standard deviation of each feature, set mean to 0
# 4. Scale each feature (i.e. age and weight) by its (population) standard deviation, and set its mean to zero. 
# 
# Implement gradient descent to find a regression model
#     a. Initialize your β’s to zero
#     b. Calculate β j's by given formula until convergance?
#     b. Run the gradient descent algorithm using the following learning rates: 
#         α ∈ {0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10}.
#     c. 

from sys import *
import numpy as np
# import matplotlib as mpl
# import matplotlib.pyplot as plt

def main():
    inputFile = argv[1]
    outputFile = argv[2]
    data = np.genfromtxt(inputFile, delimiter=',')
    # data = np.genfromtxt(inputFile, delimiter=',', skip_header = 0,
    #                         skip_footer = 0, names = ['x', 'y', 'z'])

    # 1. add the vector 1 (intercept) ahead of your matrix
    intercept = np.ones(len(data))
    data = np.c_[intercept,data]
    # 3. determine standard deviation of each feature, set mean to 0
    stdDev = np.std(data,axis=0)
    mean = np.mean(data,axis=0)

    # 4. Scale each feature (i.e. age and weight) by its (population) standard deviation, and set its mean to zero. 
    dataScaled = np.copy(data)
    for itm in dataScaled:
        itm[1] = (itm[1]-mean[1])/stdDev[1]
        itm[2] = (itm[2]-mean[2])/stdDev[2]

    
    # Repeat until convergence
    # convergence = False
    f = open(outputFile,'wt')
    # while not convergence:
    for alpha in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10,0.2]:
        beta = [0,0,0]
        for item in list(range(0,100)):
            for j in list(range(0,3)):
                f_of_x = sum([ ((d[0]*beta[0]) + (d[1]*beta[1]) + (d[2]*beta[2]) - d[3]) * d[j]  for d in dataScaled])
                beta[j] = beta[j] - ((alpha/len(dataScaled))*f_of_x)
        # print(beta)
        f.write(str(alpha) + ",100," + str(beta[0]) + "," + str(beta[1]) + "," + str(beta[2]) + "\n")
        

if __name__ == "__main__":
	main()