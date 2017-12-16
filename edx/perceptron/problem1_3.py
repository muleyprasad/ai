from sys import *
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


def main():
    inputFile = argv[1]
    outputFile = argv[2]
    data = np.genfromtxt(inputFile, delimiter=',', skip_header = 0,
                            skip_footer = 0, names = ['x', 'y', 'z'])
    w = pla(data, outputFile)
    # w = [1,1,1]
    plotData(data, w)

def pla(data, outputFile):
    # Initialize the weights wj to 0 8j 2 {0,··· ,d}
    w = [0,0,0]
    # Repeat until convergence
    convergence = False
    f = open(outputFile,'wt')
    while not convergence:
        # save a copy of weights to determine convergence
        wCopy = list(w)
        # For each example xi 8i 2 {1,··· ,n}
        for i in range(0,len(data)):
            # if yif(xi)  0 #an error?
            f_of_x = -1 if ((w[0] * 1) + (w[1] * data[i]['x']) + (w[2] * data[i]['y'])) < 0 else 1
            if data[i]['z'] * f_of_x <= 0:
                # update all wj with wj := wj + yixi #adjust the weights
                w[0] = w[0] + (data[i]['z'] * 1)
                w[1] = w[1] + (data[i]['z'] * data[i]['x'])
                w[2] = w[2] + (data[i]['z'] * data[i]['y'])

        if wCopy == w:
            convergence = True
        f.write(str(w[1]) + "," + str(w[2]) + "," + str(w[0]) + "\n")
    return w

def plotData(data, w):
    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    for d in data:
        ax1.plot(d['x'], d['y'], 'ro' if d['z'] == 1 else 'bo')

    # Draw decision boundry
    # initialize x
    x = list(range(1,20))
    # use for w0 + w1x1 + w2x2 = 0 forumula for generating y
    y = [ (-w[0] - w[1]* xi)/w[2] for xi in x]
    plt.plot(x,y)
    plt.show()
    
if __name__ == "__main__":
	main()

