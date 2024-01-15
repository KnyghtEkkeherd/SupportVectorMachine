import numpy, random
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# create 2-dimensional data
def createData(n: int, std: float) -> numpy.array:
    """
    Create data for testing the algorithm

    Args:
        n (int): number of data points
        std (float): standard deviation of the data point distribution

    Returns:
        numpy.array: tsting data
    """
    
    SHIFT1 = [-1, 0.5]
    SHIFT2 = [1, 0.5]
    SHIFT3 = [0, 1.5]
    
    classA = numpy.concatenate(
        (numpy.random.randn(n//4, 2) * std + SHIFT1,
         numpy.random.randn(n//4, 2) * std + SHIFT2))
    classB = numpy.random.randn(n//2, 2) * std + SHIFT3

    inputs = numpy.concatenate((classA, classB))
    targets = numpy.concatenate(
        (numpy.ones(classA.shape[0]),
        -numpy.ones(classB.shape[0])))
    N = inputs.shape[0] # Number of rows (samples)

    # Randomly shuffle the samples
    permute = list(range(N))
    random.shuffle(permute)
    inputs = inputs[permute, :]
    targets = targets[permute]

    return inputs, targets, classA, classB


def plotData(classA: numpy.array, classB: numpy.array, filePath: str=None, 
    sv_array: numpy.array=None) -> None:
    """
    Plot the generated data

    Args:
        classA (numpy.array): array of class A
        classB (numpy.array): array of class B
        filePath (str, optional): file path for saving the plot. Defaults to None.
        sv_array (numpy.array, optional): used when ploting with support vectors. Defaults to None.
    """
    
    plt.plot([p[0] for p in classA],
             [p[1] for p in classA],
             'b.', label="ClassA")
    plt.plot([p[0] for p in classB],
             [p[1] for p in classB],
             'r.', label="ClassB")
    if sv_array is not None:
        x = []
        y = []
        for entry in sv_array:
            x.append(entry[1][0])
            y.append(entry[1][1])
        plt.plot(x, y, 'x', label="Support Vectors")
    plt.axis('equal')
    plt.legend()
    if filePath is not None:
        plt.savefig(filePath)
    plt.show()


def linearKernel(x1: numpy.array, x2: numpy.array) -> float:
    """
    Linear kernel transformation

    Args:
        x1 (numpy.array): first vector
        x2 (numpy.array): second vector

    Returns:
        float: inner product of the two vectors
    """
    
    x1 = numpy.transpose(x1)
    scalarResult = numpy.dot(x1, x2)
    return scalarResult


def polynomialKernel(x1: numpy.array, x2: numpy.array, p: int) -> numpy.array:
    """
    Polynomial kernel transoformation

    Args:
        x1 (numpy.array): first vector
        x2 (numpy.array): second vector
        p (int): polynomial degree

    Returns:
        numpy.array: transformed vector
    """
    
    scalarResult = (numpy.dot(x1, x2)+1)**p
    return scalarResult


def radialKernel(x1: numpy.array, x2: numpy.array, s: int) -> numpy.array:
    """
    Radial kernel transformation

    Args:
        x1 (numpy.array): first vector
        x2 (numpy.array): second vector
        s (int): radial variable

    Returns:
        numpy.array: transformed vector
    """
    
    norm_sq = (numpy.linalg.norm(x1 - x2))**2
    exponent = -0.5*norm_sq/(s**2)
    return numpy.exp(exponent)
    

def extractNonZero(array: numpy.array, threshold_exp: int, inputs: numpy.array, targets: numpy.array) -> numpy.array:
    """
    Extracts non-zero alpha values from the alpha array and creates 
    entries with corresponding values of inputs and target values

    Args:
        array (numpy.array): array of alpha values
        threshold_exp (int, optional): threshold for alpha selection.
        inputs (numpy.array): array of input values
        targets (numpy.array): array of target values for the inputs

    Returns:
        numpy.array: _description_
    """
    
    new_array = []
    for element in range(len(array)):
        if array[element] >= 10**(threshold_exp):
            new_array.append([array[element], inputs[element], targets[element]])
        else:
            continue
    return new_array


def getb(sv_array: numpy.array, c: float=None) -> float:
    """_summary_

    Args:
        sv_array (numpy.array): array with all support vectors
        c (float, optional): margin slack. Defaults to None.

    Returns:
        float: margin offset
    """
    
    if c is None:
        support_vector_alpha = sv_array[0][0]
        support_vector_input = sv_array[0][1]
        support_vector_target = sv_array[0][2]
    else:
        for element in sv_array:
            if element[0] < c:
                support_vector_alpha = element[0]
                support_vector_input = element[1]
                support_vector_target = element[2]

    b = 0.0
    for i in range(len(sv_array)):
        b += sv_array[i][0]*sv_array[i][2]*linearKernel(support_vector_input, sv_array[i][1])
    b -= support_vector_target
    return b


def indicatorFunction(new_data_point: numpy.array, sv_array: numpy.array, b: float) -> float:
    """
    Indicator function to establish whether a point is in class A or class B

    Args:
        new_data_point (numpy.array): new data point vector
        sv_array (numpy.array): support vector array
        b (float): margin of the discrimant function

    Returns:
        float: result of the classification
    """
    
    result = 0.0
    for i in range(len(sv_array)):
        result += sv_array[i][0]*sv_array[i][2]*linearKernel(new_data_point, sv_array[i][1])
    result -= b
    return result
   
    
def plotDecisionBoundary(sv_array: numpy.array, b: float, classA: numpy.array, classB: numpy.array, filePath: str=None) -> None:
    """
    Plot the decision boundary

    Args:
        sv_array (numpy.array): support vector matrix
        b (float): margin offset
        classA (numpy.array): first class array
        classB (numpy.array): second class array
        filePath (str, optional): file path. Defaults to None.
    """
    
    xgrid = numpy.linspace(-2.5, 2.5)
    ygrid = numpy.linspace(-2, 2)
    
    grid = numpy.array([[indicatorFunction([x, y], sv_array, b)
                       for x in xgrid]
                          for y in ygrid])
    plt.contour(xgrid, ygrid, grid,
               (-1.0, 0.0, 1.0),
               colors=('red', 'black', 'blue'),
               linewidths=(1, 1, 1))
    plotData(classA, classB, filePath, sv_array)


def partition(data: numpy.array, fraction: float) -> numpy.array:
    """
    Partition the data into training and validation sets

    Args:
        data (numpy.array): initial training data
        fraction (float): ratio of training to validation

    Returns:
        numpy.array: output arrays
    """
    
    ldata = list(data)
    random.shuffle(ldata)
    breakPoint = int(len(ldata) * fraction)
    return ldata[:breakPoint], ldata[breakPoint:]


def error(classified_data: numpy.array, validation_data: numpy.array) -> float:
    """
    Calculate the fraction of correct answers

    Args:
        classified_data (numpy.array): classified data
        validation_data (numpy.array): ground truth

    Returns:
        float: classification correctness
    """
    
    wrong = 0
    for i in range(len(classified_data)):
        if classified_data[i] != validation_data[i]:
            wrong += 1
        else:
            continue
    return wrong/len(validation_data)


