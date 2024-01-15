from functions import *

class LinearSV:
    def __init__(self, inputs: numpy.array, targets: numpy.array, classA: numpy.array, classB: numpy.array, C: float=None):
        """
        Linear model support vector machine

        Args:
            inputs (numpy.array): input training data array
            targets (numpy.array): target values for the input training data
            classA (numpy.array): array of the first class
            classB (numpy.array): array of the second class
            C (float, optional): margin slack
        """
           
        self.inputs = inputs
        self.targets = targets
        self.classA = classA
        self.classB = classB
        self.N = len(inputs)
        self.P_matrix = numpy.zeros((self.N, self.N))
        self.C = C
        self.b = 0.0
        self.support_vector_array = []


    def precompute(self):
        for i in range(self.N):
            for j in range(self.N):
                self.P_matrix[i][j] = self.targets[i]*self.targets[j]*linearKernel(self.inputs[i], self.inputs[j])


    def objective(self, alpha: numpy.array) -> float:
        """
        The main objective optimilisation function.
        Args:
            alpha (numpy.array): variable to optimise

        Returns:
            float: Score for the given alpha
        """
        
        sum_result = 0.0
        for i in range(self.N):
            for j in range(self.N):
                sum_result += alpha[i]*alpha[j]*self.P_matrix[i][j]
        sum_result = 0.5*sum_result
        minus_sum = numpy.sum(alpha)
        sum_result = sum_result - minus_sum
        return sum_result


    def zerofun(self, alpha: numpy.array) -> float:
        """
        The optimisation constraint function

        Args:
            alpha (numpy.arraty): variable to optimise

        Returns:
            float: Score for the given alpha given the constraint
        """
        
        return numpy.dot(self.targets, alpha)          
        

    def indicatorFunction(self, new_data_point):
        """
        Indicator funciton used to classify new data points and calculating the margin of the divider

        Args:
            new_data_point (numpy.array): A new data point vector or a support vector

        Returns:
            float: Result of the classification. Postive is class A and negative is class B
        """
        
        result = 0.0
        for i in range(len(self.support_vector_array)):
            result += self.support_vector_array[i][0]*self.support_vector_array[i][2]*linearKernel(new_data_point, self.support_vector_array[i][1])
        result -= self.b
        return result


    def plot(self, fileName: str=None) -> None:
        xgrid = numpy.linspace(-4, 4)
        ygrid = numpy.linspace(-4, 4)

        grid = numpy.array([[self.indicatorFunction([x, y]) for x in xgrid] for y in ygrid])
        plt.contour(xgrid, ygrid, grid,
                    (-1.0, 0.0, 1.0),
                    colors=('red', 'black', 'blue'),
                    linewidths=(1, 1, 1))
        plt.title(f"""Data Points N={self.N}""")
        plotData(self.classA, self.classB, fileName, self.support_vector_array)


    def train(self, alpha_threshold: int=-5) -> None:
        """
        Train the model on the data provided.

        Args:
            alpha_threshold (int, optional): Threshold for the alpha selection. Defaults to -5.
        """
        
        self.precompute()
        XC = {'type':'eq', 'fun':self.zerofun}
        start = numpy.zeros((self.N, ))
        B = [(0, self.C) for b in range(self.N)] 
        ret = minimize(self.objective, start, bounds=B, constraints=XC)
        sv_array = extractNonZero(ret['x'], alpha_threshold, self.inputs, self.targets)
        sv_b = getb(sv_array, self.C)
        self.support_vector_array = sv_array
        self.b = sv_b

      
    def classify(self, new_data_point: numpy.array) -> int:
        """
        Classify a new data point

        Args:
            new_data_point (numpy.array): A new point to classify

        Returns:
            int: Result of the classification: 1 means class A, -1 means class B
        """
        
        value = self.indicatorFunction(new_data_point)
        if value > 0:
            return 1
        elif value < 0:
            return -1


class PolynomialSV(LinearSV):
    def __init__(self, *args, **kwargs):
        super(PolynomialSV, self).__init__(*args, **kwargs)
        self.p = 0.0

  
    def precompute(self):
        for i in range(self.N):
            for j in range(self.N):
                self.P_matrix[i][j] = self.targets[i]*self.targets[j]*polynomialKernel(self.inputs[i], self.inputs[j], self.p)
 
    
    def indicatorFunction(self, new_data_point: numpy.array) -> float:
        """
        Indicator function for the polynomial function

        Args:
            new_data_point (numpy.array): A new data point or a support vector

        Returns:
            float: result of the classification
        """
        
        result = 0.0
        for i in range(len(self.support_vector_array)):
            result += self.support_vector_array[i][0]*self.support_vector_array[i][2]*polynomialKernel(new_data_point, self.support_vector_array[i][1], self.p)
        result -= self.b
        return result
  
    
    def train(self, p: int, alpha_threshold: int=-5) -> None:
        """
        Train the model on the provided data set

        Args:
            p (int): degree of the polynomial
            alpha_threshold (int, optional): threshold exponent. Defaults to -5.
        """
        
        self.p = p
        self.precompute()
        XC = {'type':'eq', 'fun':self.zerofun}
        start = numpy.zeros((self.N, ))
        B = [(0, self.C) for b in range(self.N)] 
        ret = minimize(self.objective, start, bounds=B, constraints=XC)
        sv_array = extractNonZero(ret['x'], alpha_threshold, self.inputs, self.targets)
        sv_b = self.getb(sv_array, self.C)
        self.support_vector_array = sv_array
        self.b = sv_b

        
    def getb(self, support_vectors_array: numpy.array, C: float=None) -> float:
        """
        Get the margin value for the discriminant function

        Args:
            support_vectors_array (numpy.array): support vector array 
            C (float, optional): margin slack. Defaults to None.

        Returns:
            float: margin offset
        """
        
        if C is None:
            support_vector_alpha = support_vectors_array[0][0]
            support_vector_input = support_vectors_array[0][1]
            support_vector_target = support_vectors_array[0][2]
        else:
            for element in support_vectors_array:
                if element[0] < C:
                    support_vector_alpha = element[0]
                    support_vector_input = element[1]
                    support_vector_target = element[2]

        b = 0.0
        for i in range(len(support_vectors_array)):
            b += support_vectors_array[i][0]*support_vectors_array[i][2]*polynomialKernel(support_vector_input, support_vectors_array[i][1], self.p)
        b -= support_vector_target
        return b
 
        
class RadialSV(LinearSV):
    def __init__(self, *args, **kwargs):
        super(RadialSV, self).__init__(*args, **kwargs)
        self.s = 0.0


    def precompute(self):
        for i in range(self.N):
            for j in range(self.N):
                self.P_matrix[i][j] = self.targets[i]*self.targets[j]*radialKernel(self.inputs[i], self.inputs[j], self.s)
    
    
    def indicatorFunction(self, new_data_point):
        """
        Indicator funciton used to classify new data points and calculating the margin of the divider

        Args:
            new_data_point (numpy.array): A new data point vector or a support vector

        Returns:
            float: Result of the classification. Postive is class A and negative is class B
        """
        
        result = 0.0
        for i in range(len(self.support_vector_array)):
            result += self.support_vector_array[i][0]*self.support_vector_array[i][2]*radialKernel(new_data_point, self.support_vector_array[i][1], self.s)
        result -= self.b
        return result
  
    
    def train(self, s: int, alpha_threshold: int=-5) -> None:
        """
        Train the model on the provided data set

        Args:
            s (int): radial variable
            alpha_threshold (int, optional): threshold exponent for alpha selection. Defaults to -5.
        """
        
        self.s = s
        self.precompute()
        XC = {'type':'eq', 'fun':self.zerofun}
        start = numpy.zeros((self.N, ))
        B = [(0, self.C) for b in range(self.N)] 
        ret = minimize(self.objective, start, bounds=B, constraints=XC)
        sv_array = extractNonZero(ret['x'], alpha_threshold, self.inputs, self.targets)
        sv_b = self.getb(sv_array, self.C)
        self.support_vector_array = sv_array
        self.b = sv_b
        

    def getb(self, support_vectors_array: numpy.array, C: float=None) -> float:
        """
        Get the margin value for the discriminant function

        Args:
            support_vectors_array (numpy.array): support vector array 
            C (float, optional): margin slack. Defaults to None.

        Returns:
            float: margin offset
        """
        
        if C is None:
            support_vector_alpha = support_vectors_array[0][0]
            support_vector_input = support_vectors_array[0][1]
            support_vector_target = support_vectors_array[0][2]
        else:
            for element in support_vectors_array:
                if element[0] < C:
                    support_vector_alpha = element[0]
                    support_vector_input = element[1]
                    support_vector_target = element[2]

        b = 0.0
        for i in range(len(support_vectors_array)):
            b += support_vectors_array[i][0]*support_vectors_array[i][2]*radialKernel(support_vector_input, support_vectors_array[i][1], self.s)
        b -= support_vector_target
        return b