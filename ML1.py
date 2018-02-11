from numpy import *
import matplotlib.pyplot as plt

def error(m , c ,x , y):
    result = 0
    for i in range(len(x)):
        result += ( (y[i] - (x[i]*m + c)) ** 2 )
    result = result / len(x)
    print(result)

def step_gradient(m_current , c_current , learning_rate , x , y):
    djdm = 0
    djdc = 0
    N = len(x)
    for i in range(len(x)):
        djdm += -(2/N)*x[i]*(y[i] - (m_current*x[i] + c_current))
        djdc += -(2/N)*(y[i] - (m_current*x[i] + c_current))
    m_new = m_current - learning_rate*djdm
    c_new = c_current - learning_rate*djdc
    error(m_new , c_new , x ,y)
    return [m_new , c_new]


def gradient_decent_algo(c , m , learning_rate , x , y , number_of_ittration):
    for i in range(number_of_ittration):
        m , c = step_gradient(m , c , learning_rate , x , y)
    return [m , c]


def run():
    #data set
    points = genfromtxt('data.csv',delimiter=',')
    initial_m = 0
    initial_c = 0
    learning_rate = 0.00001
    number_of_ittration = 10000

    #split the points into x and y
    x = []
    y = []
    for i in range(len(points)):
        x.append(points[i,0])
        y.append(points[i,1])
    plt.scatter(x,y)
    #finding the optimal value of m and c
    m , c = gradient_decent_algo(initial_c , initial_m , learning_rate , x , y , number_of_ittration)
    ypred = []
    for i in range(len(x)):
        ypred.append(m*x[i] + c)
    plt.plot(x,ypred,'r')
    plt.show()
if __name__ == '__main__':
    run()
