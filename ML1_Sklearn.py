
import numpy
import matplotlib.pyplot as plt
def run():
    points = numpy.genfromtxt('data.csv',delimiter = ',')
    x = []
    y = []
    for i in range(len(points)):
        x.append(points[i,0])
        y.append(points[i,1])
    x = numpy.array(x).reshape(len(points),1)
    y = numpy.array(y).reshape(len(points),1)
    plt.scatter(x,y)
    from sklearn import linear_model
    model = linear_model.LinearRegression()
    model.fit(x,y)
    ypred = model.predict(x)
    plt.plot(x,ypred,'r')
    plt.show()
    print(ypred)
if __name__ == '__main__':
    run()
