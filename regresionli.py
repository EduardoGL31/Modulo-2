'''
Algoritmo de regresion lineal

- Hipotesis de la funcion h(x) = m*x + b
- Costo de la funcion J(theta) = 1/2m * sum((h(x) - y)^2)
- Algoritmo de gradiente descendiente

'''

import numpy as np  
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 

'''Esta funcion evalua la funcion lineal generica h(x) con los paramatros actuales'''
def hyp(m_b, x):
    acum = 0
    size_mb = len(m_b)
    for i in range(size_mb):
        acum = acum + m_b[i]*x[i]
    return acum

''' Esta funcion calcula el costo de la funcion con MSEJ(theta) = 1/2m * sum((h(x) - y)^2)'''
def costo(m_b, x, y):
    acum = 0
    size = len(x)
    for i in range(size):
        h = hyp(m_b, x[i])
        error = h - y[i]
        acum = acum + error**2
    errores.append(acum/size)

''' Aqui se aplica el algoritmo de gradiente descendente repetidamente'''

def gradiente(m_b, x, y, alpha):
    size_mb = len(m_b)
    size_x = len(x)
    theta = list(m_b)
    for i in range(size_mb):
        acum = 0
        for j in range(size_x):
            error = hyp(m_b, x[j]) - y[j]
            acum = acum + error*x[j][i]
        theta[i] = theta[i] - alpha*(1/size_x)*acum
    return theta

'''Esta funcion escala los datos en un rango de 0 y 1'''
def minmax(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))

'''Esta funcion agrega unos para poder hacer las operaciones'''
def add(x):
    for i in range(len(x)):
        x[i].append(1)
    return x

errores = []
if __name__ == "__main__":
    columnas = ['Sex', 'Length', 'Diameter', 'Height', 'Whole weight', 'Shucked weight', 'Viscera weight', 'Shell weight', 'Rings']
    #df = pd.read_csv('C:/Users/Edu/Desktop/ML2/abalone.data', names=columnas)
    df = pd.read_csv('data/abalone.data', names=columnas)
    df_x = df.iloc[:, 5:8].to_numpy().tolist()
    df_y = df.iloc[:, 4].to_numpy().tolist()
    params = np.zeros(len(df_x[0])+1).tolist()

    # dividimos los parametros de prueba y de entrenamiento
    x_train, xtest, y_train, y_test = train_test_split(df_x, df_y, random_state=1)

    lr = 0.05  # learning rate
    epochs = 0

    x_train = add(x_train)
    
    # hacemos el entrenamiento 
    while True:  
        oparams = list(params)
        params = gradiente(params, x_train, y_train, lr)	
        costo(params, x_train, y_train)  
        epochs += 1
        if(oparams == params or epochs == 200):   
            print ("params finales:",params)            
            print("error:",errores[-1])
            break

    plt.plot(errores)

    # Checamos el test
    xtest = add(xtest)
    yp = [np.dot(x, params) for x in xtest]

    acum_e = 0
    for i in range(len(y_test)):
        acum_e = acum_e + (y_test[i] - yp[i]) ** 2
    
    print("Erros del test:",acum_e)

    plt.show()
