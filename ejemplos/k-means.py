#implementación del algoritmo de k means
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

def plot_classif(X, X0, classif):
	plt.figure()
	plt.scatter(X[:,0], X0[:, 1], marker = 'x', c = classif)
	plt.scatter(X[:,0], X0[:, 1], marker = 'o', c = sorted(np.unique(classif)))
	plt.title('Muestras y centroides')

iris = load_iris()
X = iris.data[:, 2:4] #datos a agrupar

sc = StandardScaler()
sc.fit(X) #Normalizo
X_std = sc.transform(X)

# Inicializació: elegimos K centroides, en este caso 3
n_iters = 10
K = 3
idx = np.random.choice(X.shape[0],K,replace=False)

X0 = X[idx, :]

for n in range(n_iters):
	#calculo las distancias
	d_sq = np.zeros([X.shape[0], K])
	for i_k in range (K):
		d_sq[:, i_k] = np.sum((X0-X[i_k,:])**2, axis=1) #hace broadcasting en X0 (a cada valor de X le resto el centroide). Calculo las distancias al primer eje

	#clasificamos a lo largo del eje 1
	classif = np.argmin(d_sq, axis=1) #argumento que minimiza la distancia a lo largo del eje 1

	plot_classif(X, X0, classif)

	#fase 2: reasignación del centroide
	for i_k in range(K):
		X0[i_k,:] = np.mean(X[classif==i_k], axis=0)
		
	plot_classif(X, X0, classif)

