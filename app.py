import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# Multiclass Perceptron Class
class PerceptronMulticlass:
    def __init__(self, learning_rate=0.01, n_iters=1000, n_classes=3, n_features=2):
        self.learning_rate = learning_rate
        self.n_iters = n_iters
        self.n_classes = n_classes
        self.n_features = n_features

        # Inicializar pesos y sesgos a ceros
        self.W = np.zeros((n_classes, n_features))  # Matriz de pesos
        self.b = np.zeros(n_classes)                # Vector de sesgos 

    def step_function(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        y = y.flatten().astype(int)
        for _ in range(self.n_iters):
            for idx, x_i in enumerate(X):
                x_i = x_i.reshape(1, -1)
                target = np.zeros(self.n_classes)
                target[y[idx]] = 1  # One-hot encoding
                linear_output = np.dot(self.W, x_i.T).flatten() + self.b
                y_predicted = self.step_function(linear_output)
                update = self.learning_rate * (target - y_predicted)
                self.W += np.outer(update, x_i)
                self.b += update

    def predict(self, X):
        linear_output = np.dot(X, self.W.T) + self.b
        y_predicted = self.step_function(linear_output)
        return np.argmax(y_predicted, axis=1)

    def plot_decision_boundary(self, X, y):
        # Visualización del límite de decisión
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                             np.arange(y_min, y_max, 0.05))
        grid = np.c_[xx.ravel(), yy.ravel()]
        Z = self.predict(grid)
        Z = Z.reshape(xx.shape)
        plt.figure()
        plt.contourf(xx, yy, Z, cmap=plt.cm.Paired, alpha=0.3)
        plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap=plt.cm.Paired, edgecolors='k', marker='o')
        st.pyplot(plt.gcf())
        plt.close()

# Función principal
def main():
    st.title("Perceptrón Multiclase")
    col1, col2 = st.columns(2)

    # Columna Izquierda: Generación de Datos
    with col1:
        st.header("Generación de Datos")
        num_samples = st.number_input("Número de Muestras", min_value=10, max_value=10000, value=1000, step=10)
        num_classes = st.number_input("Número de Clases", min_value=2, max_value=10, value=3, step=1)
        if st.button("Generar Datos"):
            X, y = make_blobs(n_samples=int(num_samples), centers=int(num_classes), n_features=2, random_state=42)
            y = y.reshape((len(y), 1))
            fig, ax = plt.subplots()
            ax.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='viridis', edgecolors='k', marker='o')
            st.pyplot(fig)
            st.session_state['X'] = X
            st.session_state['y'] = y
        
        elif 'X' in st.session_state and 'y' in st.session_state:
            X = st.session_state['X']
            y = st.session_state['y']
            fig, ax = plt.subplots()
            ax.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='viridis', edgecolors='k', marker='o')
            st.pyplot(fig)

    # Columna Derecha: Ejecutar Perceptrón
    with col2:
        st.header("Ejecutar Perceptrón")
        if 'X' in st.session_state and 'y' in st.session_state:
            learning_rate = st.number_input("Tasa de Aprendizaje (η)", min_value=0.0001, max_value=1.0, value=0.01)
            n_iter = st.number_input("Número de Iteraciones (T)", min_value=1, max_value=10000, value=100)

            X = st.session_state['X']
            y = st.session_state['y']
            n_features = X.shape[1]
            n_classes = len(np.unique(y))

            # Inicializar el modelo
            model = PerceptronMulticlass(
                learning_rate=learning_rate,
                n_iters=int(n_iter),
                n_classes=int(n_classes),
                n_features=n_features
            )
            
            st.session_state['model'] = model

            # Entrenar el perceptrón y hacer predicciones
            if st.button("Estimar Red Neuronal"):
                # Entrenar el modelo
                model.fit(X, y)

                # Predecir en datos de entrenamiento
                predictions = model.predict(X).flatten()
                y_true = y.flatten()

                # Visualizar la frontera de decisión
                model.plot_decision_boundary(X, y)

                # Calcular la precisión y almacenar en el estado
                accuracy = np.mean(predictions == y_true)
                st.session_state['accuracy'] = accuracy
                st.session_state['weights'] = model.W
                st.session_state['biases'] = model.b
                st.session_state['predictions'] = predictions
                st.session_state['y_true'] = y_true

    if 'accuracy' in st.session_state:
        st.success(f'Precisión: {st.session_state["accuracy"] * 100:.2f}%', icon="✅")

        # Mostrar un panel desplegable con detalles adicionales
        with st.expander("Ver detalles (Pesos, Sesgos, Clases Reales y Predichas)"):
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.write("Pesos (W):")
                st.write(st.session_state['weights'])

            with col2:
                st.write("Sesgos (b):")
                st.write(st.session_state['biases'])
                
            with col3:
                st.write("Clases Predichas:")
                st.write(st.session_state['predictions'])

            with col4:
                st.write("Clases Reales:")
                st.write(st.session_state['y_true'])

if __name__ == "__main__":
    main()
