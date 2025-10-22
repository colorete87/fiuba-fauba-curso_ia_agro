#!/usr/bin/env python3
"""
Ejemplo simple de pipeline de Machine Learning con el dataset Iris
Demuestra las etapas principales: carga, preprocesamiento, entrenamiento y evaluación
"""

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("🌺 Pipeline de Machine Learning - Dataset Iris")
    print("=" * 50)
    
    # 1. CARGA DE DATOS
    print("\n1️⃣ Cargando datos...")
    iris = load_iris()
    X = iris.data  # Características (4 medidas)
    y = iris.target  # Etiquetas (3 especies)
    feature_names = iris.feature_names
    target_names = iris.target_names
    
    print(f"   • {X.shape[0]} muestras")
    print(f"   • {X.shape[1]} características: {', '.join(feature_names)}")
    print(f"   • {len(target_names)} especies: {', '.join(target_names)}")
    
    # Mostrar estadísticas básicas
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [target_names[i] for i in y]
    print(f"\n   📊 Estadísticas básicas:")
    print(df.describe().round(2))
    
    # 2. PREPROCESAMIENTO
    print("\n2️⃣ Preprocesando datos...")
    
    # Separar datos de entrenamiento y evaluación (70% - 30%)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    print(f"   • Datos de entrenamiento: {X_train.shape[0]} muestras")
    print(f"   • Datos de evaluación: {X_test.shape[0]} muestras")
    
    # Normalización (escalar características para que tengan el mismo rango)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("   • Características normalizadas (media=0, desv_est=1)")
    
    # 3. ENTRENAMIENTO DEL MODELO
    print("\n3️⃣ Entrenando modelo...")
    model = SVC(kernel='linear', random_state=42)
    model.fit(X_train_scaled, y_train)
    print("   • Modelo SVM entrenado exitosamente")
    
    # 4. EVALUACIÓN
    print("\n4️⃣ Evaluando modelo...")
    
    # Predicciones
    y_pred = model.predict(X_test_scaled)
    
    # Métricas
    accuracy = accuracy_score(y_test, y_pred)
    print(f"   • Precisión: {accuracy:.3f} ({accuracy*100:.1f}%)")
    
    # Reporte detallado
    print(f"\n   📊 Reporte de clasificación:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    print(f"\n   🔢 Matriz de confusión:")
    print("   " + " " * 12 + "Predicción")
    print("   " + " " * 8 + "setosa versicolor virginica")
    for i, species in enumerate(target_names):
        print(f"   {species:8} {cm[i][0]:7} {cm[i][1]:10} {cm[i][2]:9}")
    
    # 5. VISUALIZACIÓN (opcional)
    print("\n5️⃣ Creando visualización...")
    
    # Crear gráfico de dispersión de las dos primeras características
    plt.figure(figsize=(10, 6))
    
    # Datos de entrenamiento
    plt.subplot(1, 2, 1)
    colors = ['red', 'green', 'blue']
    for i, species in enumerate(target_names):
        mask = y_train == i
        plt.scatter(X_train[mask, 0], X_train[mask, 1], 
                   c=colors[i], label=species, alpha=0.7)
    plt.xlabel('Longitud del sépalo (cm)')
    plt.ylabel('Ancho del sépalo (cm)')
    plt.title('Datos de Entrenamiento')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Datos de prueba con predicciones
    plt.subplot(1, 2, 2)
    for i, species in enumerate(target_names):
        mask = y_test == i
        plt.scatter(X_test[mask, 0], X_test[mask, 1], 
                   c=colors[i], label=species, alpha=0.7, marker='o')
        # Marcar predicciones incorrectas
        wrong_mask = mask & (y_pred != y_test)
        if np.any(wrong_mask):
            plt.scatter(X_test[wrong_mask, 0], X_test[wrong_mask, 1], 
                       c='black', marker='x', s=100, label='Error' if i == 0 else "")
    
    plt.xlabel('Longitud del sépalo (cm)')
    plt.ylabel('Ancho del sépalo (cm)')
    plt.title('Datos de Prueba (X = Error)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('iris_classification.png', dpi=150, bbox_inches='tight')
    print("   • Gráfico guardado como 'iris_classification.png'")
    
    # 6. RESUMEN
    print("\n✅ RESUMEN DEL PIPELINE")
    print("=" * 30)
    print(f"• Dataset: {X.shape[0]} muestras de {len(target_names)} especies")
    print(f"• Características: {X.shape[1]} medidas morfológicas")
    print(f"• Modelo: SVM con kernel lineal")
    print(f"• Precisión final: {accuracy:.1%}")
    print(f"• El modelo puede clasificar correctamente {accuracy:.1%} de las flores iris")
    
    print(f"\n🎯 Este ejemplo demuestra las 6 etapas principales:")
    print("   1. Carga de datos")
    print("   2. Preprocesamiento (separación + normalización)")
    print("   3. Entrenamiento del modelo")
    print("   4. Evaluación con métricas")
    print("   5. Visualización de resultados")
    print("   6. Interpretación de resultados")

if __name__ == "__main__":
    main()
