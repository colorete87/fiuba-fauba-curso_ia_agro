import pygame
import sys
import random
import math
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Inicialización de pygame
pygame.init()

# Tamaño de la ventana
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pipeline Completo de IA - Dataset Iris")

clock = pygame.time.Clock()

# Colores (evitando repetición según preferencia del usuario)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 100, 200)
GREEN = (0, 150, 0)
RED = (200, 0, 0)
ORANGE = (255, 140, 0)
PURPLE = (128, 0, 128)
YELLOW = (255, 215, 0)
GRAY = (200, 200, 200)
LIGHT_GRAY = (240, 240, 240)

# Estados del pipeline
class PipelineState:
    DATA_ACQUISITION = 0
    PREPROCESSING = 1
    MODEL_SELECTION = 2
    TRAINING = 3
    EVALUATION = 4
    DEPLOYMENT = 5

current_state = PipelineState.DATA_ACQUISITION

# Cargar dataset Iris
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# Variables globales para el estado del pipeline
X_train, X_test, y_train, y_test = None, None, None, None
X_scaled_train, X_scaled_test = None, None
scaler = None
selected_model = None
trained_model = None
predictions = None
accuracy = 0.0

###############################################################################
# Clases de UI
###############################################################################
class Button:
    def __init__(self, x, y, width, height, text, color=GRAY, text_color=BLACK):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.font = pygame.font.SysFont(None, 24)
        self.hovered = False

    def draw(self, surface):
        # Cambiar color si está hovered
        current_color = self.color
        if self.hovered:
            current_color = tuple(min(255, c + 30) for c in self.color)
        
        pygame.draw.rect(surface, current_color, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)
        
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

    def handle_mouse_motion(self, pos):
        self.hovered = self.rect.collidepoint(pos)

class InfoPanel:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.font = pygame.font.SysFont(None, 20)
        self.title_font = pygame.font.SysFont(None, 28)

    def draw(self, surface, title, content):
        # Fondo del panel
        pygame.draw.rect(surface, LIGHT_GRAY, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)
        
        # Título
        title_surf = self.title_font.render(title, True, BLACK)
        surface.blit(title_surf, (self.rect.x + 10, self.rect.y + 10))
        
        # Contenido
        y_offset = 50
        for line in content:
            if isinstance(line, str):
                text_surf = self.font.render(line, True, BLACK)
                surface.blit(text_surf, (self.rect.x + 10, self.rect.y + y_offset))
                y_offset += 25
            elif isinstance(line, tuple):  # (text, color)
                text_surf = self.font.render(line[0], True, line[1])
                surface.blit(text_surf, (self.rect.x + 10, self.rect.y + y_offset))
                y_offset += 25

class DataVisualization:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.font = pygame.font.SysFont(None, 16)

    def draw_scatter_plot(self, surface, X_data, y_data, title="Visualización de Datos"):
        # Fondo del gráfico
        pygame.draw.rect(surface, WHITE, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)
        
        # Título
        title_surf = self.font.render(title, True, BLACK)
        surface.blit(title_surf, (self.rect.x + 10, self.rect.y + 5))
        
        # Mapear datos a coordenadas de pantalla
        if len(X_data) > 0:
            # Usar las primeras dos características para la visualización 2D
            x_min, x_max = X_data[:, 0].min(), X_data[:, 0].max()
            y_min, y_max = X_data[:, 1].min(), X_data[:, 1].max()
            
            margin = 20
            plot_width = self.rect.width - 2 * margin
            plot_height = self.rect.height - 40
            
            colors = [BLUE, GREEN, RED]
            
            for i in range(len(X_data)):
                # Obtener las primeras dos características
                x = X_data[i, 0]
                y = X_data[i, 1]
                
                # Mapear a coordenadas de pantalla
                screen_x = self.rect.x + margin + int((x - x_min) / (x_max - x_min) * plot_width)
                screen_y = self.rect.y + 30 + int((y - y_min) / (y_max - y_min) * plot_height)
                
                # Dibujar punto
                color = colors[y_data[i]] if i < len(y_data) else BLACK
                pygame.draw.circle(surface, color, (screen_x, screen_y), 4)

###############################################################################
# Funciones del pipeline
###############################################################################
def load_and_explain_data():
    """Fase 1: Adquisición de datos"""
    global X, y, feature_names, target_names
    
    content = [
        "📊 ADQUISICIÓN DE DATOS",
        "",
        "El dataset Iris es uno de los más famosos en ML:",
        "",
        f"• {len(X)} muestras de flores Iris",
        f"• {len(feature_names)} características por muestra:",
        f"  - {feature_names[0]}",
        f"  - {feature_names[1]}", 
        f"  - {feature_names[2]}",
        f"  - {feature_names[3]}",
        "",
        f"• {len(target_names)} clases de especies:",
        f"  - {target_names[0]}",
        f"  - {target_names[1]}",
        f"  - {target_names[2]}",
        "",
        "💡 En la vida real, estas muestras se obtienen",
        "   midiendo físicamente las flores en el campo",
        "   o laboratorio, registrando cada medida",
        "   cuidadosamente para crear el dataset."
    ]
    
    return content

def preprocess_data():
    """Fase 2: Preprocesamiento"""
    global X, y, X_train, X_test, y_train, y_test, X_scaled_train, X_scaled_test, scaler
    
    # División train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Normalización
    scaler = StandardScaler()
    X_scaled_train = scaler.fit_transform(X_train)
    X_scaled_test = scaler.transform(X_test)
    
    content = [
        "🔧 PREPROCESAMIENTO",
        "",
        "1. DIVISIÓN DE DATOS:",
        f"   • Conjunto de entrenamiento: {len(X_train)} muestras",
        f"   • Conjunto de prueba: {len(X_test)} muestras",
        f"   • Proporción: 70% entrenamiento, 30% prueba",
        "",
        "2. NORMALIZACIÓN:",
        "   • Estandarización: (x - media) / desviación",
        "   • Todas las características en escala similar",
        "   • Evita que una característica domine el modelo",
        "",
        "3. VERIFICACIÓN:",
        f"   • Datos faltantes: {np.isnan(X).sum()}",
        f"   • Valores únicos por clase: {len(np.unique(y))}",
        "   • Distribución balanceada entre clases"
    ]
    
    return content

def select_model():
    """Fase 3: Selección de modelo"""
    global selected_model
    
    models = {
        "Logistic Regression": LogisticRegression(random_state=42),
        "SVM": SVC(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42),
        "Neural Network": MLPClassifier(random_state=42, max_iter=1000)
    }
    
    content = [
        "🤖 SELECCIÓN DE MODELO",
        "",
        "Consideramos 4 algoritmos diferentes:",
        "",
        "1. REGRESIÓN LOGÍSTICA:",
        "   • Modelo lineal simple y rápido",
        "   • Buena para problemas de clasificación",
        "   • Interpretable y estable",
        "",
        "2. MÁQUINA DE VECTORES DE SOPORTE (SVM):",
        "   • Encuentra el mejor hiperplano separador",
        "   • Efectivo en espacios de alta dimensión",
        "   • Robusto a outliers",
        "",
        "3. BOSQUE ALEATORIO:",
        "   • Múltiples árboles de decisión",
        "   • Reduce overfitting",
        "   • Maneja bien datos no lineales",
        "",
        "4. RED NEURONAL:",
        "   • Aprende patrones complejos",
        "   • Muy flexible y potente",
        "   • Puede requerir más datos"
    ]
    
    return content, models

def train_model(model_name, model):
    """Fase 4: Entrenamiento"""
    global trained_model, X_scaled_train, y_train
    
    trained_model = model
    trained_model.fit(X_scaled_train, y_train)
    
    content = [
        f"🏋️ ENTRENAMIENTO - {model_name}",
        "",
        "El modelo aprende los patrones de los datos:",
        "",
        "1. ALGORITMO:",
        f"   • {model_name}",
        f"   • Parámetros ajustados automáticamente",
        "",
        "2. PROCESO:",
        f"   • {len(X_scaled_train)} muestras de entrenamiento",
        "   • Optimización de parámetros internos",
        "   • Minimización de error de predicción",
        "",
        "3. RESULTADO:",
        "   • Modelo entrenado y listo para predecir",
        "   • Patrones aprendidos almacenados",
        "   • Capacidad de generalización desarrollada"
    ]
    
    return content

def evaluate_model():
    """Fase 5: Evaluación"""
    global trained_model, X_scaled_test, y_test, predictions, accuracy
    
    predictions = trained_model.predict(X_scaled_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Calcular métricas adicionales
    report = classification_report(y_test, predictions, target_names=target_names, output_dict=True)
    
    content = [
        "📈 EVALUACIÓN DEL MODELO",
        "",
        f"PRECISIÓN GENERAL: {accuracy:.3f} ({accuracy*100:.1f}%)",
        "",
        "MÉTRICAS POR CLASE:",
    ]
    
    for i, class_name in enumerate(target_names):
        if str(i) in report:
            precision = report[str(i)]['precision']
            recall = report[str(i)]['recall']
            f1 = report[str(i)]['f1-score']
            content.extend([
                f"• {class_name}:",
                f"  - Precisión: {precision:.3f}",
                f"  - Sensibilidad: {recall:.3f}",
                f"  - F1-Score: {f1:.3f}",
                ""
            ])
    
    content.extend([
        "MATRIZ DE CONFUSIÓN:",
        "Muestra predicciones correctas vs incorrectas",
        "Diagonal = predicciones correctas",
        "Fuera de diagonal = errores de clasificación"
    ])
    
    return content

def explain_deployment():
    """Fase 6: Despliegue (explicación teórica)"""
    content = [
        "🚀 DESPLIEGUE DEL MODELO",
        "",
        "Una vez entrenado y evaluado, el modelo debe",
        "desplegarse para uso en producción:",
        "",
        "1. SERIALIZACIÓN:",
        "   • Guardar el modelo entrenado (pickle, joblib)",
        "   • Incluir el preprocesador (scaler)",
        "   • Documentar versiones y dependencias",
        "",
        "2. INFRAESTRUCTURA:",
        "   • Servidor web o API REST",
        "   • Base de datos para nuevas predicciones",
        "   • Monitoreo de rendimiento",
        "",
        "3. INTEGRACIÓN:",
        "   • Aplicación móvil o web",
        "   • Sistema de recomendaciones",
        "   • Herramientas de análisis",
        "",
        "4. MANTENIMIENTO:",
        "   • Reentrenamiento periódico",
        "   • Actualización con nuevos datos",
        "   • Monitoreo de deriva de datos",
        "",
        "💡 En nuestro caso, el modelo podría usarse",
        "   para identificar especies de Iris en campo",
        "   midiendo solo las 4 características."
    ]
    
    return content

###############################################################################
# Inicialización de UI
###############################################################################
# Botones de navegación
btn_prev = Button(50, 50, 100, 40, "← Anterior", BLUE, WHITE)
btn_next = Button(170, 50, 100, 40, "Siguiente →", GREEN, WHITE)
btn_reset = Button(290, 50, 100, 40, "Reiniciar", ORANGE, WHITE)

# Botones de selección de modelo
btn_lr = Button(50, 120, 150, 40, "Regresión Logística", GRAY)
btn_svm = Button(220, 120, 150, 40, "SVM", GRAY)
btn_rf = Button(390, 120, 150, 40, "Bosque Aleatorio", GRAY)
btn_nn = Button(560, 120, 150, 40, "Red Neuronal", GRAY)

# Paneles de información
info_panel = InfoPanel(50, 180, 700, 400)
viz_panel = DataVisualization(770, 180, 380, 400)

# Variables de estado
models = {}
selected_model_name = None

###############################################################################
# Bucle principal
###############################################################################
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
            
            # Navegación
            if btn_prev.is_clicked(pos) and current_state > 0:
                current_state -= 1
            elif btn_next.is_clicked(pos) and current_state < PipelineState.DEPLOYMENT:
                current_state += 1
            elif btn_reset.is_clicked(pos):
                current_state = PipelineState.DATA_ACQUISITION
                selected_model_name = None
                trained_model = None
            
            # Selección de modelo
            if current_state == PipelineState.MODEL_SELECTION:
                if btn_lr.is_clicked(pos):
                    selected_model_name = "Logistic Regression"
                elif btn_svm.is_clicked(pos):
                    selected_model_name = "SVM"
                elif btn_rf.is_clicked(pos):
                    selected_model_name = "Random Forest"
                elif btn_nn.is_clicked(pos):
                    selected_model_name = "Neural Network"
        
        # Manejo de hover
        for btn in [btn_prev, btn_next, btn_reset, btn_lr, btn_svm, btn_rf, btn_nn]:
            btn.handle_mouse_motion(event.pos if event.type == pygame.MOUSEMOTION else (0, 0))

    # Dibujar fondo
    screen.fill(WHITE)
    
    # Dibujar botones de navegación
    btn_prev.draw(screen)
    btn_next.draw(screen)
    btn_reset.draw(screen)
    
    # Dibujar botones de selección de modelo si estamos en esa fase
    if current_state == PipelineState.MODEL_SELECTION:
        btn_lr.draw(screen)
        btn_svm.draw(screen)
        btn_rf.draw(screen)
        btn_nn.draw(screen)
    
    # Mostrar contenido según el estado actual
    if current_state == PipelineState.DATA_ACQUISITION:
        content = load_and_explain_data()
        info_panel.draw(screen, "FASE 1: ADQUISICIÓN DE DATOS", content)
        viz_panel.draw_scatter_plot(screen, X, y, "Dataset Iris Original")
        
    elif current_state == PipelineState.PREPROCESSING:
        content = preprocess_data()
        info_panel.draw(screen, "FASE 2: PREPROCESAMIENTO", content)
        if X_train is not None:
            viz_panel.draw_scatter_plot(screen, X_train, y_train, "Datos de Entrenamiento")
    
    elif current_state == PipelineState.MODEL_SELECTION:
        content, models = select_model()
        info_panel.draw(screen, "FASE 3: SELECCIÓN DE MODELO", content)
        if selected_model_name:
            # Resaltar modelo seleccionado
            highlight_text = f"Modelo seleccionado: {selected_model_name}"
            font = pygame.font.SysFont(None, 24)
            text_surf = font.render(highlight_text, True, GREEN)
            screen.blit(text_surf, (50, 600))
    
    elif current_state == PipelineState.TRAINING:
        if selected_model_name and selected_model_name in models:
            content = train_model(selected_model_name, models[selected_model_name])
            info_panel.draw(screen, "FASE 4: ENTRENAMIENTO", content)
            if X_scaled_train is not None:
                viz_panel.draw_scatter_plot(screen, X_scaled_train, y_train, "Datos Normalizados")
        else:
            content = ["Selecciona un modelo primero en la fase anterior"]
            info_panel.draw(screen, "FASE 4: ENTRENAMIENTO", content)
    
    elif current_state == PipelineState.EVALUATION:
        if trained_model is not None:
            content = evaluate_model()
            info_panel.draw(screen, "FASE 5: EVALUACIÓN", content)
            if X_scaled_test is not None and predictions is not None:
                viz_panel.draw_scatter_plot(screen, X_scaled_test, predictions, "Predicciones del Modelo")
        else:
            content = ["Entrena un modelo primero en la fase anterior"]
            info_panel.draw(screen, "FASE 5: EVALUACIÓN", content)
    
    elif current_state == PipelineState.DEPLOYMENT:
        content = explain_deployment()
        info_panel.draw(screen, "FASE 6: DESPLIEGUE", content)
        # Mostrar resumen final
        if accuracy > 0:
            summary = f"Modelo final: {selected_model_name} - Precisión: {accuracy:.3f}"
            font = pygame.font.SysFont(None, 24)
            text_surf = font.render(summary, True, BLACK)
            screen.blit(text_surf, (50, 600))
    
    # Mostrar estado actual
    state_names = [
        "Adquisición de Datos",
        "Preprocesamiento", 
        "Selección de Modelo",
        "Entrenamiento",
        "Evaluación",
        "Despliegue"
    ]
    
    font = pygame.font.SysFont(None, 20)
    state_text = f"Fase {current_state + 1}/6: {state_names[current_state]}"
    text_surf = font.render(state_text, True, BLACK)
    screen.blit(text_surf, (50, 100))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()
