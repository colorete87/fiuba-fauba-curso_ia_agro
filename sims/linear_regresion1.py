import pygame
import sys
import random
import math

# Inicialización de pygame
pygame.init()

# Tamaño de la ventana (ancho x alto)
WIDTH, HEIGHT = 800, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulación de Regresión Lineal y Gradiente Descendente")

clock = pygame.time.Clock()

# Definir el área de gráfico (margen superior izquierdo y dimensiones)
PLOT_LEFT, PLOT_TOP = 50, 50
PLOT_WIDTH, PLOT_HEIGHT = 600, 500
PLOT_RECT = pygame.Rect(PLOT_LEFT, PLOT_TOP, PLOT_WIDTH, PLOT_HEIGHT)

# Colores básicos
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE  = (0, 0, 255)
RED   = (255, 0, 0)
GRAY  = (200, 200, 200)

###############################################################################
# Clases para elementos de UI: Slider y Button
###############################################################################
class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, init_val, label=""):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self._value = init_val
        self.label = label
        self.handle_radius = height // 2
        # Posición horizontal del handle
        self.handle_x = self.get_handle_x()
        self.dragging = False
        self.font = pygame.font.SysFont(None, 20)

    def get_handle_x(self):
        # Mapea el valor actual al rango del slider (posición en píxeles)
        ratio = (self._value - self.min_val) / (self.max_val - self.min_val)
        return self.rect.x + int(ratio * self.rect.width)

    def set_value(self, new_val):
        # Fija el valor y actualiza la posición del handle
        self._value = max(min(new_val, self.max_val), self.min_val)
        self.handle_x = self.get_handle_x()

    @property
    def value(self):
        return self._value

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if pygame.Rect(self.handle_x - self.handle_radius, self.rect.centery - self.handle_radius,
                           self.handle_radius*2, self.handle_radius*2).collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                # Actualiza el valor según la posición del mouse
                x = event.pos[0]
                # Limitar el movimiento al ancho del slider
                x = max(self.rect.x, min(x, self.rect.x + self.rect.width))
                ratio = (x - self.rect.x) / self.rect.width
                self._value = self.min_val + ratio * (self.max_val - self.min_val)
                self.handle_x = x

    def draw(self, surface):
        # Dibuja la pista
        pygame.draw.rect(surface, GRAY, self.rect)
        # Dibuja el handle (círculo)
        pygame.draw.circle(surface, BLACK, (self.handle_x, self.rect.centery), self.handle_radius)
        # Dibuja etiqueta y valor
        label_surf = self.font.render(f"{self.label}: {self._value:.5f}", True, BLACK)
        surface.blit(label_surf, (self.rect.x, self.rect.y - 20))


class Button:
    def __init__(self, x, y, width, height, text):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.font = pygame.font.SysFont(None, 24)

    def draw(self, surface):
        pygame.draw.rect(surface, GRAY, self.rect)
        pygame.draw.rect(surface, BLACK, self.rect, 2)
        text_surf = self.font.render(self.text, True, BLACK)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

    def is_clicked(self, pos):
        return self.rect.collidepoint(pos)

###############################################################################
# Funciones para generación de datos y procesamiento
###############################################################################
def generate_points(dataset_type, n=100):
    """
    Genera una lista de puntos (x, y) según el tipo de conjunto:
      - "lineal": puntos generados a partir de una función lineal con error
      - "cuadratico": puntos generados a partir de una función cuadrática con error
    """
    points = []
    if dataset_type == "lineal":
        # Ejemplo: f(x) = 0.8*x + 10 con error gaussiano
        for _ in range(n):
            x = random.uniform(0, 100)
            true_y = 0.7 * x + 20
            noise = random.gauss(0, 5)
            y = true_y + noise
            points.append((x, y))
    elif dataset_type == "cuadratico":
        # Ejemplo: f(x) = 0.05*x^2 + 2*x + 5 con error
        for _ in range(n):
            x = random.uniform(0, 100)
            true_y = 0.03 * (x-30)**2 + 1
            noise = random.gauss(0, 5)
            y = true_y + noise
            points.append((x, y))
    return points

def compute_error(points, a, b):
    """
    Calcula el error cuadrático medio (MSE) para la función f(x)= a*x + b
    """
    mse = 0
    n = len(points)
    for (x, y) in points:
        pred = a * x + b
        mse += (pred - y) ** 2
    return mse / n if n > 0 else 0

def gradient_descent_step(points, a, b, lr):
    """
    Realiza un paso de gradiente descendiente para ajustar a y b,
    utilizando la derivada del error MSE.
    """
    n = len(points)
    grad_a = 0
    grad_b = 0
    for (x, y) in points:
        pred = a * x + b
        error = pred - y
        grad_a += (2/n) * x * error
        grad_b += (2/n) * error
    a_new = a - lr * grad_a
    b_new = b - lr * grad_b * 1000
    return a_new, b_new

def data_to_screen(x, y):
    """
    Mapea un par (x, y) en coordenadas de datos (asumidas en [0,100])
    a coordenadas de pantalla en el área de gráfico.
    """
    # Mapeo lineal para x:
    x_screen = PLOT_LEFT + (x / 100) * PLOT_WIDTH
    # Para y: en datos asumimos 0 (abajo) a 100 (arriba) y en pantalla y=0 es la parte superior.
    y_screen = PLOT_TOP + PLOT_HEIGHT - (y / 100) * PLOT_HEIGHT
    return int(x_screen), int(y_screen)

###############################################################################
# Estado inicial, elementos de UI y variables globales
###############################################################################
# Tipo de conjunto ("lineal" o "cuadratico")
dataset_type = "lineal"
points = generate_points(dataset_type)

# Parámetros iniciales del modelo de regresión
a = 0.0
b = 0.0

# Configuración de sliders y botones (ubicados en área de UI por debajo del plot)
button_dataset = Button(50, 570, 150, 40, "Cambiar Conjunto")
button_step = Button(220, 570, 150, 40, "Paso GD")

slider_a = Slider(50, 630, 200, 20, -10, 10, a, label="a")
slider_b = Slider(300, 630, 200, 20, -50, 50, b, label="b")
slider_lr = Slider(550, 630, 200, 20, 0.00001, 0.0005, 0.00003, label="Learning Rate")

###############################################################################
# Bucle principal
###############################################################################
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Pasar eventos a los sliders
        slider_a.handle_event(event)
        slider_b.handle_event(event)
        slider_lr.handle_event(event)

        # Eventos de clic para los botones
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
            if button_dataset.is_clicked(pos):
                # Cambia entre conjuntos de puntos
                dataset_type = "cuadratico" if dataset_type == "lineal" else "lineal"
                points = generate_points(dataset_type)
            if button_step.is_clicked(pos):
                # Al hacer clic se realiza un paso de gradiente descendiente
                current_lr = slider_lr.value
                # Se toma el valor actual de los sliders para a y b
                new_a, new_b = gradient_descent_step(points, slider_a.value, slider_b.value, current_lr)
                # Se actualizan los sliders para reflejar el nuevo valor
                slider_a.set_value(new_a)
                slider_b.set_value(new_b)

    # También se actualizan las variables globales a y b según los sliders (para graficar)
    a = slider_a.value
    b = slider_b.value

    # Dibujo del fondo y área de gráfico
    screen.fill(WHITE)
    pygame.draw.rect(screen, BLACK, PLOT_RECT, 2)

    # Dibujo de los puntos
    for (x, y) in points:
        pos = data_to_screen(x, y)
        pygame.draw.circle(screen, BLUE, pos, 3)

    # Dibujo de la línea de regresión: se grafica de x=0 a x=100 (en coordenadas de datos)
    x1_data = 0
    x2_data = 100
    y1_data = a * x1_data + b
    y2_data = a * x2_data + b
    p1 = data_to_screen(x1_data, y1_data)
    p2 = data_to_screen(x2_data, y2_data)
    pygame.draw.line(screen, RED, p1, p2, 2)

    # Dibujo de los elementos de UI: sliders y botones
    slider_a.draw(screen)
    slider_b.draw(screen)
    slider_lr.draw(screen)
    button_dataset.draw(screen)
    button_step.draw(screen)

    # Mostrar error (MSE) en la esquina superior derecha del plot
    error_value = compute_error(points, a, b)
    font = pygame.font.SysFont(None, 24)
    error_text = font.render(f"Error: {error_value:.2f}", True, BLACK)
    screen.blit(error_text, (PLOT_LEFT + PLOT_WIDTH + 20, PLOT_TOP))

    # Actualiza la pantalla
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()

