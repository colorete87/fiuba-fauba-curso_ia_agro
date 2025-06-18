import pygame
import sys
import random
import math

# Inicialización de pygame
pygame.init()

# Tamaño de la ventana
WIDTH, HEIGHT = 800, 750
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulación de Regresión Lineal y Gradiente Descendente")

clock = pygame.time.Clock()

# Área del gráfico
PLOT_LEFT, PLOT_TOP = 50, 50
PLOT_WIDTH, PLOT_HEIGHT = 600, 500
PLOT_RECT = pygame.Rect(PLOT_LEFT, PLOT_TOP, PLOT_WIDTH, PLOT_HEIGHT)

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE  = (0, 0, 255)
RED   = (255, 0, 0)
GRAY  = (200, 200, 200)

###############################################################################
# Clases para elementos de UI: Slider, Button y TextBox
###############################################################################
class Slider:
    def __init__(self, x, y, width, height, min_val, max_val, init_val, label=""):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self._value = init_val
        self.label = label
        self.handle_radius = height // 2
        self.handle_x = self.get_handle_x()
        self.dragging = False
        self.font = pygame.font.SysFont(None, 20)

    def get_handle_x(self):
        ratio = (self._value - self.min_val) / (self.max_val - self.min_val)
        return self.rect.x + int(ratio * self.rect.width)

    def set_value(self, new_val):
        self._value = max(min(new_val, self.max_val), self.min_val)
        self.handle_x = self.get_handle_x()

    @property
    def value(self):
        return self._value

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            handle_rect = pygame.Rect(self.handle_x - self.handle_radius, self.rect.centery - self.handle_radius,
                                      self.handle_radius*2, self.handle_radius*2)
            if handle_rect.collidepoint(event.pos):
                self.dragging = True
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging:
                x = event.pos[0]
                x = max(self.rect.x, min(x, self.rect.x + self.rect.width))
                ratio = (x - self.rect.x) / self.rect.width
                self._value = self.min_val + ratio * (self.max_val - self.min_val)
                self.handle_x = x

    def draw(self, surface):
        pygame.draw.rect(surface, GRAY, self.rect)
        pygame.draw.circle(surface, BLACK, (self.handle_x, self.rect.centery), self.handle_radius)
        label_surf = self.font.render(f"{self.label}: {self._value:.6f}", True, BLACK)
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


class TextBox:
    def __init__(self, x, y, width, height, text="1"):
        self.rect = pygame.Rect(x, y, width, height)
        self.active = False
        self.text = text
        self.font = pygame.font.SysFont(None, 24)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            # Activa el textbox si se hace clic dentro, de lo contrario lo desactiva
            if self.rect.collidepoint(event.pos):
                self.active = True
            else:
                self.active = False
        if event.type == pygame.KEYDOWN and self.active:
            # Permitir terminar la edición con Enter
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            # Se permiten solo dígitos ya que es un entero (pasos por segundo)
            elif event.unicode.isdigit():
                self.text += event.unicode

    def draw(self, surface):
        # Dibuja el rectángulo del textbox
        pygame.draw.rect(surface, WHITE, self.rect)
        borde = BLACK if not self.active else RED
        pygame.draw.rect(surface, borde, self.rect, 2)
        display_text = self.text
        # Agrega un cursor intermitente si está activo
        if self.active:
            if (pygame.time.get_ticks() // 500) % 2 == 0:
                display_text += "|"
        text_surf = self.font.render(display_text, True, BLACK)
        surface.blit(text_surf, (self.rect.x + 5, self.rect.y + 5))

    def get_value(self):
        try:
            return max(1, int(self.text))
        except:
            return 1

###############################################################################
# Funciones para generación de datos y procesamiento
###############################################################################
def generate_points(dataset_type, n=100):
    points = []
    if dataset_type == "lineal":
        for _ in range(n):
            x = random.uniform(0, 100)
            true_y = 0.7 * x + 20
            noise = random.gauss(0, 5)
            y = true_y + noise
            points.append((x, y))
    elif dataset_type == "cuadratico":
        for _ in range(n):
            x = random.uniform(0, 100)
            true_y = 0.015 * (x - 30) ** 2 + 1
            noise = random.gauss(0, 5)
            y = true_y + noise
            points.append((x, y))
    return points

def compute_error(points, a, b):
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
    x_screen = PLOT_LEFT + (x / 100) * PLOT_WIDTH
    y_screen = PLOT_TOP + PLOT_HEIGHT - (y / 100) * PLOT_HEIGHT
    return int(x_screen), int(y_screen)

###############################################################################
# Estado inicial, elementos de UI y variables globales
###############################################################################
dataset_type = "lineal"
points = generate_points(dataset_type)

# Parámetros del modelo
a = 0.0
b = 0.0

# Elementos de UI
button_dataset = Button(50, 570, 150, 40, "Cambiar Conjunto")
button_step = Button(220, 570, 150, 40, "Paso GD")
button_play = Button(400, 570, 150, 40, "Play")  # Botón Play/Pause

slider_a = Slider(50, 630, 200, 20, -10, 10, a, label="a")
slider_b = Slider(300, 630, 200, 20, -50, 50, b, label="b")
slider_lr = Slider(550, 630, 200, 20, 0.000001, 0.001, 0.00001, label="Learning Rate")

# Cuadro de texto para configurar n_steps (pasos por segundo)
textbox_steps = TextBox(50, 680, 100, 40, text="4")
textbox_n_points = TextBox(300, 680, 100, 40, text="50")
font = pygame.font.SysFont(None, 24)

# Variables para el modo automático
play_mode = False
accumulated_time = 0.0

###############################################################################
# Bucle principal
###############################################################################
running = True
while running:
    dt = clock.tick(60) / 1000.0  # Tiempo en segundos entre frames

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Salir con la tecla "q"
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

        # Manejo de eventos para UI
        slider_a.handle_event(event)
        slider_b.handle_event(event)
        slider_lr.handle_event(event)
        textbox_steps.handle_event(event)
        textbox_n_points.handle_event(event)

        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
            if button_dataset.is_clicked(pos):
                dataset_type = "cuadratico" if dataset_type == "lineal" else "lineal"
                n_points = textbox_n_points.get_value()
                points = generate_points(dataset_type, n_points)
            if button_step.is_clicked(pos):
                current_lr = slider_lr.value
                new_a, new_b = gradient_descent_step(points, slider_a.value, slider_b.value, current_lr)
                slider_a.set_value(new_a)
                slider_b.set_value(new_b)
            if button_play.is_clicked(pos):
                play_mode = not play_mode
                button_play.text = "Pause" if play_mode else "Play"

    # Actualización automática de GD en modo play
    if play_mode:
        accumulated_time += dt
        steps_per_sec = textbox_steps.get_value()
        n_points = textbox_n_points.get_value()
        while accumulated_time >= 1.0 / steps_per_sec:
            current_lr = slider_lr.value
            new_a, new_b = gradient_descent_step(points, slider_a.value, slider_b.value, current_lr)
            slider_a.set_value(new_a)
            slider_b.set_value(new_b)
            accumulated_time -= 1.0 / steps_per_sec

    # Actualizar variables globales para graficar
    a = slider_a.value
    b = slider_b.value

    # Dibujar fondo y área de gráfico
    screen.fill(WHITE)
    pygame.draw.rect(screen, BLACK, PLOT_RECT, 2)

    # Dibujar puntos
    for (x, y) in points:
        pos = data_to_screen(x, y)
        pygame.draw.circle(screen, BLUE, pos, 3)

    # Dibujar la línea de regresión (para x en [0, 100])
    x1_data = 0
    x2_data = 100
    y1_data = a * x1_data + b
    y2_data = a * x2_data + b
    p1 = data_to_screen(x1_data, y1_data)
    p2 = data_to_screen(x2_data, y2_data)
    pygame.draw.line(screen, RED, p1, p2, 2)

    # Dibujar elementos de UI: sliders, botones y textbox
    slider_a.draw(screen)
    slider_b.draw(screen)
    slider_lr.draw(screen)
    button_dataset.draw(screen)
    button_step.draw(screen)
    button_play.draw(screen)
    textbox_steps.draw(screen)
    textbox_n_points.draw(screen)
    steps_label = font.render("Steps/s", True, BLACK)
    screen.blit(steps_label, (textbox_steps.rect.x + textbox_steps.rect.width + 10, textbox_steps.rect.y + 10))
    n_points_label = font.render("Points", True, BLACK)
    screen.blit(n_points_label, (textbox_n_points.rect.x + textbox_n_points.rect.width + 10, textbox_n_points.rect.y + 10))

    # Mostrar el error (MSE) en la parte superior derecha del plot
    error_value = compute_error(points, a, b)
    error_text = font.render(f"MSE: {error_value:.2f}", True, BLACK)
    screen.blit(error_text, (PLOT_LEFT + PLOT_WIDTH + 20, PLOT_TOP))

    pygame.display.flip()

pygame.quit()
sys.exit()

