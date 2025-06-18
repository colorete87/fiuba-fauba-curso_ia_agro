import pygame
import sys
import random
import math

# Inicialización de pygame
pygame.init()

# Tamaño de la ventana: se aumenta la altura para disponer de más elementos de UI
WIDTH, HEIGHT = 800, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulación de MLP y Gradiente Descendente")

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
# Clases de UI: Slider, Button y TextBox
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
                                        self.handle_radius * 2, self.handle_radius * 2)
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
            if self.rect.collidepoint(event.pos):
                self.active = True
            else:
                self.active = False
        if event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.unicode.isdigit():
                self.text += event.unicode

    def draw(self, surface):
        pygame.draw.rect(surface, WHITE, self.rect)
        borde = BLACK if not self.active else RED
        pygame.draw.rect(surface, borde, self.rect, 2)
        display_text = self.text
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
# Modelo MLP de 1 entrada, 1 capa oculta con 2 neuronas (ReLU) y 1 salida
###############################################################################
# Inicializamos los parámetros de forma aleatoria
w1 = random.uniform(-1, 1)
b1 = random.uniform(-1, 1)
w2 = random.uniform(-1, 1)
b2 = random.uniform(-1, 1)
v1 = random.uniform(-1, 1)
v2 = random.uniform(-1, 1)
c  = random.uniform(-1, 1)

def relu(x):
    #return x if x > 0 else 0
    return math.atan(x)

def nn_forward(x):
    global w1, b1, w2, b2, v1, v2, c
    z1 = w1 * x + b1
    h1 = relu(z1)
    z2 = w2 * x + b2
    h2 = relu(z2)
    y_pred = v1 * h1 + v2 * h2 + c
    return y_pred

def compute_error_nn(points):
    mse = 0.0
    n = len(points)
    for (x, y) in points:
        pred = nn_forward(x)
        mse += (pred - y) ** 2
    return mse / n if n > 0 else 0

#def gradient_descent_step_nn(points, lr):
#    global w1, b1, w2, b2, v1, v2, c
#    n = len(points)
#    sum_dw1 = 0.0
#    sum_db1 = 0.0
#    sum_dw2 = 0.0
#    sum_db2 = 0.0
#    sum_dv1 = 0.0
#    sum_dv2 = 0.0
#    sum_dc  = 0.0
#    for (x, y) in points:
#        # Forward
#        z1 = w1 * x + b1
#        h1 = relu(z1)
#        z2 = w2 * x + b2
#        h2 = relu(z2)
#        y_pred = v1 * h1 + v2 * h2 + c
#        delta = 2 * (y_pred - y)  # derivada del error cuadrático
#        # Capa de salida:
#        sum_dv1 += delta * h1
#        sum_dv2 += delta * h2
#        sum_dc  += delta
#        # Capa oculta, neurona 1:
#        dReLU1 = 1 if z1 > 0 else 0
#        d_hidden1 = delta * v1 * dReLU1
#        sum_dw1 += d_hidden1 * x
#        sum_db1 += d_hidden1
#        # Capa oculta, neurona 2:
#        dReLU2 = 1 if z2 > 0 else 0
#        d_hidden2 = delta * v2 * dReLU2
#        sum_dw2 += d_hidden2 * x
#        sum_db2 += d_hidden2
#    # Promediar sobre n muestras:
#    sum_dw1 /= n
#    sum_db1 /= n
#    sum_dw2 /= n
#    sum_db2 /= n
#    sum_dv1 /= n
#    sum_dv2 /= n
#    sum_dc  /= n
#    # Actualizar parámetros
#    w1 = w1 - lr * sum_dw1 * 1
#    b1 = b1 - lr * sum_db1 * 1
#    w2 = w2 - lr * sum_dw2 * 1
#    b2 = b2 - lr * sum_db2 * 1
#    v1 = v1 - lr * sum_dv1 * 1
#    v2 = v2 - lr * sum_dv2 * 1
#    c  = c  - lr * sum_dc
def gradient_descent_step_nn(points, lr):
    global w1, b1, w2, b2, v1, v2, c
    n = len(points)
    sum_dw1 = 0.0
    sum_db1 = 0.0
    sum_dw2 = 0.0
    sum_db2 = 0.0
    sum_dv1 = 0.0
    sum_dv2 = 0.0
    sum_dc  = 0.0
    for (x, _) in points:
        # Calculamos el valor objetivo usando atan(x/100)
        target = math.atan(x/100)
        # Propagación hacia adelante
        z1 = w1 * x + b1
        h1 = relu(z1)
        z2 = w2 * x + b2
        h2 = relu(z2)
        y_pred = v1 * h1 + v2 * h2 + c
        # Gradiente de la pérdida (error cuadrático) respecto a la salida:
        delta = 2 * (y_pred - target)
        # Gradientes de la capa de salida
        sum_dv1 += delta * h1
        sum_dv2 += delta * h2
        sum_dc  += delta
        # Gradientes para la capa oculta, neurona 1
        dReLU1 = 1 if z1 > 0 else 0
        d_hidden1 = delta * v1 * dReLU1
        sum_dw1 += d_hidden1 * x
        sum_db1 += d_hidden1
        # Gradientes para la capa oculta, neurona 2
        dReLU2 = 1 if z2 > 0 else 0
        d_hidden2 = delta * v2 * dReLU2
        sum_dw2 += d_hidden2 * x
        sum_db2 += d_hidden2
    # Promediar sobre n muestras
    sum_dw1 /= n
    sum_db1 /= n
    sum_dw2 /= n
    sum_db2 /= n
    sum_dv1 /= n
    sum_dv2 /= n
    sum_dc  /= n
    # Actualizar parámetros
    w1 = w1 - lr * sum_dw1 * 1000
    b1 = b1 - lr * sum_db1 * 1000
    w2 = w2 - lr * sum_dw2 * 1000
    b2 = b2 - lr * sum_db2 * 1000
    v1 = v1 - lr * sum_dv1 * 1000
    v2 = v2 - lr * sum_dv2 * 1000
    c  = c  - lr * sum_dc  * 1000


###############################################################################
# Generación de puntos de datos
###############################################################################
def generate_points(dataset_type, n=100):
    points = []
    if dataset_type == "lineal":
        # Ejemplo: y = 0.7*x + 20 + ruido
        for _ in range(n):
            x = random.uniform(0, 100)
            true_y = 0.7 * x + 20
            noise = random.gauss(0, 5)
            y = true_y + noise
            points.append((x, y))
    elif dataset_type == "cuadratico":
        # Ejemplo: y = 0.03*(x-30)^2 + 1 + ruido
        for _ in range(n):
            x = random.uniform(0, 100)
            true_y = 0.7 * x + 20 + 6 * math.sin(0.3 * x)
            noise = random.gauss(0, 5)
            y = true_y + noise
            points.append((x, y))
    return points

def data_to_screen(x, y):
    # Suponemos datos en el rango [0, 100] para x e y
    x_screen = PLOT_LEFT + (x / 100) * PLOT_WIDTH
    y_screen = PLOT_TOP + PLOT_HEIGHT - (y / 100) * PLOT_HEIGHT
    return int(x_screen), int(y_screen)

def draw_model_curve(surface):
    """Dibuja la función aprendida por la red neuronal en el rango de x de 0 a 100."""
    puntos = []
    # Usamos una resolución de 101 puntos
    for i in range(101):
        x = i
        y = nn_forward(x)
        puntos.append(data_to_screen(x, y))
    if len(puntos) > 1:
        pygame.draw.lines(surface, RED, False, puntos, 2)

###############################################################################
# Estado inicial, elementos de UI y variables globales
###############################################################################
dataset_type = "lineal"
n_points = 50
points = generate_points(dataset_type, n_points)

# Botones (fila superior de UI)
button_dataset = Button(50, 570, 150, 40, "Cambiar Conjunto")
button_step = Button(220, 570, 150, 40, "Paso GD")
button_play = Button(400, 570, 150, 40, "Play")

# Sliders para los parámetros del MLP, organizados en 4 filas (2 sliders por fila)
slider_w1 = Slider( 50, 630, 200, 20, -100, 100, w1, label="w1")
slider_b1 = Slider(300, 630, 200, 20, -1000, 1000, b1, label="b1")
slider_w2 = Slider( 50, 660, 200, 20, -100, 100, w2, label="w2")
slider_b2 = Slider(300, 660, 200, 20, -1000, 1000, b2, label="b2")
slider_v1 = Slider( 50, 690, 200, 20, -100, 100, v1, label="v1")
slider_v2 = Slider(300, 690, 200, 20, -100, 100, v2, label="v2")
slider_c  = Slider( 50, 720, 200, 20, -100, 100, c, label="c")
slider_lr = Slider(550, 630, 200, 20, 0.000001, 0.001, 0.0001, label="Learning Rate")

# Cuadros de texto para configurar steps por segundo y número de puntos
textbox_steps = TextBox(50, 760, 100, 40, text="100")
textbox_n_points = TextBox(300, 760, 100, 40, text="50")
font_ui = pygame.font.SysFont(None, 24)

# Variables para modo automático (Play/Pause)
play_mode = False
accumulated_time = 0.0

###############################################################################
# Bucle principal
###############################################################################
running = True
while running:
    dt = clock.tick(60) / 1000.0  # dt en segundos

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        # Salir presionando la tecla "q"
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

        # Manejo de eventos de UI para botones y controles
        button_dataset_clicked = False
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
            if button_dataset.is_clicked(pos):
                dataset_type = "cuadratico" if dataset_type == "lineal" else "lineal"
                n_points = textbox_n_points.get_value()
                points = generate_points(dataset_type, n_points)
                button_dataset_clicked = True
            if button_step.is_clicked(pos):
                current_lr = slider_lr.value
                gradient_descent_step_nn(points, current_lr)
                # Después de GD, se actualizan los sliders para reflejar los nuevos parámetros
                slider_w1.set_value(w1)
                slider_b1.set_value(b1)
                slider_w2.set_value(w2)
                slider_b2.set_value(b2)
                slider_v1.set_value(v1)
                slider_v2.set_value(v2)
                slider_c.set_value(c)
            if button_play.is_clicked(pos):
                play_mode = not play_mode
                button_play.text = "Pause" if play_mode else "Play"

        # Manejo de eventos para los sliders de parámetros y learning rate
        slider_w1.handle_event(event)
        slider_b1.handle_event(event)
        slider_w2.handle_event(event)
        slider_b2.handle_event(event)
        slider_v1.handle_event(event)
        slider_v2.handle_event(event)
        slider_c.handle_event(event)
        slider_lr.handle_event(event)
        # Manejo de eventos para los textboxes
        textbox_steps.handle_event(event)
        textbox_n_points.handle_event(event)

    # Si se está en modo Play, se ejecuta GD automáticamente
    if play_mode:
        accumulated_time += dt
        steps_per_sec = textbox_steps.get_value()
        while accumulated_time >= 1.0 / steps_per_sec:
            current_lr = slider_lr.value
            gradient_descent_step_nn(points, current_lr)
            accumulated_time -= 1.0 / steps_per_sec
            # Actualizar sliders con los nuevos valores
            slider_w1.set_value(w1)
            slider_b1.set_value(b1)
            slider_w2.set_value(w2)
            slider_b2.set_value(b2)
            slider_v1.set_value(v1)
            slider_v2.set_value(v2)
            slider_c.set_value(c)

    # Actualizar globalmente los parámetros con lo que indique el usuario (si se arrastran los sliders)
    w1 = slider_w1.value
    b1 = slider_b1.value
    w2 = slider_w2.value
    b2 = slider_b2.value
    v1 = slider_v1.value
    v2 = slider_v2.value
    c  = slider_c.value

    # Dibujar fondo y área de gráfico
    screen.fill(WHITE)
    pygame.draw.rect(screen, BLACK, PLOT_RECT, 2)

    # Dibujar los puntos de datos
    for (x, y) in points:
        pos = data_to_screen(x, y)
        pygame.draw.circle(screen, BLUE, pos, 3)

    # Dibujar la curva que representa la función aprendida por la red
    draw_model_curve(screen)

    # Dibujar elementos de UI: botones
    button_dataset.draw(screen)
    button_step.draw(screen)
    button_play.draw(screen)

    # Dibujar sliders para los parámetros y learning rate
    slider_w1.draw(screen)
    slider_b1.draw(screen)
    slider_w2.draw(screen)
    slider_b2.draw(screen)
    slider_v1.draw(screen)
    slider_v2.draw(screen)
    slider_c.draw(screen)
    slider_lr.draw(screen)

    # Dibujar cuadros de texto y sus etiquetas
    textbox_steps.draw(screen)
    textbox_n_points.draw(screen)
    steps_label = font_ui.render("Steps/s", True, BLACK)
    screen.blit(steps_label, (textbox_steps.rect.x + textbox_steps.rect.width + 10, textbox_steps.rect.y + 10))
    n_points_label = font_ui.render("Points", True, BLACK)
    screen.blit(n_points_label, (textbox_n_points.rect.x + textbox_n_points.rect.width + 10, textbox_n_points.rect.y + 10))

    # Mostrar el error (MSE) calculado sobre el dataset
    error_value = compute_error_nn(points)
    error_text = font_ui.render(f"Error: {error_value:.2f}", True, BLACK)
    screen.blit(error_text, (PLOT_LEFT + PLOT_WIDTH + 20, PLOT_TOP))

    pygame.display.flip()

pygame.quit()
sys.exit()

