import pygame
import sys
import random
import math

# Inicialización de pygame
pygame.init()

# Tamaño de la ventana: se aumenta la altura para disponer de más elementos de UI
WIDTH, HEIGHT = 800, 950
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulación de MLP (5 neuronas ocultas) y Gradiente Descendente")

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
# Modelo MLP de 1 entrada, 1 capa oculta con 5 neuronas (arctan) y 1 salida
###############################################################################
# Parámetros para la capa oculta (5 neuronas) y de salida
w1 = random.uniform(-10, 10)
b1 = random.uniform(-10, 10)
w2 = random.uniform(-10, 10)
b2 = random.uniform(-10, 10)
w3 = random.uniform(-10, 10)
b3 = random.uniform(-10, 10)
w4 = random.uniform(-10, 10)
b4 = random.uniform(-10, 10)
w5 = random.uniform(-10, 10)
b5 = random.uniform(-10, 10)
v1 = random.uniform(-10, 10)
v2 = random.uniform(-10, 10)
v3 = random.uniform(-10, 10)
v4 = random.uniform(-10, 10)
v5 = random.uniform(-10, 10)
c  = random.uniform(-10, 10)

def activation(x):
    return math.atan(x)

def d_activation(x):
    return 1 / (1 + x * x)

def nn_forward(x):
    global w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, v1, v2, v3, v4, v5, c
    # Capa oculta: 5 neuronas
    z1 = w1 * x + b1; h1 = activation(z1)
    z2 = w2 * x + b2; h2 = activation(z2)
    z3 = w3 * x + b3; h3 = activation(z3)
    z4 = w4 * x + b4; h4 = activation(z4)
    z5 = w5 * x + b5; h5 = activation(z5)
    # Capa de salida:
    y_pred = v1 * h1 + v2 * h2 + v3 * h3 + v4 * h4 + v5 * h5 + c
    return y_pred

def compute_error_nn(points):
    mse = 0.0
    n = len(points)
    for (x, y) in points:
        pred = nn_forward(x)
        mse += (pred - y) ** 2
    return mse / n if n > 0 else 0

def gradient_descent_step_nn(points, lr):
    global w1, b1, w2, b2, w3, b3, w4, b4, w5, b5, v1, v2, v3, v4, v5, c
    n = len(points)
    # Initialize accumulators
    sum_dw1 = 0.0; sum_db1 = 0.0
    sum_dw2 = 0.0; sum_db2 = 0.0
    sum_dw3 = 0.0; sum_db3 = 0.0
    sum_dw4 = 0.0; sum_db4 = 0.0
    sum_dw5 = 0.0; sum_db5 = 0.0
    sum_dv1 = 0.0; sum_dv2 = 0.0; sum_dv3 = 0.0; sum_dv4 = 0.0; sum_dv5 = 0.0
    sum_dc  = 0.0
    for (x, y) in points:
        # Forward pass
        z1 = w1 * x + b1; h1 = activation(z1)
        z2 = w2 * x + b2; h2 = activation(z2)
        z3 = w3 * x + b3; h3 = activation(z3)
        z4 = w4 * x + b4; h4 = activation(z4)
        z5 = w5 * x + b5; h5 = activation(z5)
        y_pred = v1 * h1 + v2 * h2 + v3 * h3 + v4 * h4 + v5 * h5 + c
        
        # Compute gradient delta for squared error loss
        delta = 2 * (y_pred - y)
        
        # Output layer gradients:
        sum_dv1 += delta * h1
        sum_dv2 += delta * h2
        sum_dv3 += delta * h3
        sum_dv4 += delta * h4
        sum_dv5 += delta * h5
        sum_dc  += delta
        
        # Hidden layer gradients:
        dAct1 = d_activation(z1)
        d_hidden1 = delta * v1 * dAct1
        sum_dw1 += d_hidden1 * x
        sum_db1 += d_hidden1
        
        dAct2 = d_activation(z2)
        d_hidden2 = delta * v2 * dAct2
        sum_dw2 += d_hidden2 * x
        sum_db2 += d_hidden2
        
        dAct3 = d_activation(z3)
        d_hidden3 = delta * v3 * dAct3
        sum_dw3 += d_hidden3 * x
        sum_db3 += d_hidden3
        
        dAct4 = d_activation(z4)
        d_hidden4 = delta * v4 * dAct4
        sum_dw4 += d_hidden4 * x
        sum_db4 += d_hidden4
        
        dAct5 = d_activation(z5)
        d_hidden5 = delta * v5 * dAct5
        sum_dw5 += d_hidden5 * x
        sum_db5 += d_hidden5

    # Average gradients
    sum_dw1 /= n; sum_db1 /= n
    sum_dw2 /= n; sum_db2 /= n
    sum_dw3 /= n; sum_db3 /= n
    sum_dw4 /= n; sum_db4 /= n
    sum_dw5 /= n; sum_db5 /= n
    sum_dv1 /= n; sum_dv2 /= n; sum_dv3 /= n; sum_dv4 /= n; sum_dv5 /= n
    sum_dc  /= n

    # Update parameters
    w1 = w1 - lr * sum_dw1
    b1 = b1 - lr * sum_db1
    w2 = w2 - lr * sum_dw2
    b2 = b2 - lr * sum_db2
    w3 = w3 - lr * sum_dw3
    b3 = b3 - lr * sum_db3
    w4 = w4 - lr * sum_dw4
    b4 = b4 - lr * sum_db4
    w5 = w5 - lr * sum_dw5
    b5 = b5 - lr * sum_db5
    v1 = v1 - lr * sum_dv1
    v2 = v2 - lr * sum_dv2
    v3 = v3 - lr * sum_dv3
    v4 = v4 - lr * sum_dv4
    v5 = v5 - lr * sum_dv5
    c  = c  - lr * sum_dc

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
            true_y = 0.015 * (x - 30) ** 2 + 1
            noise = random.gauss(0, 5)
            y = true_y + noise
            points.append((x, y))
    return points

def data_to_screen(x, y):
    # Se asume que los datos están en el rango [0, 100]
    x_screen = PLOT_LEFT + (x / 100) * PLOT_WIDTH
    y_screen = PLOT_TOP + PLOT_HEIGHT - (y / 100) * PLOT_HEIGHT
    return int(x_screen), int(y_screen)

def draw_model_curve(surface):
    """Dibuja la función aprendida por la red en el rango de x de 0 a 100."""
    puntos = []
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

# Sliders para los parámetros de la capa oculta (5 neuronas)
slider_w1 = Slider( 50, 630, 200, 20, -50, 50, w1, label="w1")
slider_b1 = Slider(300, 630, 200, 20, -50, 50, b1, label="b1")
slider_w2 = Slider( 50, 660, 200, 20, -50, 50, w2, label="w2")
slider_b2 = Slider(300, 660, 200, 20, -50, 50, b2, label="b2")
slider_w3 = Slider( 50, 690, 200, 20, -50, 50, w3, label="w3")
slider_b3 = Slider(300, 690, 200, 20, -50, 50, b3, label="b3")
slider_w4 = Slider( 50, 720, 200, 20, -50, 50, w4, label="w4")
slider_b4 = Slider(300, 720, 200, 20, -50, 50, b4, label="b4")
slider_w5 = Slider( 50, 750, 200, 20, -50, 50, w5, label="w5")
slider_b5 = Slider(300, 750, 200, 20, -50, 50, b5, label="b5")
# Sliders para la capa de salida (5 neuronas) y el sesgo
slider_v1 = Slider( 50, 780, 200, 20, -50, 50, v1, label="v1")
slider_v2 = Slider(300, 780, 200, 20, -50, 50, v2, label="v2")
slider_v3 = Slider( 50, 810, 200, 20, -50, 50, v3, label="v3")
slider_v4 = Slider(300, 810, 200, 20, -50, 50, v4, label="v4")
slider_v5 = Slider( 50, 840, 200, 20, -50, 50, v5, label="v5")
slider_c  = Slider(300, 840, 200, 20, -50, 50, c,  label="c")
slider_lr = Slider(550, 750, 200, 20, 0.000001, 0.00005, 0.00001, label="Learning Rate")

# Cuadros de texto para configurar steps por segundo y número de puntos
textbox_steps = TextBox(50, 880, 100, 40, text="5000")
textbox_n_points = TextBox(300, 880, 100, 40, text="100")
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

        # Manejo de eventos para botones y controles
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
            if button_dataset.is_clicked(pos):
                dataset_type = "cuadratico" if dataset_type == "lineal" else "lineal"
                n_points = textbox_n_points.get_value()
                points = generate_points(dataset_type, n_points)
            if button_step.is_clicked(pos):
                current_lr = slider_lr.value
                gradient_descent_step_nn(points, current_lr)
                # Actualizar sliders con los nuevos parámetros
                slider_w1.set_value(w1)
                slider_b1.set_value(b1)
                slider_w2.set_value(w2)
                slider_b2.set_value(b2)
                slider_w3.set_value(w3)
                slider_b3.set_value(b3)
                slider_w4.set_value(w4)
                slider_b4.set_value(b4)
                slider_w5.set_value(w5)
                slider_b5.set_value(b5)
                slider_v1.set_value(v1)
                slider_v2.set_value(v2)
                slider_v3.set_value(v3)
                slider_v4.set_value(v4)
                slider_v5.set_value(v5)
                slider_c.set_value(c)
            if button_play.is_clicked(pos):
                play_mode = not play_mode
                button_play.text = "Pause" if play_mode else "Play"

        # Manejo de eventos para los sliders
        slider_w1.handle_event(event)
        slider_b1.handle_event(event)
        slider_w2.handle_event(event)
        slider_b2.handle_event(event)
        slider_w3.handle_event(event)
        slider_b3.handle_event(event)
        slider_w4.handle_event(event)
        slider_b4.handle_event(event)
        slider_w5.handle_event(event)
        slider_b5.handle_event(event)
        slider_v1.handle_event(event)
        slider_v2.handle_event(event)
        slider_v3.handle_event(event)
        slider_v4.handle_event(event)
        slider_v5.handle_event(event)
        slider_c.handle_event(event)
        slider_lr.handle_event(event)
        # Eventos para los TextBoxes
        textbox_steps.handle_event(event)
        textbox_n_points.handle_event(event)

    # Modo automático: ejecutar GD automáticamente en Play/Pause
    if play_mode:
        accumulated_time += dt
        steps_per_sec = textbox_steps.get_value()
        while accumulated_time >= 1.0 / steps_per_sec:
            current_lr = slider_lr.value
            gradient_descent_step_nn(points, current_lr)
            accumulated_time -= 1.0 / steps_per_sec
            # Actualizar sliders con los nuevos parámetros
            slider_w1.set_value(w1)
            slider_b1.set_value(b1)
            slider_w2.set_value(w2)
            slider_b2.set_value(b2)
            slider_w3.set_value(w3)
            slider_b3.set_value(b3)
            slider_w4.set_value(w4)
            slider_b4.set_value(b4)
            slider_w5.set_value(w5)
            slider_b5.set_value(b5)
            slider_v1.set_value(v1)
            slider_v2.set_value(v2)
            slider_v3.set_value(v3)
            slider_v4.set_value(v4)
            slider_v5.set_value(v5)
            slider_c.set_value(c)

    # Actualizar globalmente los parámetros con los valores actuales de los sliders
    w1 = slider_w1.value
    b1 = slider_b1.value
    w2 = slider_w2.value
    b2 = slider_b2.value
    w3 = slider_w3.value
    b3 = slider_b3.value
    w4 = slider_w4.value
    b4 = slider_b4.value
    w5 = slider_w5.value
    b5 = slider_b5.value
    v1 = slider_v1.value
    v2 = slider_v2.value
    v3 = slider_v3.value
    v4 = slider_v4.value
    v5 = slider_v5.value
    c  = slider_c.value

    # Dibujar fondo y área de gráfico
    screen.fill(WHITE)
    pygame.draw.rect(screen, BLACK, PLOT_RECT, 2)

    # Dibujar puntos de datos
    for (x, y) in points:
        pos = data_to_screen(x, y)
        pygame.draw.circle(screen, BLUE, pos, 3)

    # Dibujar la curva que representa la función aprendida por la red
    draw_model_curve(screen)

    # Dibujar botones
    button_dataset.draw(screen)
    button_step.draw(screen)
    button_play.draw(screen)

    # Dibujar sliders para la capa oculta
    slider_w1.draw(screen)
    slider_b1.draw(screen)
    slider_w2.draw(screen)
    slider_b2.draw(screen)
    slider_w3.draw(screen)
    slider_b3.draw(screen)
    slider_w4.draw(screen)
    slider_b4.draw(screen)
    slider_w5.draw(screen)
    slider_b5.draw(screen)
    # Dibujar sliders para la capa de salida y sesgo
    slider_v1.draw(screen)
    slider_v2.draw(screen)
    slider_v3.draw(screen)
    slider_v4.draw(screen)
    slider_v5.draw(screen)
    slider_c.draw(screen)
    slider_lr.draw(screen)

    # Dibujar cuadros de texto y sus etiquetas
    textbox_steps.draw(screen)
    textbox_n_points.draw(screen)
    steps_label = font_ui.render("Steps/s", True, BLACK)
    screen.blit(steps_label, (textbox_steps.rect.x + textbox_steps.rect.width + 10, textbox_steps.rect.y + 10))
    n_points_label = font_ui.render("Points", True, BLACK)
    screen.blit(n_points_label, (textbox_n_points.rect.x + textbox_n_points.rect.width + 10, textbox_n_points.rect.y + 10))

    # Mostrar el error (MSE)
    error_value = compute_error_nn(points)
    error_text = font_ui.render(f"Error: {error_value:.2f}", True, BLACK)
    screen.blit(error_text, (PLOT_LEFT + PLOT_WIDTH + 20, PLOT_TOP))

    pygame.display.flip()

pygame.quit()
sys.exit()

