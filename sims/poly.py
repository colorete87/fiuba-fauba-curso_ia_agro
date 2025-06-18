import pygame
import sys
import random
import math

# Inicialización de pygame
pygame.init()

# Tamaño de la ventana: se aumenta la altura para disponer de todos los controles de UI
WIDTH, HEIGHT = 800, 900
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Simulación de Modelo Polinomial de Orden 8 y Gradiente Descendente")

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
            handle_rect = pygame.Rect(
                self.handle_x - self.handle_radius, 
                self.rect.centery - self.handle_radius,
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
# Modelo Polinomial de Orden 8
#
# f(x)= c0 + c1*x + c2*x^2 + c3*x^3 + c4*x^4 + c5*x^5 + c6*x^6 + c7*x^7 + c8*x^8
###############################################################################
c0 = 18.0#random.uniform(-0.0001, 0.0001)
c1 = 0.7#random.uniform(-0.0001, 0.0001)
c2 = 0.0#random.uniform(-0.0001, 0.0001)
c3 = 0.0#random.uniform(-0.0001, 0.0001)
c4 = 0.0#random.uniform(-0.0001, 0.0001)
c5 = 0.0#random.uniform(-0.0001, 0.0001)
c6 = 0.0#random.uniform(-0.0001, 0.0001)
c7 = 0.0#random.uniform(-0.0001, 0.0001)
c8 = 0.0#random.uniform(-0.0001, 0.0001)

def poly_forward(x):
    return (c0 + c1*x + c2*(x**2) + c3*(x**3) + c4*(x**4) +
            c5*(x**5) + c6*(x**6) + c7*(x**7) + c8*(x**8))

def compute_error_poly(points):
    mse = 0.0
    n = len(points)
    for (x, y) in points:
        pred = poly_forward(x)
        mse += (pred - y) ** 2
    return mse / n if n > 0 else 0

def gradient_descent_step_poly(points, lr):
    global c0, c1, c2, c3, c4, c5, c6, c7, c8
    n = len(points)
    grads = [0.0 for _ in range(9)]
    for (x, y) in points:
        pred = poly_forward(x)
        error = pred - y
        # Gradiente para cada coeficiente: x**j
        for j in range(9):
            grads[j] += (2/n) * error * (x ** j)
    #print(lr)
    #print(grads)
    c0 = c0 - lr * grads[0]
    c1 = c1 - lr * grads[1] / 1e2
    c2 = c2 - lr * grads[2] / 1e3
    c3 = c3 - lr * grads[3] / 1e4
    #c4 = c4 - lr * grads[4] / 1e25
    #c5 = c5 - lr * grads[5] / 1e25
    #c6 = c6 - lr * grads[6] / 1e25
    #c7 = c7 - lr * grads[7] / 1e25
    #c8 = c8 - lr * grads[8] / 1e25
    print(f"c0 = {c0:e}")
    print(f"c1 = {c1:e}")
    print(f"c2 = {c2:e}")
    print(f"c3 = {c3:e}")
    print(f"c4 = {c4:e}")
    print(f"c5 = {c5:e}")
    print(f"c6 = {c6:e}")
    print(f"c7 = {c7:e}")
    print(f"c8 = {c8:e}")

def data_to_screen(x, y):
    x_screen = PLOT_LEFT + (x / 100) * PLOT_WIDTH
    y_screen = PLOT_TOP + PLOT_HEIGHT - (y / 100) * PLOT_HEIGHT
    return int(x_screen), int(y_screen)

def draw_model_curve(surface):
    """Dibuja la función aprendida por el polinomio en el rango de x de 0 a 100."""
    puntos = []
    for i in range(101):
        x = i
        y = poly_forward(x)
        puntos.append(data_to_screen(x, y))
    if len(puntos) > 1:
        pygame.draw.lines(surface, RED, False, puntos, 2)

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
        # Ejemplo: y = 0.015*(x-30)^2 + 1 + ruido
        for _ in range(n):
            x = random.uniform(0, 100)
            true_y = 0.015 * (x - 30) ** 2 + 1
            noise = random.gauss(0, 5)
            y = true_y + noise
            points.append((x, y))
    return points

###############################################################################
# Estado inicial, elementos de UI y variables globales
###############################################################################
dataset_type = "lineal"
points = generate_points(dataset_type, 10)

# Parámetros del modelo se representan ahora con c0...c8

# Elementos de UI
button_dataset = Button(50, 570, 150, 40, "Cambiar Conjunto")
button_step = Button(220, 570, 150, 40, "Paso GD")
button_play = Button(400, 570, 150, 40, "Play")  # Botón Play/Pause

# Creamos 9 sliders para los coeficientes del polinomio y uno para el learning rate.
slider_c0 = Slider( 50, 630, 200, 20, -100, 100, c0, label="c0")
slider_c1 = Slider(300, 630, 200, 20, -2, 2, c1, label="c1")
slider_c2 = Slider( 50, 660, 200, 20, -0.01, 0.01, c2, label="c2")
slider_c3 = Slider(300, 660, 200, 20, -0.01, 0.01, c3, label="c3")
slider_c4 = Slider( 50, 690, 200, 20, -0.01, 0.01, c4, label="c4")
slider_c5 = Slider(300, 690, 200, 20, -0.01, 0.01, c5, label="c5")
slider_c6 = Slider( 50, 720, 200, 20, -0.01, 0.01, c6, label="c6")
slider_c7 = Slider(300, 720, 200, 20, -0.01, 0.01, c7, label="c7")
slider_c8 = Slider( 50, 750, 200, 20, -0.01, 0.01, c8, label="c8")
slider_lr = Slider(300, 750, 200, 20, -30, -3, -15, label="Learning Rate")

# Cuadros de texto para configurar los pasos por segundo y el número de puntos
textbox_steps = TextBox(50, 800, 100, 40, text="1000")
textbox_n_points = TextBox(300, 800, 100, 40, text="10")
font_ui = pygame.font.SysFont(None, 24)

# Variables para el modo automático (Play/Pause)
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
        slider_c0.handle_event(event)
        slider_c1.handle_event(event)
        slider_c2.handle_event(event)
        slider_c3.handle_event(event)
        slider_c4.handle_event(event)
        slider_c5.handle_event(event)
        slider_c6.handle_event(event)
        slider_c7.handle_event(event)
        slider_c8.handle_event(event)
        slider_lr.handle_event(event)
        textbox_steps.handle_event(event)
        textbox_n_points.handle_event(event)
        
        if event.type == pygame.MOUSEBUTTONDOWN:
            pos = event.pos
            if button_dataset.is_clicked(pos):
                dataset_type = "cuadratico" if dataset_type == "lineal" else "lineal"
                n_val = textbox_n_points.get_value()
                points = generate_points(dataset_type, n_val)
            if button_step.is_clicked(pos):
                current_lr = 10**slider_lr.value
                gradient_descent_step_poly(points, current_lr)
                # Actualizamos los sliders para reflejar los nuevos valores de los coeficientes
                slider_c0.set_value(c0)
                slider_c1.set_value(c1)
                slider_c2.set_value(c2)
                slider_c3.set_value(c3)
                slider_c4.set_value(c4)
                slider_c5.set_value(c5)
                slider_c6.set_value(c6)
                slider_c7.set_value(c7)
                slider_c8.set_value(c8)
            if button_play.is_clicked(pos):
                play_mode = not play_mode
                button_play.text = "Pause" if play_mode else "Play"
                
    # Si estamos en modo Play, ejecutar automáticamente varios pasos de GD
    if play_mode:
        accumulated_time += dt
        steps_per_sec = textbox_steps.get_value()
        while accumulated_time >= 1.0 / steps_per_sec:
            current_lr = 10**slider_lr.value
            gradient_descent_step_poly(points, current_lr)
            accumulated_time -= 1.0 / steps_per_sec
            slider_c0.set_value(c0)
            slider_c1.set_value(c1)
            slider_c2.set_value(c2)
            slider_c3.set_value(c3)
            slider_c4.set_value(c4)
            slider_c5.set_value(c5)
            slider_c6.set_value(c6)
            slider_c7.set_value(c7)
            slider_c8.set_value(c8)
            
    # Actualizamos los parámetros globales con lo que indique el usuario (si se arrastran los sliders)
    c0 = slider_c0.value
    c1 = slider_c1.value
    c2 = slider_c2.value
    c3 = slider_c3.value
    c4 = slider_c4.value
    c5 = slider_c5.value
    c6 = slider_c6.value
    c7 = slider_c7.value
    c8 = slider_c8.value
    
    # Dibujar fondo y área de gráfico
    screen.fill(WHITE)
    pygame.draw.rect(screen, BLACK, PLOT_RECT, 2)
    
    # Dibujar los puntos de datos
    for (x, y) in points:
        pos = data_to_screen(x, y)
        pygame.draw.circle(screen, BLUE, pos, 3)
    
    # Dibujar la curva del polinomio (modelo aprendido)
    puntos_modelo = []
    for i in range(101):
        x = i
        y = poly_forward(x)
        puntos_modelo.append(data_to_screen(x, y))
    if len(puntos_modelo) > 1:
        pygame.draw.lines(screen, RED, False, puntos_modelo, 2)
    
    # Dibujar elementos de UI: botones
    button_dataset.draw(screen)
    button_step.draw(screen)
    button_play.draw(screen)
    
    # Dibujar sliders para los coeficientes y learning rate
    slider_c0.draw(screen)
    slider_c1.draw(screen)
    slider_c2.draw(screen)
    slider_c3.draw(screen)
    slider_c4.draw(screen)
    slider_c5.draw(screen)
    slider_c6.draw(screen)
    slider_c7.draw(screen)
    slider_c8.draw(screen)
    slider_lr.draw(screen)
    
    # Dibujar cuadros de texto y sus etiquetas
    textbox_steps.draw(screen)
    textbox_n_points.draw(screen)
    steps_label = font_ui.render("Steps/s", True, BLACK)
    screen.blit(steps_label, (textbox_steps.rect.x + textbox_steps.rect.width + 10, textbox_steps.rect.y + 10))
    n_points_label = font_ui.render("Points", True, BLACK)
    screen.blit(n_points_label, (textbox_n_points.rect.x + textbox_n_points.rect.width + 10, textbox_n_points.rect.y + 10))
    
    # Mostrar error (MSE) en la parte superior derecha del plot
    error_value = compute_error_poly(points)
    error_text = font_ui.render(f"MSE: {error_value:.2f}", True, BLACK)
    screen.blit(error_text, (PLOT_LEFT + PLOT_WIDTH + 20, PLOT_TOP))
    
    pygame.display.flip()

pygame.quit()
sys.exit()


