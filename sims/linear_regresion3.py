import pygame
import sys
import random
import math
import numpy as np

# Inicialización de pygame
pygame.init()

# Tamaño de la ventana
WIDTH, HEIGHT = 1200, 800
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Advanced Linear Regression & Gradient Descent Visualization")

clock = pygame.time.Clock()

# Área del gráfico principal
PLOT_LEFT, PLOT_TOP = 50, 50
PLOT_WIDTH, PLOT_HEIGHT = 500, 400
PLOT_RECT = pygame.Rect(PLOT_LEFT, PLOT_TOP, PLOT_WIDTH, PLOT_HEIGHT)

# Área del gráfico de coste
COST_PLOT_LEFT, COST_PLOT_TOP = 620, 50
COST_PLOT_WIDTH, COST_PLOT_HEIGHT = 500, 200
COST_PLOT_RECT = pygame.Rect(COST_PLOT_LEFT, COST_PLOT_TOP, COST_PLOT_WIDTH, COST_PLOT_HEIGHT)

# Paleta de colores moderna
COLORS = {
    'bg_primary': (248, 249, 250),
    'bg_secondary': (233, 236, 239),
    'bg_tertiary': (108, 117, 125),
    'primary': (0, 123, 255),
    'secondary': (108, 117, 125),
    'success': (40, 167, 69),
    'danger': (220, 53, 69),
    'warning': (255, 193, 7),
    'info': (23, 162, 184),
    'light': (248, 249, 250),
    'dark': (52, 58, 64),
    'white': (255, 255, 255),
    'black': (0, 0, 0),
    'gray_100': (248, 249, 250),
    'gray_200': (233, 236, 239),
    'gray_300': (222, 226, 230),
    'gray_400': (206, 212, 218),
    'gray_500': (173, 181, 189),
    'gray_600': (108, 117, 125),
    'gray_700': (73, 80, 87),
    'gray_800': (52, 58, 64),
    'gray_900': (33, 37, 41),
    'data_points': (52, 144, 220),
    'regression_line': (220, 53, 69),
    'cost_curve': (40, 167, 69)
}

###############################################################################
# Clases para elementos de UI mejorados
###############################################################################

class ModernSlider:
    def __init__(self, x, y, width, height, min_val, max_val, init_val, label="", color=COLORS['primary']):
        self.rect = pygame.Rect(x, y, width, height)
        self.min_val = min_val
        self.max_val = max_val
        self._value = init_val
        self.label = label
        self.color = color
        self.handle_radius = height // 2 + 2
        self.handle_x = self.get_handle_x()
        self.dragging = False
        self.font = pygame.font.SysFont('Arial', 16)
        self.hovered = False

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
            self.hovered = self.rect.collidepoint(event.pos)
            if self.dragging:
                x = event.pos[0]
                x = max(self.rect.x, min(x, self.rect.x + self.rect.width))
                ratio = (x - self.rect.x) / self.rect.width
                self._value = self.min_val + ratio * (self.max_val - self.min_val)
                self.handle_x = x

    def draw(self, surface):
        # Fondo del slider
        bg_color = COLORS['gray_200'] if not self.hovered else COLORS['gray_300']
        pygame.draw.rect(surface, bg_color, self.rect, border_radius=10)
        
        # Línea de progreso
        progress_rect = pygame.Rect(self.rect.x, self.rect.centery - 2, 
                                   self.handle_x - self.rect.x, 4)
        pygame.draw.rect(surface, self.color, progress_rect, border_radius=2)
        
        # Handle
        handle_color = self.color if self.dragging else (self.color[0], self.color[1], self.color[2], 200)
        pygame.draw.circle(surface, handle_color, (self.handle_x, self.rect.centery), self.handle_radius)
        pygame.draw.circle(surface, COLORS['white'], (self.handle_x, self.rect.centery), self.handle_radius - 2)
        
        # Etiqueta
        label_text = f"{self.label}: {self._value:.6f}"
        label_surf = self.font.render(label_text, True, COLORS['dark'])
        surface.blit(label_surf, (self.rect.x, self.rect.y - 25))

class ModernButton:
    def __init__(self, x, y, width, height, text, color=COLORS['primary'], text_color=COLORS['white']):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.color = color
        self.text_color = text_color
        self.font = pygame.font.SysFont('Arial', 18, bold=True)
        self.hovered = False
        self.pressed = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.pressed = True
        elif event.type == pygame.MOUSEBUTTONUP:
            if self.pressed and self.rect.collidepoint(event.pos):
                self.pressed = False
                return True
            self.pressed = False
        return False

    def draw(self, surface):
        # Color del botón
        if self.pressed:
            btn_color = tuple(max(0, c - 30) for c in self.color)
        elif self.hovered:
            btn_color = tuple(min(255, c + 20) for c in self.color)
        else:
            btn_color = self.color
            
        # Dibujar botón con bordes redondeados
        pygame.draw.rect(surface, btn_color, self.rect, border_radius=8)
        pygame.draw.rect(surface, COLORS['dark'], self.rect, 2, border_radius=8)
        
        # Texto
        text_surf = self.font.render(self.text, True, self.text_color)
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

class ModernTextBox:
    def __init__(self, x, y, width, height, text="1", label=""):
        self.rect = pygame.Rect(x, y, width, height)
        self.active = False
        self.text = text
        self.label = label
        self.font = pygame.font.SysFont('Arial', 16)
        self.hovered = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN:
            if self.rect.collidepoint(event.pos):
                self.active = True
            else:
                self.active = False
        elif event.type == pygame.MOUSEMOTION:
            self.hovered = self.rect.collidepoint(event.pos)
        elif event.type == pygame.KEYDOWN and self.active:
            if event.key == pygame.K_RETURN:
                self.active = False
            elif event.key == pygame.K_BACKSPACE:
                self.text = self.text[:-1]
            elif event.unicode.isdigit() or event.unicode == '.':
                self.text += event.unicode

    def draw(self, surface):
        # Fondo del textbox
        bg_color = COLORS['white'] if not self.hovered else COLORS['gray_100']
        border_color = COLORS['primary'] if self.active else COLORS['gray_400']
        
        pygame.draw.rect(surface, bg_color, self.rect, border_radius=5)
        pygame.draw.rect(surface, border_color, self.rect, 2, border_radius=5)
        
        # Texto
        display_text = self.text
        if self.active and (pygame.time.get_ticks() // 500) % 2 == 0:
            display_text += "|"
            
        text_surf = self.font.render(display_text, True, COLORS['dark'])
        surface.blit(text_surf, (self.rect.x + 8, self.rect.y + 8))
        
        # Etiqueta
        if self.label:
            label_surf = self.font.render(self.label, True, COLORS['gray_600'])
            surface.blit(label_surf, (self.rect.x, self.rect.y - 20))

    def get_value(self):
        try:
            return max(0.001, float(self.text))
        except:
            return 0.001

class DatasetSelector:
    def __init__(self, x, y, width, height):
        self.rect = pygame.Rect(x, y, width, height)
        self.selected = 0
        self.options = ["Linear", "Quadratic", "Sine Wave", "Random"]
        self.font = pygame.font.SysFont('Arial', 16)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            self.selected = (self.selected + 1) % len(self.options)

    def draw(self, surface):
        pygame.draw.rect(surface, COLORS['gray_200'], self.rect, border_radius=8)
        pygame.draw.rect(surface, COLORS['dark'], self.rect, 2, border_radius=8)
        
        text = f"Dataset: {self.options[self.selected]}"
        text_surf = self.font.render(text, True, COLORS['dark'])
        text_rect = text_surf.get_rect(center=self.rect.center)
        surface.blit(text_surf, text_rect)

###############################################################################
# Funciones para generación de datos y procesamiento
###############################################################################

def generate_points(dataset_type, n=100, noise_level=8):
    points = []
    if dataset_type == 0:  # Linear
        for _ in range(n):
            x = random.uniform(0, 100)
            true_y = 0.7 * x + 20
            noise = random.gauss(0, noise_level)
            y = true_y + noise
            points.append((x, y))
    elif dataset_type == 1:  # Quadratic
        for _ in range(n):
            x = random.uniform(0, 100)
            true_y = 0.015 * (x - 30) ** 2 + 10
            noise = random.gauss(0, noise_level)
            y = true_y + noise
            points.append((x, y))
    elif dataset_type == 2:  # Sine Wave
        for _ in range(n):
            x = random.uniform(0, 100)
            true_y = 30 * math.sin(0.1 * x) + 50
            noise = random.gauss(0, noise_level)
            y = true_y + noise
            points.append((x, y))
    elif dataset_type == 3:  # Random
        for _ in range(n):
            x = random.uniform(0, 100)
            y = random.uniform(0, 100)
            points.append((x, y))
    return points

def compute_error(points, a, b):
    mse = 0
    n = len(points)
    for (x, y) in points:
        pred = a * x + b
        mse += (pred - y) ** 2
    return mse / n if n > 0 else 0

def compute_r_squared(points, a, b):
    y_mean = sum(y for _, y in points) / len(points)
    ss_tot = sum((y - y_mean) ** 2 for _, y in points)
    ss_res = sum((y - (a * x + b)) ** 2 for x, y in points)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

def gradient_descent_step(points, a, b, lr, lr_a=1.0, lr_b=100.0):
    n = len(points)
    grad_a = 0
    grad_b = 0
    for (x, y) in points:
        pred = a * x + b
        error = pred - y
        grad_a += (2/n) * x * error
        grad_b += (2/n) * error
    a_new = a - lr * lr_a * grad_a
    b_new = b - lr * lr_b * grad_b
    return a_new, b_new

def data_to_screen(x, y, plot_rect):
    x_screen = plot_rect.x + (x / 100) * plot_rect.width
    y_screen = plot_rect.y + plot_rect.height - (y / 100) * plot_rect.height
    return int(x_screen), int(y_screen)

def draw_grid(surface, rect, color=COLORS['gray_300']):
    # Líneas verticales
    for i in range(0, 101, 20):
        x = rect.x + (i / 100) * rect.width
        pygame.draw.line(surface, color, (x, rect.y), (x, rect.y + rect.height), 1)
    
    # Líneas horizontales
    for i in range(0, 101, 20):
        y = rect.y + rect.height - (i / 100) * rect.height
        pygame.draw.line(surface, color, (rect.x, y), (rect.x + rect.width, y), 1)

def draw_axis_labels(surface, rect, font):
    # Etiquetas X
    for i in range(0, 101, 20):
        x = rect.x + (i / 100) * rect.width
        label_surf = font.render(str(i), True, COLORS['gray_600'])
        surface.blit(label_surf, (x - 10, rect.y + rect.height + 5))
    
    # Etiquetas Y
    for i in range(0, 101, 20):
        y = rect.y + rect.height - (i / 100) * rect.height
        label_surf = font.render(str(i), True, COLORS['gray_600'])
        surface.blit(label_surf, (rect.x - 25, y - 10))

###############################################################################
# Estado inicial y elementos de UI
###############################################################################

# Estado inicial
dataset_type = 0
points = generate_points(dataset_type, noise_level=8)
cost_history = []
max_cost_history = 100000

# Parámetros del modelo
a = 0.0
b = 0.0

# Elementos de UI
dataset_selector = DatasetSelector(50, 490, 200, 40)
button_step = ModernButton(270, 490, 120, 40, "Step", COLORS['success'])
button_play = ModernButton(410, 490, 120, 40, "Play", COLORS['info'])
button_reset = ModernButton(550, 490, 120, 40, "Reset", COLORS['warning'])
button_auto_fit = ModernButton(690, 490, 120, 40, "Auto Fit", COLORS['danger'])

# Sliders (moved down)
slider_a = ModernSlider(50, 580, 200, 20, -2, 2, a, "Slope (a)", COLORS['primary'])
slider_b = ModernSlider(270, 580, 200, 20, -50, 150, b, "Intercept (b)", COLORS['secondary'])
slider_lr = ModernSlider(490, 580, 200, 20, 0.000001, 0.001, 0.0001, "Learning Rate", COLORS['info'])
slider_noise = ModernSlider(710, 580, 200, 20, 0, 20, 8, "Noise Level", COLORS['warning'])

# Log scale button for cost function
button_log_scale = ModernButton(50, 630, 150, 40, "Linear Scale", COLORS['secondary'])

# TextBoxes (moved down)
textbox_steps = ModernTextBox(220, 640, 80, 30, "4", "Steps/sec")
textbox_n_points = ModernTextBox(320, 640, 80, 30, "100", "Points")

# Variables para el modo automático
play_mode = False
accumulated_time = 0.0
log_scale = False  # For cost function x-axis
previous_noise_level = 8  # Track noise level changes

# Fuentes
title_font = pygame.font.SysFont('Arial', 24, bold=True)
subtitle_font = pygame.font.SysFont('Arial', 18, bold=True)
text_font = pygame.font.SysFont('Arial', 14)

###############################################################################
# Bucle principal
###############################################################################

running = True
while running:
    dt = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                running = False

        # Manejo de eventos para UI
        dataset_selector.handle_event(event)
        slider_a.handle_event(event)
        slider_b.handle_event(event)
        slider_lr.handle_event(event)
        slider_noise.handle_event(event)
        textbox_steps.handle_event(event)
        textbox_n_points.handle_event(event)

        # Botones
        if button_step.handle_event(event):
            current_lr = slider_lr.value
            new_a, new_b = gradient_descent_step(points, slider_a.value, slider_b.value, current_lr)
            slider_a.set_value(new_a)
            slider_b.set_value(new_b)
            cost_history.append(compute_error(points, new_a, new_b))
            if len(cost_history) > max_cost_history:
                cost_history.pop(0)

        if button_play.handle_event(event):
            play_mode = not play_mode
            button_play.text = "Pause" if play_mode else "Play"

        if button_reset.handle_event(event):
            slider_a.set_value(0.0)
            slider_b.set_value(0.0)
            cost_history = []

        if button_auto_fit.handle_event(event):
            # Implementación simple de ajuste automático
            for _ in range(100):
                current_lr = slider_lr.value
                new_a, new_b = gradient_descent_step(points, slider_a.value, slider_b.value, current_lr)
                slider_a.set_value(new_a)
                slider_b.set_value(new_b)
                cost_history.append(compute_error(points, new_a, new_b))
                if len(cost_history) > max_cost_history:
                    cost_history.pop(0)

        if button_log_scale.handle_event(event):
            log_scale = not log_scale
            button_log_scale.text = "Log Scale" if log_scale else "Linear Scale"

        # Cambio de dataset
        if dataset_selector.selected != dataset_type:
            dataset_type = dataset_selector.selected
            n_points = int(textbox_n_points.get_value())
            noise_level = slider_noise.value
            points = generate_points(dataset_type, n_points, noise_level)
            cost_history = []
        

    # Actualización automática en modo play
    if play_mode:
        accumulated_time += dt
        steps_per_sec = textbox_steps.get_value()
        while accumulated_time >= 1.0 / steps_per_sec:
            current_lr = slider_lr.value
            new_a, new_b = gradient_descent_step(points, slider_a.value, slider_b.value, current_lr)
            slider_a.set_value(new_a)
            slider_b.set_value(new_b)
            cost_history.append(compute_error(points, new_a, new_b))
            if len(cost_history) > max_cost_history:
                cost_history.pop(0)
            accumulated_time -= 1.0 / steps_per_sec

    # Actualizar variables globales
    a = slider_a.value
    b = slider_b.value
    
    # Check if noise level changed and regenerate dataset
    current_noise_level = slider_noise.value
    if abs(current_noise_level - previous_noise_level) > 0.1:  # Small threshold to avoid constant regeneration
        n_points = int(textbox_n_points.get_value())
        points = generate_points(dataset_type, n_points, current_noise_level)
        cost_history = []
        previous_noise_level = current_noise_level

    # Dibujar fondo
    screen.fill(COLORS['bg_primary'])

    # Dibujar gráfico principal
    pygame.draw.rect(screen, COLORS['white'], PLOT_RECT, border_radius=10)
    pygame.draw.rect(screen, COLORS['dark'], PLOT_RECT, 2, border_radius=10)
    draw_grid(screen, PLOT_RECT)
    draw_axis_labels(screen, PLOT_RECT, text_font)

    # Dibujar puntos de datos
    for (x, y) in points:
        pos = data_to_screen(x, y, PLOT_RECT)
        pygame.draw.circle(screen, COLORS['data_points'], pos, 4)
        pygame.draw.circle(screen, COLORS['white'], pos, 2)

    # Dibujar línea de regresión
    if len(points) > 0:
        x1_data, x2_data = 0, 100
        y1_data = a * x1_data + b
        y2_data = a * x2_data + b
        p1 = data_to_screen(x1_data, y1_data, PLOT_RECT)
        p2 = data_to_screen(x2_data, y2_data, PLOT_RECT)
        pygame.draw.line(screen, COLORS['regression_line'], p1, p2, 3)

    # Dibujar gráfico de coste
    pygame.draw.rect(screen, COLORS['white'], COST_PLOT_RECT, border_radius=10)
    pygame.draw.rect(screen, COLORS['dark'], COST_PLOT_RECT, 2, border_radius=10)
    
    if len(cost_history) > 1:
        # Normalizar coste para visualización
        max_cost = max(cost_history) if cost_history else 1
        min_cost = min(cost_history) if cost_history else 0
        cost_range = max_cost - min_cost if max_cost > min_cost else 1
        
        points_cost = []
        total_steps = len(cost_history)
        
        for i, cost in enumerate(cost_history):
            # X-axis: auto-adjust to show all steps proportionally
            x_ratio = i / max(1, total_steps - 1)
            x = COST_PLOT_RECT.x + x_ratio * COST_PLOT_RECT.width
            
            # Y-axis: optional log scale for cost values
            if log_scale and cost > 0:
                # Log scale: use log10 of cost value
                log_cost = math.log10(cost)
                log_min = math.log10(min_cost) if min_cost > 0 else log_cost
                log_max = math.log10(max_cost) if max_cost > 0 else log_cost
                log_range = log_max - log_min if log_max > log_min else 1
                y_ratio = (log_cost - log_min) / log_range
            else:
                # Linear scale: normal cost values
                y_ratio = (cost - min_cost) / cost_range
            
            y = COST_PLOT_RECT.y + COST_PLOT_RECT.height - y_ratio * COST_PLOT_RECT.height
            points_cost.append((x, y))
        
        if len(points_cost) > 1:
            pygame.draw.lines(screen, COLORS['cost_curve'], False, points_cost, 2)
        
        # Draw grid lines for cost plot
        if total_steps > 1:
            # Horizontal grid lines for Y-axis (cost values)
            for i in range(1, 6):
                if log_scale and min_cost > 0:
                    # Log scale grid lines
                    log_min = math.log10(min_cost)
                    log_max = math.log10(max_cost)
                    log_range = log_max - log_min
                    cost_value = 10**(log_min + (i * log_range / 5))
                    y_ratio = (math.log10(cost_value) - log_min) / log_range
                else:
                    # Linear scale grid lines
                    cost_value = min_cost + (i * cost_range / 5)
                    y_ratio = (cost_value - min_cost) / cost_range
                
                y = COST_PLOT_RECT.y + COST_PLOT_RECT.height - y_ratio * COST_PLOT_RECT.height
                if COST_PLOT_RECT.y < y < COST_PLOT_RECT.y + COST_PLOT_RECT.height:
                    pygame.draw.line(screen, COLORS['gray_300'], (COST_PLOT_RECT.x, y), (COST_PLOT_RECT.x + COST_PLOT_RECT.width, y), 1)
            
            # Vertical grid lines for X-axis (steps)
            for i in range(1, 6):
                x_ratio = (i * total_steps / 5) / total_steps
                x = COST_PLOT_RECT.x + x_ratio * COST_PLOT_RECT.width
                if COST_PLOT_RECT.x < x < COST_PLOT_RECT.x + COST_PLOT_RECT.width:
                    pygame.draw.line(screen, COLORS['gray_300'], (x, COST_PLOT_RECT.y), (x, COST_PLOT_RECT.y + COST_PLOT_RECT.height), 1)
            
            # Draw axis labels and ticks for cost plot
            # X-axis labels (steps)
            for i in range(0, 6):
                step_value = int(i * total_steps / 5)
                x_ratio = i / 5
                x = COST_PLOT_RECT.x + x_ratio * COST_PLOT_RECT.width
                step_text = text_font.render(str(step_value), True, COLORS['gray_600'])
                step_rect = step_text.get_rect(center=(x, COST_PLOT_RECT.y + COST_PLOT_RECT.height + 15))
                screen.blit(step_text, step_rect)
            
            # Y-axis labels (cost values)
            for i in range(0, 6):
                if log_scale and min_cost > 0:
                    log_min = math.log10(min_cost)
                    log_max = math.log10(max_cost)
                    log_range = log_max - log_min
                    cost_value = 10**(log_min + (i * log_range / 5))
                    cost_text = f"{cost_value:.1e}" if cost_value > 100 else f"{cost_value:.2f}"
                else:
                    cost_value = min_cost + (i * cost_range / 5)
                    cost_text = f"{cost_value:.2f}"
                
                y_ratio = i / 5
                y = COST_PLOT_RECT.y + COST_PLOT_RECT.height - y_ratio * COST_PLOT_RECT.height
                cost_label = text_font.render(cost_text, True, COLORS['gray_600'])
                cost_rect = cost_label.get_rect(center=(COST_PLOT_RECT.x - 30, y))
                screen.blit(cost_label, cost_rect)

    # Dibujar elementos de UI
    dataset_selector.draw(screen)
    button_step.draw(screen)
    button_play.draw(screen)
    button_reset.draw(screen)
    button_auto_fit.draw(screen)
    button_log_scale.draw(screen)
    
    slider_a.draw(screen)
    slider_b.draw(screen)
    slider_lr.draw(screen)
    slider_noise.draw(screen)
    
    textbox_steps.draw(screen)
    textbox_n_points.draw(screen)

    # Mostrar estadísticas
    error_value = compute_error(points, a, b)
    r_squared = compute_r_squared(points, a, b)
    
    # Panel de estadísticas
    stats_rect = pygame.Rect(620, 300, 500, 150)
    pygame.draw.rect(screen, COLORS['white'], stats_rect, border_radius=10)
    pygame.draw.rect(screen, COLORS['dark'], stats_rect, 2, border_radius=10)
    
    stats_title = subtitle_font.render("Statistics", True, COLORS['dark'])
    screen.blit(stats_title, (stats_rect.x + 10, stats_rect.y + 10))
    
    mse_text = text_font.render(f"MSE: {error_value:.4f}", True, COLORS['dark'])
    screen.blit(mse_text, (stats_rect.x + 10, stats_rect.y + 40))
    
    r2_text = text_font.render(f"R²: {r_squared:.4f}", True, COLORS['dark'])
    screen.blit(r2_text, (stats_rect.x + 10, stats_rect.y + 60))
    
    params_text = text_font.render(f"y = {a:.4f}x + {b:.4f}", True, COLORS['dark'])
    screen.blit(params_text, (stats_rect.x + 10, stats_rect.y + 80))
    
    steps_text = text_font.render(f"Steps: {len(cost_history)}", True, COLORS['dark'])
    screen.blit(steps_text, (stats_rect.x + 10, stats_rect.y + 100))

    # Títulos
    main_title = title_font.render("Linear Regression Visualization", True, COLORS['dark'])
    screen.blit(main_title, (PLOT_LEFT, 10))
    
    cost_title = subtitle_font.render("Cost Function", True, COLORS['dark'])
    screen.blit(cost_title, (COST_PLOT_LEFT, 10))
    
    # Axis labels for cost plot
    if len(cost_history) > 1:
        # X-axis label (Steps)
        x_axis_label = text_font.render("Steps", True, COLORS['gray_600'])
        x_axis_rect = x_axis_label.get_rect(center=(COST_PLOT_RECT.centerx, COST_PLOT_RECT.y + COST_PLOT_RECT.height + 35))
        screen.blit(x_axis_label, x_axis_rect)
        
        # Y-axis label (Cost)
        y_axis_label = text_font.render("Cost", True, COLORS['gray_600'])
        y_axis_rect = y_axis_label.get_rect(center=(COST_PLOT_RECT.x - 50, COST_PLOT_RECT.centery))
        # Rotate the Y-axis label (simulate rotation by positioning)
        screen.blit(y_axis_label, y_axis_rect)

    pygame.display.flip()

pygame.quit()
sys.exit()
