import pygame
import sys
import random
import math
import numpy as np

# Inicialización de pygame
pygame.init()

# Tamaño de la ventana
WIDTH, HEIGHT = 1200, 850
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Advanced Polynomial & MLP Function Fitting Visualization")

clock = pygame.time.Clock()

# Área del gráfico principal
PLOT_LEFT, PLOT_TOP = 50, 50
PLOT_WIDTH, PLOT_HEIGHT = 500, 400
PLOT_RECT = pygame.Rect(PLOT_LEFT, PLOT_TOP, PLOT_WIDTH, PLOT_HEIGHT)

# Área del gráfico de coste
COST_PLOT_LEFT, COST_PLOT_TOP = 620, 50
COST_PLOT_WIDTH, COST_PLOT_HEIGHT = 500, 200
COST_PLOT_RECT = pygame.Rect(COST_PLOT_LEFT, COST_PLOT_TOP, COST_PLOT_WIDTH, COST_PLOT_HEIGHT)

# Área del panel de control
CONTROL_PANEL_LEFT, CONTROL_PANEL_TOP = 50, 530
CONTROL_PANEL_WIDTH, CONTROL_PANEL_HEIGHT = 1100, 300
CONTROL_PANEL_RECT = pygame.Rect(CONTROL_PANEL_LEFT, CONTROL_PANEL_TOP, CONTROL_PANEL_WIDTH, CONTROL_PANEL_HEIGHT)

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
    'cost_curve': (40, 167, 69),
    'real_function': (200, 200, 200)  # Light gray for real function
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
        self.visible = True

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
        if not self.visible:
            return
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
        if not self.visible:
            return
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
        if isinstance(self._value, float):
            label_text = f"{self.label}: {self._value:.6f}"
        else:
            label_text = f"{self.label}: {int(self._value)}"
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
        self.selected = False

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
        if self.selected:
            btn_color = tuple(min(255, c + 40) for c in self.color)
        elif self.pressed:
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

###############################################################################
# MLP Neural Network Implementation
###############################################################################

class MLP:
    def __init__(self, input_size=1, hidden_sizes=[], output_size=1):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        
        # Input to first hidden layer
        if hidden_sizes:
            # He initialization for ReLU networks
            self.weights.append(np.random.randn(input_size, hidden_sizes[0]) * np.sqrt(2.0 / input_size))
            self.biases.append(np.zeros(hidden_sizes[0]))
            
            # Hidden to hidden layers
            for i in range(len(hidden_sizes) - 1):
                self.weights.append(np.random.randn(hidden_sizes[i], hidden_sizes[i+1]) * np.sqrt(2.0 / hidden_sizes[i]))
                self.biases.append(np.zeros(hidden_sizes[i+1]))
            
            # Last hidden to output (smaller initialization for output layer)
            self.weights.append(np.random.randn(hidden_sizes[-1], output_size) * 0.1)
            self.biases.append(np.zeros(output_size))
        else:
            # Direct input to output
            self.weights.append(np.random.randn(input_size, output_size) * 0.1)
            self.biases.append(np.zeros(output_size))
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(np.float64)
    
    def forward(self, x):
        self.activations = [x]
        self.z_values = []
        
        current_input = x
        
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(current_input, w) + b
            self.z_values.append(z)
            
            if i < len(self.weights) - 1:  # Not the output layer
                a = self.relu(z)
            else:  # Output layer - no activation
                a = z
                
            self.activations.append(a)
            current_input = a
        
        return self.activations[-1]
    
    def backward(self, x, y, learning_rate):
        # Forward pass
        output = self.forward(x)
        
        # Calculate output error
        output_error = output - y
        
        # Backpropagate
        deltas = [output_error]
        
        # Hidden layers (backwards)
        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * self.relu_derivative(self.z_values[i-1])
            deltas.insert(0, delta)
        
        # Update weights and biases
        for i in range(len(self.weights)):
            if i == 0:
                prev_activation = x
            else:
                prev_activation = self.activations[i]
            
            # Ensure deltas[i] has the right shape for bias update
            delta_for_bias = deltas[i].flatten() if deltas[i].ndim > 1 else deltas[i]
            
            # Ensure bias shape matches delta shape
            if delta_for_bias.shape != self.biases[i].shape:
                delta_for_bias = delta_for_bias.reshape(self.biases[i].shape)
            
            self.weights[i] -= learning_rate * np.outer(prev_activation, deltas[i])
            self.biases[i] -= learning_rate * delta_for_bias
    
    def get_parameters(self):
        """Get all parameters as a flat array"""
        params = []
        for w in self.weights:
            params.extend(w.flatten())
        for b in self.biases:
            params.extend(b.flatten())
        return np.array(params)
    
    def set_parameters(self, params):
        """Set parameters from a flat array"""
        idx = 0
        for i, w in enumerate(self.weights):
            size = w.size
            self.weights[i] = params[idx:idx+size].reshape(w.shape)
            idx += size
        for i, b in enumerate(self.biases):
            size = b.size
            self.biases[i] = params[idx:idx+size].reshape(b.shape)
            idx += size

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

def get_real_function_value(x, dataset_type):
    """Get the real function value without noise for a given dataset type"""
    if dataset_type == 0:  # Linear
        return 0.7 * x + 20
    elif dataset_type == 1:  # Quadratic
        return 0.015 * (x - 30) ** 2 + 10
    elif dataset_type == 2:  # Sine Wave
        return 30 * math.sin(0.1 * x) + 50
    elif dataset_type == 3:  # Random - no real function
        return None
    return None

def predict_value(x, fit_type, params, mlp=None):
    """Predict y value based on fit function type and parameters"""
    if fit_type == 0:  # Linear: y = ax + b
        a, b = params[0], params[1]
        return a * x + b
    elif fit_type == 1:  # Quadratic: y = ax² + bx + c
        a, b, c = params[0], params[1], params[2]
        return a * x * x + b * x + c
    elif fit_type == 2:  # Sine Wave: y = A*sin(f*x + p) + bias
        A, f, p, bias = params[0], params[1], params[2], params[3]
        return A * math.sin(f * x + p) + bias
    elif fit_type in [3, 4]:  # MLP models
        if mlp is not None:
            x_norm = np.array([[x / 100.0]])  # Normalize input
            y_pred = mlp.forward(x_norm)[0, 0]
            return y_pred * 100.0  # Denormalize output
    return 0

def compute_error(points, fit_type, params, mlp=None):
    mse = 0
    n = len(points)
    for (x, y) in points:
        pred = predict_value(x, fit_type, params, mlp)
        mse += (pred - y) ** 2
    return mse / n if n > 0 else 0

def compute_r_squared(points, fit_type, params, mlp=None):
    y_mean = sum(y for _, y in points) / len(points)
    ss_tot = sum((y - y_mean) ** 2 for _, y in points)
    ss_res = sum((y - predict_value(x, fit_type, params, mlp)) ** 2 for x, y in points)
    return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

def gradient_descent_step(points, fit_type, params, lr, mlp=None):
    """Gradient descent step for different fit functions"""
    n = len(points)
    
    if fit_type == 0:  # Linear: y = ax + b
        a, b = params[0], params[1]
        grad_a = 0
        grad_b = 0
        for (x, y) in points:
            pred = a * x + b
            error = pred - y
            grad_a += (2/n) * x * error
            grad_b += (2/n) * error
        return [a - lr * grad_a * 1.0, b - lr * grad_b * 100.0]
    
    elif fit_type == 1:  # Quadratic: y = ax² + bx + c
        a, b, c = params[0], params[1], params[2]
        grad_a = 0
        grad_b = 0
        grad_c = 0
        for (x, y) in points:
            pred = a * x * x + b * x + c
            error = pred - y
            grad_a += (2/n) * x * x * error
            grad_b += (2/n) * x * error
            grad_c += (2/n) * error
        return [a - lr * grad_a * 0.001, b - lr * grad_b * 1.0, c - lr * grad_c * 100.0]
    
    elif fit_type == 2:  # Sine Wave: y = A*sin(f*x + p) + bias
        A, f, p, bias = params[0], params[1], params[2], params[3]
        grad_A = 0
        grad_f = 0
        grad_p = 0
        grad_bias = 0
        for (x, y) in points:
            sin_arg = f * x + p
            pred = A * math.sin(sin_arg) + bias
            error = pred - y
            grad_A += (2/n) * math.sin(sin_arg) * error
            grad_f += (2/n) * A * math.cos(sin_arg) * x * error
            grad_p += (2/n) * A * math.cos(sin_arg) * error
            grad_bias += (2/n) * error
        return [A - lr * grad_A * 0.1, f - lr * grad_f * 0.001, p - lr * grad_p * 0.1, bias - lr * grad_bias * 100.0]
    
    elif fit_type in [3, 4]:  # MLP models
        if mlp is not None:
            # Prepare data for MLP training
            X = np.array([[x/100.0] for x, _ in points])  # Normalize inputs
            y = np.array([[y/100.0] for _, y in points])  # Normalize outputs
            
            # Train MLP for one step
            for i in range(len(points)):
                mlp.backward(X[i:i+1], y[i:i+1], lr)
            
            return params  # MLP parameters are stored internally
    
    return params

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

def update_slider_visibility(fit_type, sliders):
    """Update slider visibility based on fit function type"""
    # Hide all sliders first
    for slider in sliders:
        slider.visible = False
    
    if fit_type == 0:  # Linear: a, b
        sliders[0].visible = True   # a_linear
        sliders[1].visible = True   # b_linear
    elif fit_type == 1:  # Quadratic: a, b, c
        sliders[2].visible = True   # a_quad
        sliders[3].visible = True   # b_quad
        sliders[4].visible = True   # c_quad
    elif fit_type == 2:  # Sine: A, f, p, bias
        sliders[5].visible = True   # A_sine
        sliders[6].visible = True   # f_sine
        sliders[7].visible = True   # p_sine
        sliders[8].visible = True   # bias_sine
    elif fit_type == 3:  # MLP 1 hidden layer: only neuron count
        sliders[9].visible = True   # mlp1_neurons
    elif fit_type == 4:  # MLP 2 hidden layers: neuron counts
        sliders[10].visible = True  # mlp2_neurons1
        sliders[11].visible = True  # mlp2_neurons2

###############################################################################
# Estado inicial y elementos de UI
###############################################################################

# Estado inicial
dataset_type = 0
fit_type = 0  # 0: linear, 1: quadratic, 2: sine, 3: MLP1, 4: MLP2
points = generate_points(dataset_type, noise_level=8)
cost_history = []
max_cost_history = 100000

# Parámetros del modelo (inicializados según el tipo de ajuste)
params = [0.0, 0.0, 0.0, 0.0]  # [a, b, c, A, f, p, bias] - some will be unused

# MLP models
mlp1 = None
mlp2 = None

# Elementos de UI - Reorganized layout within control panel
# First line: step, play, reset, auto_fit, learning_rate, noise_level, steps/s, points
button_step = ModernButton(CONTROL_PANEL_LEFT + 20, CONTROL_PANEL_TOP + 20, 100, 40, "Step", COLORS['success'])
button_play = ModernButton(CONTROL_PANEL_LEFT + 140, CONTROL_PANEL_TOP + 20, 100, 40, "Play", COLORS['info'])
button_reset = ModernButton(CONTROL_PANEL_LEFT + 260, CONTROL_PANEL_TOP + 20, 100, 40, "Reset", COLORS['warning'])
button_auto_fit = ModernButton(CONTROL_PANEL_LEFT + 380, CONTROL_PANEL_TOP + 20, 100, 40, "Auto Fit", COLORS['danger'])

# Learning rate and noise level sliders (first line)
slider_lr = ModernSlider(CONTROL_PANEL_LEFT + 500, CONTROL_PANEL_TOP + 40, 180, 20, 0.000001, 0.001, 0.0001, "Learning Rate", COLORS['warning'])
slider_noise = ModernSlider(CONTROL_PANEL_LEFT + 700, CONTROL_PANEL_TOP + 40, 180, 20, 0, 20, 8, "Noise Level", COLORS['danger'])

# Steps/sec and points textboxes (first line)
textbox_steps = ModernTextBox(CONTROL_PANEL_LEFT + 900, CONTROL_PANEL_TOP + 30, 80, 30, "4", "Steps/sec")
textbox_n_points = ModernTextBox(CONTROL_PANEL_LEFT + 1000, CONTROL_PANEL_TOP + 30, 80, 30, "100", "Points")

# Second line: Dataset buttons
dataset_buttons = [
    ModernButton(CONTROL_PANEL_LEFT + 20, CONTROL_PANEL_TOP + 80, 100, 40, "Linear", COLORS['primary']),
    ModernButton(CONTROL_PANEL_LEFT + 140, CONTROL_PANEL_TOP + 80, 100, 40, "Quadratic", COLORS['secondary']),
    ModernButton(CONTROL_PANEL_LEFT + 260, CONTROL_PANEL_TOP + 80, 100, 40, "Sine", COLORS['info']),
    ModernButton(CONTROL_PANEL_LEFT + 380, CONTROL_PANEL_TOP + 80, 100, 40, "Random", COLORS['warning'])
]

# Third line: Fit function buttons
fit_buttons = [
    ModernButton(CONTROL_PANEL_LEFT + 20, CONTROL_PANEL_TOP + 140, 100, 40, "Linear", COLORS['primary']),
    ModernButton(CONTROL_PANEL_LEFT + 140, CONTROL_PANEL_TOP + 140, 100, 40, "Quadratic", COLORS['secondary']),
    ModernButton(CONTROL_PANEL_LEFT + 260, CONTROL_PANEL_TOP + 140, 100, 40, "Sine", COLORS['info']),
    ModernButton(CONTROL_PANEL_LEFT + 380, CONTROL_PANEL_TOP + 140, 100, 40, "MLP-1", COLORS['success']),
    ModernButton(CONTROL_PANEL_LEFT + 500, CONTROL_PANEL_TOP + 140, 100, 40, "MLP-2", COLORS['danger'])
]

# Fifth line: Sliders para diferentes tipos de ajuste (within control panel)
# Linear fit sliders
slider_a_linear = ModernSlider(CONTROL_PANEL_LEFT + 20, CONTROL_PANEL_TOP + 220, 200, 20, -2, 2, 0.0, "Slope (a)", COLORS['primary'])
slider_b_linear = ModernSlider(CONTROL_PANEL_LEFT + 240, CONTROL_PANEL_TOP + 220, 200, 20, -50, 150, 0.0, "Intercept (b)", COLORS['secondary'])

# Quadratic fit sliders
slider_a_quad = ModernSlider(CONTROL_PANEL_LEFT + 20, CONTROL_PANEL_TOP + 220, 200, 20, -0.1, 0.1, 0.0, "a (x²)", COLORS['primary'])
slider_b_quad = ModernSlider(CONTROL_PANEL_LEFT + 240, CONTROL_PANEL_TOP + 220, 200, 20, -2.0, 2.0, 0.0, "b (x)", COLORS['secondary'])
slider_c_quad = ModernSlider(CONTROL_PANEL_LEFT + 460, CONTROL_PANEL_TOP + 220, 200, 20, -50.0, 150.0, 0.0, "c (const)", COLORS['info'])

# Sine fit sliders
slider_A_sine = ModernSlider(CONTROL_PANEL_LEFT + 20, CONTROL_PANEL_TOP + 220, 200, 20, -50, 50, 30.0, "Amplitude (A)", COLORS['primary'])
slider_f_sine = ModernSlider(CONTROL_PANEL_LEFT + 240, CONTROL_PANEL_TOP + 220, 200, 20, 0, 0.5, 0.1, "Frequency (f)", COLORS['secondary'])
slider_p_sine = ModernSlider(CONTROL_PANEL_LEFT + 460, CONTROL_PANEL_TOP + 220, 200, 20, -50, 50, 0.0, "Phase (p)", COLORS['info'])
slider_bias_sine = ModernSlider(CONTROL_PANEL_LEFT + 680, CONTROL_PANEL_TOP + 220, 200, 20, -100, 100, 50.0, "Bias", COLORS['warning'])

# MLP sliders
slider_mlp1_neurons = ModernSlider(CONTROL_PANEL_LEFT + 20, CONTROL_PANEL_TOP + 220, 200, 20, 2, 10, 5, "MLP-1 Neurons", COLORS['success'])
slider_mlp2_neurons1 = ModernSlider(CONTROL_PANEL_LEFT + 20, CONTROL_PANEL_TOP + 220, 200, 20, 2, 10, 5, "MLP-2 Layer 1", COLORS['success'])
slider_mlp2_neurons2 = ModernSlider(CONTROL_PANEL_LEFT + 240, CONTROL_PANEL_TOP + 220, 200, 20, 2, 10, 5, "MLP-2 Layer 2", COLORS['danger'])

# Log scale button for cost function (positioned outside control panel)
button_log_scale = ModernButton(970, 10, 150, 30, "Linear Scale", COLORS['secondary'])

# Variables para el modo automático
play_mode = False
accumulated_time = 0.0
log_scale = False
previous_noise_level = 8

# Fuentes
title_font = pygame.font.SysFont('Arial', 24, bold=True)
subtitle_font = pygame.font.SysFont('Arial', 18, bold=True)
text_font = pygame.font.SysFont('Arial', 14)

# Initialize slider visibility
all_sliders = [slider_a_linear, slider_b_linear, slider_a_quad, slider_b_quad, slider_c_quad, 
               slider_A_sine, slider_f_sine, slider_p_sine, slider_bias_sine,
               slider_mlp1_neurons, slider_mlp2_neurons1, slider_mlp2_neurons2]
update_slider_visibility(fit_type, all_sliders)

# Set initial button states
dataset_buttons[0].selected = True
fit_buttons[0].selected = True

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

        # Handle dataset buttons
        for i, button in enumerate(dataset_buttons):
            if button.handle_event(event):
                # Deselect all dataset buttons
                for b in dataset_buttons:
                    b.selected = False
                # Select current button
                button.selected = True
                dataset_type = i
                n_points = int(textbox_n_points.get_value())
                noise_level = slider_noise.value
                points = generate_points(dataset_type, n_points, noise_level)
                cost_history = []

        # Handle fit function buttons
        for i, button in enumerate(fit_buttons):
            if button.handle_event(event):
                # Deselect all fit buttons
                for b in fit_buttons:
                    b.selected = False
                # Select current button
                button.selected = True
                fit_type = i
                update_slider_visibility(fit_type, all_sliders)
                cost_history = []
                
                # Initialize MLP models if needed
                if fit_type == 3:  # MLP-1
                    neurons = int(slider_mlp1_neurons.value)
                    mlp1 = MLP(input_size=1, hidden_sizes=[neurons], output_size=1)
                elif fit_type == 4:  # MLP-2
                    neurons1 = int(slider_mlp2_neurons1.value)
                    neurons2 = int(slider_mlp2_neurons2.value)
                    mlp2 = MLP(input_size=1, hidden_sizes=[neurons1, neurons2], output_size=1)
        
        # Handle all sliders
        for slider in all_sliders + [slider_lr, slider_noise]:
            slider.handle_event(event)
        
        textbox_steps.handle_event(event)
        textbox_n_points.handle_event(event)

        # Botones
        if button_step.handle_event(event):
            current_lr = slider_lr.value
            if fit_type in [3, 4]:  # MLP models
                new_params = gradient_descent_step(points, fit_type, params, current_lr, mlp1 if fit_type == 3 else mlp2)
            else:
                new_params = gradient_descent_step(points, fit_type, params, current_lr)
            params = new_params
            
            # Update sliders with new values for non-MLP models
            if fit_type == 0:  # Linear
                slider_a_linear.set_value(params[0])
                slider_b_linear.set_value(params[1])
            elif fit_type == 1:  # Quadratic
                slider_a_quad.set_value(params[0])
                slider_b_quad.set_value(params[1])
                slider_c_quad.set_value(params[2])
            elif fit_type == 2:  # Sine
                slider_A_sine.set_value(params[0])
                slider_f_sine.set_value(params[1])
                slider_p_sine.set_value(params[2])
                slider_bias_sine.set_value(params[3])
            
            cost_history.append(compute_error(points, fit_type, params, mlp1 if fit_type == 3 else mlp2))
            if len(cost_history) > max_cost_history:
                cost_history.pop(0)

        if button_play.handle_event(event):
            play_mode = not play_mode
            button_play.text = "Pause" if play_mode else "Play"

        if button_reset.handle_event(event):
            # Reset parameters based on fit type
            if fit_type == 0:  # Linear
                params = [0.0, 0.0, 0.0, 0.0]
                slider_a_linear.set_value(0.0)
                slider_b_linear.set_value(0.0)
            elif fit_type == 1:  # Quadratic
                params = [0.0, 0.0, 0.0, 0.0]
                slider_a_quad.set_value(0.0)
                slider_b_quad.set_value(0.0)
                slider_c_quad.set_value(0.0)
            elif fit_type == 2:  # Sine
                params = [30.0, 0.1, 0.0, 50.0]
                slider_A_sine.set_value(30.0)
                slider_f_sine.set_value(0.1)
                slider_p_sine.set_value(0.0)
                slider_bias_sine.set_value(50.0)
            elif fit_type == 3:  # MLP-1
                neurons = int(slider_mlp1_neurons.value)
                mlp1 = MLP(input_size=1, hidden_sizes=[neurons], output_size=1)
            elif fit_type == 4:  # MLP-2
                neurons1 = int(slider_mlp2_neurons1.value)
                neurons2 = int(slider_mlp2_neurons2.value)
                mlp2 = MLP(input_size=1, hidden_sizes=[neurons1, neurons2], output_size=1)
            cost_history = []

        if button_auto_fit.handle_event(event):
            # Auto fit with 100 steps
            for _ in range(100):
                current_lr = slider_lr.value
                if fit_type in [3, 4]:  # MLP models
                    new_params = gradient_descent_step(points, fit_type, params, current_lr, mlp1 if fit_type == 3 else mlp2)
                else:
                    new_params = gradient_descent_step(points, fit_type, params, current_lr)
                params = new_params
                cost_history.append(compute_error(points, fit_type, params, mlp1 if fit_type == 3 else mlp2))
                if len(cost_history) > max_cost_history:
                    cost_history.pop(0)

        if button_log_scale.handle_event(event):
            log_scale = not log_scale
            button_log_scale.text = "Log Scale" if log_scale else "Linear Scale"

    # Update MLP models when neuron counts change
    if fit_type == 3:  # MLP-1
        neurons = int(slider_mlp1_neurons.value)
        if mlp1 is None or mlp1.hidden_sizes[0] != neurons:
            mlp1 = MLP(input_size=1, hidden_sizes=[neurons], output_size=1)
            cost_history = []
    elif fit_type == 4:  # MLP-2
        neurons1 = int(slider_mlp2_neurons1.value)
        neurons2 = int(slider_mlp2_neurons2.value)
        if mlp2 is None or mlp2.hidden_sizes != [neurons1, neurons2]:
            mlp2 = MLP(input_size=1, hidden_sizes=[neurons1, neurons2], output_size=1)
            cost_history = []

    # Actualización automática en modo play
    if play_mode:
        accumulated_time += dt
        steps_per_sec = textbox_steps.get_value()
        while accumulated_time >= 1.0 / steps_per_sec:
            current_lr = slider_lr.value
            if fit_type in [3, 4]:  # MLP models
                new_params = gradient_descent_step(points, fit_type, params, current_lr, mlp1 if fit_type == 3 else mlp2)
            else:
                new_params = gradient_descent_step(points, fit_type, params, current_lr)
            params = new_params
            
            # Update sliders with new parameter values for non-MLP models
            if fit_type == 0:  # Linear
                slider_a_linear.set_value(params[0])
                slider_b_linear.set_value(params[1])
            elif fit_type == 1:  # Quadratic
                slider_a_quad.set_value(params[0])
                slider_b_quad.set_value(params[1])
                slider_c_quad.set_value(params[2])
            elif fit_type == 2:  # Sine
                slider_A_sine.set_value(params[0])
                slider_f_sine.set_value(params[1])
                slider_p_sine.set_value(params[2])
                slider_bias_sine.set_value(params[3])
            
            cost_history.append(compute_error(points, fit_type, params, mlp1 if fit_type == 3 else mlp2))
            if len(cost_history) > max_cost_history:
                cost_history.pop(0)
            accumulated_time -= 1.0 / steps_per_sec

    # Actualizar parámetros desde sliders para modelos no-MLP
    if fit_type == 0:  # Linear
        params = [slider_a_linear.value, slider_b_linear.value, 0.0, 0.0]
    elif fit_type == 1:  # Quadratic
        params = [slider_a_quad.value, slider_b_quad.value, slider_c_quad.value, 0.0]
    elif fit_type == 2:  # Sine
        params = [slider_A_sine.value, slider_f_sine.value, slider_p_sine.value, slider_bias_sine.value]
    
    # Check if noise level changed and regenerate dataset
    current_noise_level = slider_noise.value
    if abs(current_noise_level - previous_noise_level) > 0.1:
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

    # Dibujar panel de control
    pygame.draw.rect(screen, COLORS['white'], CONTROL_PANEL_RECT, border_radius=10)
    pygame.draw.rect(screen, COLORS['dark'], CONTROL_PANEL_RECT, 2, border_radius=10)
    
    # Título del panel de control
    control_title = subtitle_font.render("Control Panel", True, COLORS['dark'])
    screen.blit(control_title, (CONTROL_PANEL_LEFT + 10, CONTROL_PANEL_TOP - 30))

    # Dibujar puntos de datos
    for (x, y) in points:
        pos = data_to_screen(x, y, PLOT_RECT)
        pygame.draw.circle(screen, COLORS['data_points'], pos, 4)
        pygame.draw.circle(screen, COLORS['white'], pos, 2)

    # Dibujar función real sin ruido (solo para datasets no aleatorios)
    if dataset_type != 3:  # Not random dataset
        real_curve_points = []
        for x in range(0, 101, 2):
            y = get_real_function_value(x, dataset_type)
            if y is not None and 0 <= y <= 100:  # Only draw points within the plot area
                real_curve_points.append(data_to_screen(x, y, PLOT_RECT))
        
        if len(real_curve_points) > 1:
            # Draw dashed line for real function
            for i in range(0, len(real_curve_points) - 1, 4):  # Skip every 4th point for dashed effect
                if i + 1 < len(real_curve_points):
                    pygame.draw.line(screen, COLORS['real_function'], 
                                   real_curve_points[i], real_curve_points[i + 1], 2)

    # Dibujar línea de ajuste
    if len(points) > 0:
        curve_points = []
        for x in range(0, 101, 2):
            y = predict_value(x, fit_type, params, mlp1 if fit_type == 3 else mlp2)
            if 0 <= y <= 100:  # Only draw points within the plot area
                curve_points.append(data_to_screen(x, y, PLOT_RECT))
        
        if len(curve_points) > 1:
            pygame.draw.lines(screen, COLORS['regression_line'], False, curve_points, 3)

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
    for button in dataset_buttons + fit_buttons:
        button.draw(screen)
    
    button_step.draw(screen)
    button_play.draw(screen)
    button_reset.draw(screen)
    button_auto_fit.draw(screen)
    button_log_scale.draw(screen)
    
    # Draw sliders (only visible ones)
    for slider in all_sliders + [slider_lr, slider_noise]:
        slider.draw(screen)
    
    textbox_steps.draw(screen)
    textbox_n_points.draw(screen)

    # Mostrar estadísticas
    error_value = compute_error(points, fit_type, params, mlp1 if fit_type == 3 else mlp2)
    r_squared = compute_r_squared(points, fit_type, params, mlp1 if fit_type == 3 else mlp2)
    
    # Get function names and current parameters
    dataset_names = ["Linear", "Quadratic", "Sine Wave", "Random"]
    fit_names = ["Linear", "Quadratic", "Sine Wave", "MLP-1 Hidden", "MLP-2 Hidden"]
    current_dataset_name = dataset_names[dataset_type]
    current_fit_name = fit_names[fit_type]
    
    # Real functions used to generate datasets
    real_functions = [
        "y = 0.7x + 20 + N(0,σ²)",  # Linear
        "y = 0.015(x-30)² + 10 + N(0,σ²)",  # Quadratic
        "y = 30sin(0.1x) + 50 + N(0,σ²)",  # Sine Wave
        "y = random(0,100)"  # Random
    ]
    current_real_function = real_functions[dataset_type]
    
    # Panel de estadísticas
    stats_rect = pygame.Rect(620, 300, 500, 180)
    pygame.draw.rect(screen, COLORS['white'], stats_rect, border_radius=10)
    pygame.draw.rect(screen, COLORS['dark'], stats_rect, 2, border_radius=10)
    
    stats_title = subtitle_font.render("Statistics", True, COLORS['dark'])
    screen.blit(stats_title, (stats_rect.x + 10, stats_rect.y + 10))
    
    # Dataset and real function info
    dataset_text = text_font.render(f"Dataset: {current_dataset_name}", True, COLORS['dark'])
    screen.blit(dataset_text, (stats_rect.x + 10, stats_rect.y + 35))
    
    real_function_text = text_font.render(f"Real: {current_real_function}", True, COLORS['success'])
    screen.blit(real_function_text, (stats_rect.x + 10, stats_rect.y + 55))
    
    fit_text = text_font.render(f"Fit Function: {current_fit_name}", True, COLORS['info'])
    screen.blit(fit_text, (stats_rect.x + 10, stats_rect.y + 75))
    
    # Current parameters
    if fit_type == 0:  # Linear
        params_text = text_font.render(f"y = {params[0]:.4f}x + {params[1]:.4f}", True, COLORS['dark'])
    elif fit_type == 1:  # Quadratic
        params_text = text_font.render(f"y = {params[0]:.4f}x² + {params[1]:.4f}x + {params[2]:.4f}", True, COLORS['dark'])
    elif fit_type == 2:  # Sine
        params_text = text_font.render(f"y = {params[0]:.2f}sin({params[1]:.3f}x + {params[2]:.2f}) + {params[3]:.2f}", True, COLORS['dark'])
    elif fit_type == 3:  # MLP-1
        neurons = int(slider_mlp1_neurons.value)
        params_text = text_font.render(f"MLP-1: 1 → {neurons} → 1", True, COLORS['dark'])
    elif fit_type == 4:  # MLP-2
        neurons1 = int(slider_mlp2_neurons1.value)
        neurons2 = int(slider_mlp2_neurons2.value)
        params_text = text_font.render(f"MLP-2: 1 → {neurons1} → {neurons2} → 1", True, COLORS['dark'])
    
    screen.blit(params_text, (stats_rect.x + 10, stats_rect.y + 95))
    
    # Separator line
    pygame.draw.line(screen, COLORS['gray_300'], (stats_rect.x + 10, stats_rect.y + 115), 
                     (stats_rect.x + stats_rect.width - 10, stats_rect.y + 115), 1)
    
    mse_text = text_font.render(f"MSE: {error_value:.4f}", True, COLORS['dark'])
    screen.blit(mse_text, (stats_rect.x + 10, stats_rect.y + 125))
    
    r2_text = text_font.render(f"R²: {r_squared:.4f}", True, COLORS['dark'])
    screen.blit(r2_text, (stats_rect.x + 10, stats_rect.y + 145))
    
    steps_text = text_font.render(f"Steps: {len(cost_history)}", True, COLORS['dark'])
    screen.blit(steps_text, (stats_rect.x + 10, stats_rect.y + 165))

    # Títulos
    main_title = title_font.render("Polynomial & MLP Function Fitting Visualization", True, COLORS['dark'])
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
        screen.blit(y_axis_label, y_axis_rect)

    pygame.display.flip()

pygame.quit()
sys.exit()
