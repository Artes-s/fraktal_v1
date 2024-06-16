'''

    ⠀⠀⠀⢀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⡀
    ⠀⠀⠀⠀⡏⢢⡁⠂⠤⣀⣀⣀⣀⣀ ⠤⠐⢈⡔⢹
    ⠀⠀⠀⠀⢿⡀⠙⠆⠀⠉⠀⠀⠀⠀⠉⠀⠰⠋⢀⡿
    ⠀⠀⠀⠀⠈⢷⠄⠀⠀⠀⠀⠀⸸⠀⠀ ⠀⠀⠀⠠⡾⠁
    ⠀⠀⠀⠀⠀⠀⡏⠀⠀⠀⠀⠀⠀⠀ ⠀ ⠀⠀⠀⢹
    ⣰⠊⠉⠉⠉⡇⠀⠢⣤⣄⠀⠀ ⣠⣤⠔⠀⢸
    ⠙⠓⠒⢦⠀⠱⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠎
    ⠀⠀⠀⠀⡇⠀⠀⠏⠑⠒⠀⠉⠀⠒⠊⠹
    ⡎⠉⢹⠀⠙⡶⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢦⠀⠀⡏⠉⢱
    ⢧⡈⠛⠉⠉⠀⠀⣠⠀⠀⠀⠀⠀⠀⠀⠀⣄⠀⠉⠉⠋⢁⡼
    ⠀⢉⣿⠖⠚⠛⢋⢀⠀⠀⠀⠀⠀⠀⠀⡀⡙⠛⠓⠲⣿⣄
    ⠀⢸⡇⠀⠀⠀⡞⠁⠈⡃⠀⠀⠀⠀⢘⠁⠈⢳⠀⠀⠀⢸⡇
    ⠀⠈⢷⣄⠀⠀⠙⠦⠌⠑⠢⠤⠔⠊⠁⢠⠎⠀⠀⣠⡾⠁
    ⠀⠀⠀⠈⠛⠲⠤⣤⣀⣀⣀⣀⣠⣤⣚⣡⠤⠖⠛⠁
    ⠀⠛⠲⣤⣤⣤⣤⣀Artes⣀⣤⣤⣤⣤⠖⠛⠁
    Artes product. fraktal_v1
    Version 1.0

'''

"""
# INFO
Project Info: Фрактализация с приближением по определенным предустановкам. 
Множество констант для индивидуальной настройки приближения. Тест версия 1.0

Предусмотрено: 
- изменение скорости зумирования
- изменение шага зумирования
- изменение цветового приближения по rx, ry
- изменение тикрейта частоты кадров
- изменение цветовой палитны
- +-random для плавности
- on/off DEBUG 

Предостережение: 
- скрипт загружает CPU (не cuda память GPU)
- начните свои тесты с тикрейтом 10

Почему:
Потому что я могу блять ( ͡° ͜ʖ ͡°)
"""

import pygame
import numpy as np
from numba import jit, prange
import matplotlib.pyplot as plt
import psutil
import time
import random

# Настройки окна
WIDTH, HEIGHT = 600, 600
LOW_RES_WIDTH, LOW_RES_HEIGHT = 300, 300  # уменьшенное разрешение для расчетов (всегда должно быть в 2 раза меньше чем оригинальное разрешение)
INITIAL_MAX_ITER = 256

DEBUG = True

"""
# INFO
Пред настройки для проникания *внутрь фрактала*
zoom_factor = 0.93
rx = 245 #224
ry = 256 #225
zeroFix = 1
step = 0.05 + random.uniform(-0.01, 0.01)  # шаг изменения центра для плавности
# пример https://i.imgur.com/1ovgMHR.gif
"""

"""
# INFO
Пред настройки для смещения в лево *фрактала*
zoom_factor = 0.98
rx = 245 #224
ry = 256 #225
zeroFix = 1
step = 0.05 + random.uniform(-0.01, 0.01)  # шаг изменения центра для плавности
# пример https://i.imgur.com/ZjHBPCf.gif
"""

# Глобальные переменные движения
zoom_factor = 0.98 #0.93
rx = 225 #224
ry = 256 #225
zeroFix = 1
step = 0.05 + random.uniform(-0.01, 0.01)  # шаг изменения центра для плавности #0.05

# чем больше тикрейт, тем быстрее зумирование, но и нагрузка на процессор больше.
tickrate = 30 #max=40, min=1, def=10


# Инициализация Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Fractal Zoom")
clock = pygame.time.Clock()

# Цветовая схема
"""
colors = [pygame.Color(0, 0, 0)] + [pygame.Color(*[int(c * 255) for c in plt.cm.plasma(i / 256)[:3]]) for i in range(1, 256)]
color_array = np.array([[color.r, color.g, color.b] for color in colors])

colors = [pygame.Color(0, 0, 0)] + [pygame.Color(*[int(c * 255) for c in plt.cm.viridis(i / 256)[:3]]) for i in range(1, 256)]
color_array = np.array([[color.r, color.g, color.b] for color in colors])
"""

def generate_random_colors():
    colors = [pygame.Color(0, 0, 0)]
    for _ in range(1, 256):
        color = pygame.Color(random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        colors.append(color)
    return colors

def generate_rainbow_colors():
    colors = [pygame.Color(0, 0, 0)] + [pygame.Color(*[int(c * 255) for c in plt.cm.hsv(i / 256)[:3]]) for i in range(1, 256)]
    return colors

def update_color_array(colors):
    return np.array([[color.r, color.g, color.b] for color in colors])

def generate_plasma_colors():
    colors = [pygame.Color(0, 0, 0)] + [pygame.Color(*[int(c * 255) for c in plt.cm.plasma(i / 256)[:3]]) for i in
                                        range(1, 256)]
    #color_array = np.array([[color.r, color.g, color.b] for color in colors])
    return colors

def generate_viridis_colors():
    colors = [pygame.Color(0, 0, 0)] + [pygame.Color(*[int(c * 255) for c in plt.cm.viridis(i / 256)[:3]]) for i in
                                        range(1, 256)]
    #color_array = np.array([[color.r, color.g, color.b] for color in colors])
    return colors

"""
test_1
test_2
random
rainbow
"""

# Выбор палитры
palette_mode = "test_1"  # можно переключать на ^

"""
# INFO
# изменение в функцие (i / 256)[:3], изменения с 256 до 10 приводят к изменению цветовой палитны и насыщености ветков ветвей фрейма (min=1, max=256)
"""

if palette_mode == "test_1":
    colors = generate_plasma_colors()

if palette_mode == "test_2":
    colors = generate_viridis_colors()

if palette_mode == "random":
    colors = generate_random_colors()

if palette_mode == "rainbow":
    colors = generate_rainbow_colors()

color_array = update_color_array(colors)

@jit(nopython=True)
def mandelbrot(c, max_iter):
    z = c
    for n in range(max_iter):
        if abs(z) > 2:
            return n
        z = z * z + c
    return max_iter

@jit(nopython=True, parallel=True)
def create_fractal(min_x, max_x, min_y, max_y, image, max_iter):
    height, width = image.shape
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height
    for y in prange(height):
        imag = min_y + y * pixel_size_y
        for x in prange(width):
            real = min_x + x * pixel_size_x
            color = mandelbrot(complex(real, imag), max_iter)
            image[y, x] = color

# Начальные параметры фрактала
initial_min_x, initial_max_x = -2.0, 1.0
initial_min_y, initial_max_y = -1.5, 1.5

# Глобальные переменные
cycle_count = 0
zoom_count = 0

# Функция для сброса параметров фрактала
def reset_parameters(msg=None):
    global min_x, max_x, min_y, max_y, max_iter, zoom_count, rx, ry, color_array, zeroFix, cycle_count
    if msg is not None:
        print(msg)
    #print('max zoom: ', {zoom_count})
    min_x, max_x = initial_min_x, initial_max_x
    min_y, max_y = initial_min_y, initial_max_y
    max_iter = INITIAL_MAX_ITER
    zoom_count = 0
    rx = rx + random.randint(-10, 10) * zeroFix
    ry = ry + random.randint(-10, 10) * zeroFix
    color_array = update_color_array(colors)
    cycle_count += 1

# Инициализация параметров фрактала
reset_parameters()

def find_yellow_pixel(image):
    global rx, ry
    height, width = image.shape
    target_color_range = range(rx, ry)  # диапазон индексов для желтых оттенков
    for y in range(height):
        for x in range(width):
            color_value = image[y, x]
            if color_value in target_color_range:  # если пиксель желтый
                return x, y
    return width // 2, height // 2  # если желтых пикселей нет, возвращаем центр

def calculate_new_center(min_x, max_x, min_y, max_y, target_x, target_y, step=0.1):
    width = max_x - min_x
    height = max_y - min_y
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    target_real = min_x + (target_x / LOW_RES_WIDTH) * width
    target_imag = min_y + (target_y / LOW_RES_HEIGHT) * height

    new_center_x = center_x + step * (target_real - center_x)
    new_center_y = center_y + step * (target_imag - center_y)

    return new_center_x, new_center_y

def render_text_with_shadow(screen, text, font, pos, text_color, shadow_color, shadow_offset=(2, 2)):
    shadow = font.render(text, True, shadow_color)
    screen.blit(shadow, (pos[0] + shadow_offset[0], pos[1] + shadow_offset[1]))
    text_surface = font.render(text, True, text_color)
    screen.blit(text_surface, pos)

running = True
target_x, target_y = LOW_RES_WIDTH // 2, LOW_RES_HEIGHT // 2  # начальная цель - центр

while running:
    start_time = time.time()

    # Увеличиваем количество итераций по мере увеличения масштаба
    scale_factor = (2 - (max_x - min_x) / 3)
    max_iter = int(INITIAL_MAX_ITER * scale_factor)

    # Уменьшенное изображение для расчетов
    low_res_image = np.zeros((LOW_RES_HEIGHT, LOW_RES_WIDTH), dtype=np.int32)
    create_fractal(min_x, max_x, min_y, max_y, low_res_image, max_iter)

    # Проверяем, если вся область одного цвета или scale_factor равен 2, сбрасываем параметры фрактала
    unique_colors = np.unique(low_res_image)
    #if len(unique_colors) == 1 or scale_factor >= 2: #old
    if len(unique_colors) == 1 or scale_factor >= 1.9999999999999996:
        reset_parameters('scale_factor > 1.9{16}6')
        continue

    # Найти желтый пиксель для приближения
    new_target_x, new_target_y = find_yellow_pixel(low_res_image)

    if new_target_x != target_x or new_target_y != target_y:
        target_x, target_y = new_target_x, new_target_y

    # Масштабирование изображения до полного разрешения
    scaled_image = np.kron(low_res_image, np.ones((HEIGHT // LOW_RES_HEIGHT, WIDTH // LOW_RES_WIDTH)))
    scaled_image = scaled_image.astype(np.int32)  # Преобразуем в целочисленный тип

    # Нормализация значений
    scaled_image[scaled_image >= len(colors)] = len(colors) - 1

    # Преобразование в RGB массив
    rgb_image = color_array[scaled_image]

    pygame.surfarray.blit_array(screen, rgb_image)

    # Обновление параметров для плавного приближения к новому пикселю
    new_center_x, new_center_y = calculate_new_center(min_x, max_x, min_y, max_y, target_x, target_y, step)
    width = (max_x - min_x) * zoom_factor
    height = (max_y - min_y) * zoom_factor
    min_x = new_center_x - width / 2
    max_x = new_center_x + width / 2
    min_y = new_center_y - height / 2
    max_y = new_center_y + height / 2

    # Увеличение счетчика приближений
    zoom_count += 1

    # Отображение FPS, загрузки CPU и счетчиков
    fps = clock.get_fps()
    cpu_usage = psutil.cpu_percent()

    # Отображение текущей цели
    screen_x = int((target_x / LOW_RES_WIDTH) * WIDTH)
    screen_y = int((target_y / LOW_RES_HEIGHT) * HEIGHT)

    if DEBUG == True:
        font = pygame.font.SysFont("Arial", 18)
        render_text_with_shadow(screen, f"FPS: {fps:.2f}", font, (10, 10), pygame.Color("white"), pygame.Color("black"))
        render_text_with_shadow(screen, f"CPU: {cpu_usage:.2f}%", font, (10, 30), pygame.Color("white"), pygame.Color("black"))
        render_text_with_shadow(screen, f"Zooms: {zoom_count}", font, (10, 50), pygame.Color("white"), pygame.Color("black"))
        render_text_with_shadow(screen, f"Cycles: {cycle_count}", font, (10, 70), pygame.Color("white"), pygame.Color("black"))
        render_text_with_shadow(screen, f"Scale: {scale_factor}", font, (10, 90), pygame.Color("white"), pygame.Color("black"))
        render_text_with_shadow(screen, f"rx ry: {rx},{ry}", font, (10, 110), pygame.Color("white"), pygame.Color("black"))
        pygame.draw.circle(screen, (255, 0, 0), (screen_x, screen_y), 5)

        render_text_with_shadow(screen, f"x y: {screen_x},{screen_y}", font, (10, 130), pygame.Color("white"), pygame.Color("black"))

    if (screen_x == 0 and screen_y == 0):
        reset_parameters('xy=0')
        continue

    if (screen_x == 300 and screen_y == 300):
        #reset_parameters('xy=300')
        continue

    if (screen_x >= 550 and screen_y >= 500):
        reset_parameters('xy>=550')
        continue

    pygame.display.flip()
    clock.tick(tickrate)

    frame_time = time.time() - start_time

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
