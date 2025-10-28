import pygame
import heapq
import itertools
import math

# ==============================
# CONFIGURACIÓN GENERAL
# ==============================
ROWS, COLS = 21, 21   # Cambia aquí: 7x7 o 20x20, se adapta automáticamente
TILE_SIZE = 40      # Tamaño visual de cada cuadro (px)
BIAS = 1.001        # Sesgo suave para romper empates en f (hacia la meta)

# Colores
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED   = (255, 0, 0)
BLUE  = (0, 0, 255)
GRAY  = (200, 200, 200)

# Inicializa Pygame
pygame.init()
WIN = pygame.display.set_mode((COLS * TILE_SIZE, ROWS * TILE_SIZE))
pygame.display.set_caption("Algoritmo A* (Manhattan con numeración dinámica)")
FONT = pygame.font.SysFont("Arial", max(12, TILE_SIZE // 4))

# ==============================
# CLASE NODO
# ==============================
class Nodo:
    def __init__(self, fila, col, numero):
        self.fila = fila
        self.col = col
        self.numero = numero  # número secuencial
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
        self.padre = None
        self.es_obstaculo = False

    def __lt__(self, otro):
        return self.f < otro.f

    def dibujar(self, color):
        # Celda
        pygame.draw.rect(
            WIN, color,
            (self.col * TILE_SIZE, self.fila * TILE_SIZE, TILE_SIZE, TILE_SIZE)
        )
        # Borde
        pygame.draw.rect(
            WIN, BLACK,
            (self.col * TILE_SIZE, self.fila * TILE_SIZE, TILE_SIZE, TILE_SIZE), 1
        )
        # Número (arriba-derecha)
        texto = FONT.render(str(self.numero), True, BLACK)
        text_rect = texto.get_rect(
            topright=(self.col * TILE_SIZE + TILE_SIZE - 4, self.fila * TILE_SIZE + 2)
        )
        WIN.blit(texto, text_rect)

# ==============================
# FUNCIONES DEL ALGORITMO
# ==============================
def heuristica(nodo, objetivo):
    # Distancia Manhattan (solo 4 direcciones)
    return 10 * (abs(nodo.fila - objetivo.fila) + abs(nodo.col - objetivo.col))

def reconstruir_camino(nodo):
    camino = []
    while nodo:
        camino.append(nodo)
        nodo = nodo.padre
    return camino[::-1]

def a_estrella(grid, inicio, objetivo):
    # Reiniciar valores por seguridad
    for fila in grid:
        for n in fila:
            n.g = float('inf')
            n.h = 0
            n.f = float('inf')
            n.padre = None

    # Inicialización
    inicio.g = 0
    inicio.h = heuristica(inicio, objetivo)
    inicio.f = inicio.g + inicio.h * BIAS  # tie-break hacia la meta

    tick = itertools.count()
    open_heap = []
    heapq.heappush(open_heap, (inicio.f, inicio.h, next(tick), inicio))

    open_set_nums = {inicio.numero}   # para mostrar LA
    best_g = {inicio.numero: 0}       # mejor g conocido por nodo (por número)
    closed_nums = set()               # LC para dibujar y mostrar

    movimientos = [(-1,0), (1,0), (0,-1), (0,1)]  # U, D, L, R

    print("\n🧩 Iniciando búsqueda A* (Manhattan)")
    print(f"Punto inicial: {inicio.numero}")
    print(f"Punto final:   {objetivo.numero}\n")

    while open_heap:
        _, _, _, actual = heapq.heappop(open_heap)

        # Saltar entradas obsoletas o ya cerradas
        if actual.numero in closed_nums:
            continue
        if best_g.get(actual.numero, math.inf) < actual.g:
            continue

        # Mover de LA a LC
        open_set_nums.discard(actual.numero)
        closed_nums.add(actual.numero)

        print(f"➡️ Explorando nodo: {actual.numero}  f={actual.f:.1f} g={actual.g:.1f} h={actual.h:.1f}")
        print("📘 LA (Abierta):", sorted(open_set_nums))
        print("📕 LC (Cerrada):", sorted(closed_nums))
        print("-" * 60)

        # Animación ligera
        pygame.time.delay(15)
        if actual not in (inicio, objetivo):
            actual.dibujar(GRAY)
            pygame.display.update()

        # ¿Llegamos?
        if actual is objetivo:
            print("\n✅ ¡Objetivo alcanzado!\n")
            camino_final = reconstruir_camino(actual)
            print("🟩 Camino final:")
            print(" -> ".join(str(n.numero) for n in camino_final), "-> FIN\n")
            return camino_final, closed_nums

        # ---- Orden de vecinos: prioriza los más cercanos al objetivo (menor h) ----
        vecinos = []
        for dx, dy in movimientos:
            fr, fc = actual.fila + dx, actual.col + dy
            if 0 <= fr < ROWS and 0 <= fc < COLS:
                vecinos.append(grid[fr][fc])
        vecinos.sort(key=lambda v: abs(v.fila - objetivo.fila) + abs(v.col - objetivo.col))

        # Explorar vecinos
        for vecino in vecinos:
            if vecino.es_obstaculo or vecino.numero in closed_nums:
                continue

            nuevo_g = actual.g + 10  # costo ortogonal
            if nuevo_g < best_g.get(vecino.numero, math.inf):
                vecino.g = nuevo_g
                vecino.h = heuristica(vecino, objetivo)
                vecino.f = vecino.g + vecino.h * BIAS   # tie-break
                vecino.padre = actual
                best_g[vecino.numero] = nuevo_g

                # Insertar en abierta (aunque haya otra entrada peor en heap)
                heapq.heappush(open_heap, (vecino.f, vecino.h, next(tick), vecino))
                open_set_nums.add(vecino.numero)

    print("\n❌ No se encontró un camino posible.\n")
    return [], closed_nums

# ==============================
# GENERAR LA CUADRÍCULA
# ==============================
numero_actual = 1
grid = []
for fila in range(ROWS):
    fila_nueva = []
    for col in range(COLS):
        fila_nueva.append(Nodo(fila, col, numero_actual))
        numero_actual += 1
    grid.append(fila_nueva)

# Variables de control
inicio = None
objetivo = None
fase = 0
camino = []
lista_cerrada = set()
arrastrando = False
modo_pintar = True

# ==============================
# BUCLE PRINCIPAL
# ==============================
corriendo = True
while corriendo:
    WIN.fill(WHITE)

    for fila in grid:
        for nodo in fila:
            if nodo.es_obstaculo:
                nodo.dibujar(BLACK)
            elif nodo in camino:
                nodo.dibujar(GREEN)
            elif nodo.numero in lista_cerrada:
                nodo.dibujar(GRAY)
            else:
                nodo.dibujar(WHITE)

    if inicio:
        inicio.dibujar(BLUE)
    if objetivo:
        objetivo.dibujar(RED)

    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            corriendo = False

        elif evento.type == pygame.MOUSEBUTTONDOWN:
            x, y = pygame.mouse.get_pos()
            fila, col = y // TILE_SIZE, x // TILE_SIZE

            if fase == 0:  # Seleccionar inicio
                inicio = grid[fila][col]
                fase = 1
            elif fase == 1:  # Seleccionar fin
                if (fila, col) != (inicio.fila, inicio.col):
                    objetivo = grid[fila][col]
                    fase = 2
            elif fase == 2:  # Dibujar/ borrar obstáculos con arrastre
                nodo = grid[fila][col]
                if nodo not in [inicio, objetivo]:
                    modo_pintar = not nodo.es_obstaculo
                    nodo.es_obstaculo = modo_pintar
                    arrastrando = True

        elif evento.type == pygame.MOUSEBUTTONUP:
            arrastrando = False

        elif evento.type == pygame.MOUSEMOTION and arrastrando and fase == 2:
            x, y = pygame.mouse.get_pos()
            fila, col = y // TILE_SIZE, x // TILE_SIZE
            if 0 <= fila < ROWS and 0 <= col < COLS:
                nodo = grid[fila][col]
                if nodo not in [inicio, objetivo]:
                    nodo.es_obstaculo = modo_pintar

        elif evento.type == pygame.KEYDOWN:
            if evento.key == pygame.K_SPACE and fase == 2:
                # Ejecutar A*
                fase = 3
                camino, lista_cerrada = a_estrella(grid, inicio, objetivo)

    pygame.display.update()

pygame.quit()
