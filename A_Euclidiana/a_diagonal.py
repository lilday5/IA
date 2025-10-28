import pygame
import heapq
import random

# ==============================
# CONFIGURACIÓN GENERAL
# ==============================
ROWS, COLS = 7, 7
TILE_SIZE = 50

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GRAY = (200, 200, 200)

pygame.init()
WIN = pygame.display.set_mode((COLS * TILE_SIZE, ROWS * TILE_SIZE))
pygame.display.set_caption("Algoritmo A* (Chebyshev con diagonales balanceadas)")
FONT = pygame.font.SysFont("Arial", 14)

# ==============================
# CLASE NODO
# ==============================
class Nodo:
    def __init__(self, fila, col, numero):
        self.fila = fila
        self.col = col
        self.numero = numero
        self.g = float('inf')
        self.h = 0
        self.f = float('inf')
        self.padre = None
        self.es_obstaculo = False

    def __lt__(self, otro):
        return self.f < otro.f

    def dibujar(self, color):
        pygame.draw.rect(WIN, color, (self.col * TILE_SIZE, self.fila * TILE_SIZE, TILE_SIZE, TILE_SIZE))
        pygame.draw.rect(WIN, BLACK, (self.col * TILE_SIZE, self.fila * TILE_SIZE, TILE_SIZE, TILE_SIZE), 1)

        texto = FONT.render(str(self.numero), True, BLACK)
        text_rect = texto.get_rect(topright=(self.col * TILE_SIZE + TILE_SIZE - 4, self.fila * TILE_SIZE + 2))
        WIN.blit(texto, text_rect)

# ==============================
# FUNCIONES A*
# ==============================
def heuristica(nodo, objetivo):
    """Heurística Chebyshev (permite diagonales)"""
    dx = abs(nodo.fila - objetivo.fila)
    dy = abs(nodo.col - objetivo.col)
    return 10 * max(dx, dy)

def reconstruir_camino(nodo):
    camino = []
    while nodo:
        camino.append(nodo)
        nodo = nodo.padre
    return camino[::-1]

def a_estrella_diagonal(grid, inicio, objetivo):
    lista_abierta = []
    lista_cerrada = set()
    
    # 🔄 Semilla aleatoria para variar entre izquierda/derecha
    preferencia = random.choice([-1, 1])

    inicio.g = 0
    inicio.h = heuristica(inicio, objetivo)
    inicio.f = inicio.g + inicio.h
    heapq.heappush(lista_abierta, (inicio.f, inicio))

    movimientos = [
        (-1, 0, 10),  # Arriba
        (1, 0, 10),   # Abajo
        (0, -1, 10),  # Izquierda
        (0, 1, 10),   # Derecha
        (-1, -1, 14), # Diagonal sup-izq
        (-1, 1, 14),  # Diagonal sup-der
        (1, -1, 14),  # Diagonal inf-izq
        (1, 1, 14)    # Diagonal inf-der
    ]

    # Mezclar movimientos aleatoriamente para romper simetría espejo
    random.shuffle(movimientos)

    print("\n🧩 Iniciando búsqueda A* (Chebyshev balanceado)")
    print(f"Punto inicial: {inicio.numero}")
    print(f"Punto final:   {objetivo.numero}\n")

    while lista_abierta:
        actual_f, actual = heapq.heappop(lista_abierta)
        lista_cerrada.add(actual.numero)

        print(f"➡️ Explorando nodo: {actual.numero}  f={actual.f} g={actual.g} h={actual.h}")
        print("📘 LA (Lista Abierta):", [n.numero for _, n in lista_abierta])
        print("📕 LC (Lista Cerrada):", lista_cerrada)
        print("-" * 60)

        pygame.time.delay(25)
        actual.dibujar(GRAY)
        pygame.display.update()

        if actual == objetivo:
            print("\n✅ ¡Objetivo alcanzado!\n")
            camino_final = reconstruir_camino(actual)
            print("🟩 Camino final:")
            print(" -> ".join(str(n.numero) for n in camino_final), "-> FIN\n")
            return camino_final, lista_cerrada

        for dx, dy, costo_mov in movimientos:
            fila_nueva, col_nueva = actual.fila + dx, actual.col + dy
            if not (0 <= fila_nueva < ROWS and 0 <= col_nueva < COLS):
                continue

            vecino = grid[fila_nueva][col_nueva]
            if vecino.es_obstaculo or vecino.numero in lista_cerrada:
                continue

            nuevo_g = actual.g + costo_mov
            if nuevo_g < vecino.g:
                vecino.g = nuevo_g
                vecino.h = heuristica(vecino, objetivo)
                # ⚖️ Factor de desempate dinámico
                vecino.f = vecino.g + vecino.h * (1.001 + random.uniform(0, 0.001) * preferencia)
                vecino.padre = actual
                if all(n != vecino for _, n in lista_abierta):
                    heapq.heappush(lista_abierta, (vecino.f, vecino))

    print("\n❌ No se encontró un camino posible.\n")
    return [], lista_cerrada

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

            if fase == 0:
                inicio = grid[fila][col]
                fase = 1
            elif fase == 1:
                if (fila, col) != (inicio.fila, inicio.col):
                    objetivo = grid[fila][col]
                    fase = 2
            elif fase == 2:
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
                fase = 3
                for fila in grid:
                    for nodo in fila:
                        nodo.g = float('inf')
                        nodo.f = float('inf')
                        nodo.padre = None
                camino, lista_cerrada = a_estrella_diagonal(grid, inicio, objetivo)

    pygame.display.update()

pygame.quit()
