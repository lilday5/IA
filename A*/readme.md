<h1 align="center">🧠 Algoritmo A* — Búsqueda Informada (Manhattan)</h1>

<p align="center">
  <b>Visualizador interactivo del algoritmo A* con heurística Manhattan</b><br>
  Implementado en <code>Python</code> + <code>Pygame</code>  
</p>

---

## 🚀 Características principales

✨ **A\*** con representación gráfica en tiempo real.  
🧩 Cuadrícula **dinámica y numerada automáticamente**.  
🖱️ **Dibuja obstáculos con el mouse** (click y arrastre).  
🧠 **Heurística Manhattan** (solo 4 direcciones, sin diagonales).  
⚖️ **Desempate inteligente** con sesgo hacia el objetivo.  
🎨 **Colores claros e intuitivos** para cada estado.

---

## 🎮 Interacción

| Acción | Descripción |
|--------|--------------|
| 🖱️ **Click izquierdo** | Selecciona primero el inicio (🔵), luego el fin (🔴), después dibuja paredes (⚫). |
| 🖱️ **Click + arrastre** | Dibuja o borra obstáculos fácilmente. |
| ␣ **Barra espaciadora** | Ejecuta el algoritmo A*. |
| ❌ **Cerrar ventana** | Sale del programa. |

---

## 🎨 Colores y significados

| Color | Significado |
|:------|:-------------|
| 🔵 Azul | Nodo inicial |
| 🔴 Rojo | Nodo final |
| ⚫ Negro | Obstáculo |
| 🟩 Verde | Camino óptimo encontrado |
| 🩶 Gris | Nodos explorados |
| ⚪ Blanco | Nodos no visitados |

---

## 🧮 Conceptos utilizados

### 🔹 Heurística Manhattan
> Distancia usada para estimar el costo al objetivo, **solo en direcciones ortogonales**:

\[
h(n) = 10 \times (|x_1 - x_2| + |y_1 - y_2|)
\]

👉 No considera diagonales.

---

### 🔹 Cálculo del costo total

\[
f(n) = g(n) + h(n)
\]

Donde:
- **g(n)** → Costo real desde el inicio hasta el nodo actual  
- **h(n)** → Costo estimado al objetivo  
- **f(n)** → Costo total utilizado para decidir el siguiente nodo

💡 En este proyecto, se aplica un **sesgo suave** `1.001` para romper empates:
```python
f = g + h * 1.001
