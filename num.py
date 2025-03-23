import numpy as np

# Funkční hodnoty - simulujeme, že neznáme přesný předpis funkce
x_values = np.array([1.0, 3])
f_values = np.array([-3.28, -4])

# Počáteční hodnoty x0 a x1
x0, x1 = x_values[0], x_values[1]
f0, f1 = f_values[0], f_values[1]

# Tolerance pro zastavení iterace
tol = 1e-6
max_iter = 100

# Metoda sečny
for _ in range(max_iter):
    # Vzorec metody sečny
    x2 = x1 - f1 * (x1 - x0) / (f1 - f0)

    # Odhadneme funkční hodnotu pro nové x2 pomocí lineární interpolace
    f2 = np.interp(x2, x_values, f_values)

    # Kontrola konvergence
    if abs(f2) < tol:
        break

    # Posuneme body pro další iteraci
    x0, x1 = x1, x2
    f0, f1 = f1, f2

print(x2)
