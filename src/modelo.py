import pulp 
from collections import defaultdict 

def generar_timeslots():
    dias_validos = [1, 2, 3, 4, 5, 9, 10, 11, 12]
    horarios = ["09", "12", "15", "18"]
    timeslots = []
    for d in dias_validos:
        for h in horarios:
            timeslots.append((d, h))
    return timeslots

def construir_modelo_ej1(cursos, incompatibles, capacidad_aulas=75):
    P = list(cursos.keys())
    T = generar_timeslots()

    # Modelo
    modelo = pulp.LpProblem("ProgramacionParciales", pulp.LpMaximize)

    # Variables
    x = pulp.LpVariable.dicts(
        "x", (P, range(len(T))), lowBound=0, upBound=1, cat=pulp.LpBinary
    )
    y = pulp.LpVariable.dicts(
        "y", P, lowBound=0, upBound=1, cat=pulp.LpBinary
    )

    # Objetivo: maximizar parciales asignados
    modelo += pulp.lpSum([y[p] for p in P])

    # 1) Cada parcial a lo sumo en un timeslot
    for p in P:
        modelo += pulp.lpSum([x[p][t] for t in range(len(T))]) <= 1, f"uno_por_parcial_{p}"

    # 2) Relacionamos con la variable "objetivo" y[p]
    for p in P:
        modelo += y[p] <= pulp.lpSum([x[p][t] for t in range(len(T))])
        modelo += pulp.lpSum([x[p][t] for t in range(len(T))]) <= y[p]

    # 3) Incompatibilidades
    for (p, q) in incompatibles:
        for t in range(len(T)):
            modelo += x[p][t] + x[q][t] <= 1, f"incomp_{p}_{q}_t{t}"

    # 4) Capacidad de aulas por timeslot
    for t in range(len(T)):
        modelo += pulp.lpSum([cursos[p] * x[p][t] for p in P]) <= capacidad_aulas, f"capacidad_t{t}"

    return modelo, T, P, x, y

def construir_modelo_ej2(cursos, incompatibles, capacidad_aulas=75):
    P = list(cursos.keys())
    T = generar_timeslots()

    # === Mapa día -> tiemposlots ===
    dias_validos = sorted(set(d for (d, h) in T))
    T_por_dia = {d: [] for d in dias_validos}
    for idx, (d, h) in enumerate(T):
        T_por_dia[d].append(idx)

    # === Grafo para encontrar triángulos ===
    vecinos = defaultdict(set)
    for p, q in incompatibles:
        vecinos[p].add(q)
        vecinos[q].add(p)

    # Buscar triángulos p–q–r
    triangulos = set()
    for p in vecinos:
        for q in vecinos[p]:
            if q <= p: continue
            for r in vecinos[p].intersection(vecinos[q]):
                if r <= q: continue
                triangulos.add((p, q, r))

    modelo = pulp.LpProblem("ProgramacionParciales_Ej2", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("x", (P, range(len(T))), 0, 1, pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", P, 0, 1, pulp.LpBinary)

    modelo += pulp.lpSum([y[p] for p in P])

    # Restricciones originales del Ej 1
    for p in P:
        modelo += pulp.lpSum([x[p][t] for t in range(len(T))]) <= 1

    for p in P:
        modelo += y[p] <= pulp.lpSum([x[p][t] for t in range(len(T))])
        modelo += pulp.lpSum([x[p][t] for t in range(len(T))]) <= y[p]

    for (p, q) in incompatibles:
        for t in range(len(T)):
            modelo += x[p][t] + x[q][t] <= 1

    for t in range(len(T)):
        modelo += pulp.lpSum([cursos[p] * x[p][t] for p in P]) <= capacidad_aulas

    for (p, q, r) in triangulos:
        for d in dias_validos:
            modelo += (
                pulp.lpSum([x[p][t] for t in T_por_dia[d]]) +
                pulp.lpSum([x[q][t] for t in T_por_dia[d]]) +
                pulp.lpSum([x[r][t] for t in T_por_dia[d]])
            ) <= 2

    return modelo, T, P, x, y

def construir_modelo_ej3(cursos, incompatibles, pesos, capacidad_aulas=75, alpha=0.001):
    """
    - Queremos dos cosas a la vez:
        1) Maximizamos la cantidad de parciales programados (sum_p y[p]).
        2) Con menor peso, maximizamos la dispersión:
           penalizamos si dos parciales p y q (con w[p,q] alumnos en común)
           quedan el MISMO día o en DÍAS CONSECUTIVOS.
    - Objetivo combinado:
           max  sum_p y[p]  -  alpha * (penalización por choques)
    """

    P = list(cursos.keys())
    T = generar_timeslots()

    dias_validos = sorted(set(d for (d, h) in T))
    T_por_dia = {d: [] for d in dias_validos}
    for idx, (d, h) in enumerate(T):
        T_por_dia[d].append(idx)

    pares_dias_consec = [
        (dias_validos[i], dias_validos[i + 1])
        for i in range(len(dias_validos) - 1)
    ]

    aristas = list(pesos.keys())

    modelo = pulp.LpProblem("ProgramacionParciales_Ej3", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("x", (P, range(len(T))), 0, 1, pulp.LpBinary)
    y = pulp.LpVariable.dicts("y", P, 0, 1, pulp.LpBinary)

    # z[(p,q)][d] = 1 si p y q se programan el MISMO día d
    z = pulp.LpVariable.dicts(
        "z", (aristas, dias_validos), 0, 1, pulp.LpBinary
    )

    # c[(p,q)][(d1,d2)] = 1 si p y q se programan en la ventana de dos días consecutivos (d1,d2)
    c = pulp.LpVariable.dicts(
        "c", (aristas, pares_dias_consec), 0, 1, pulp.LpBinary
    )

    # Restricciones originales del Ej 1
    for p in P:
        modelo += (
            pulp.lpSum([x[p][t] for t in range(len(T))]) <= 1,
            f"uno_por_parcial_{p}"
        )

    for p in P:
        modelo += (
            y[p] <= pulp.lpSum([x[p][t] for t in range(len(T))]),
            f"y_le_suma_{p}"
        )
        modelo += (
            pulp.lpSum([x[p][t] for t in range(len(T))]) <= y[p],
            f"suma_le_y_{p}"
        )

    for (p, q) in incompatibles:
        for t in range(len(T)):
            modelo += (
                x[p][t] + x[q][t] <= 1,
                f"incomp_{p}_{q}_t{t}"
            )

    for t in range(len(T)):
        modelo += (
            pulp.lpSum([cursos[p] * x[p][t] for p in P]) <= capacidad_aulas,
            f"capacidad_t{t}"
        )

    for (p, q) in aristas:
        for d in dias_validos:
            modelo += (
                z[(p, q)][d]
                >=
                pulp.lpSum([x[p][t] for t in T_por_dia[d]]) +
                pulp.lpSum([x[q][t] for t in T_por_dia[d]]) - 1,
                f"def_z_{p}_{q}_d{d}"
            )

    for (p, q) in aristas:
        for (d1, d2) in pares_dias_consec:
            pres_p = (
                pulp.lpSum([x[p][t] for t in T_por_dia[d1]]) +
                pulp.lpSum([x[p][t] for t in T_por_dia[d2]])
            )
            pres_q = (
                pulp.lpSum([x[q][t] for t in T_por_dia[d1]]) +
                pulp.lpSum([x[q][t] for t in T_por_dia[d2]])
            )
            modelo += (
                c[(p, q)][(d1, d2)] >= pres_p + pres_q - 1,
                f"def_c_{p}_{q}_d{d1}_{d2}"
            )
    
    # 1) Termino principal: max sum_p y[p]  (cantidad de parciales asignados)
    term_asignados = pulp.lpSum([y[p] for p in P])

    # 2) Penalización por choques (mismo día + días consecutivos),
    #    ponderada por w[p,q]
    term_penalizacion = (
        pulp.lpSum(
            pesos[(p, q)] * z[(p, q)][d]
            for (p, q) in aristas
            for d in dias_validos
        )
        +
        pulp.lpSum(
            pesos[(p, q)] * c[(p, q)][(d1, d2)]
            for (p, q) in aristas
            for (d1, d2) in pares_dias_consec
        )
    )

    # Objetivo: max (parciales asignados) - alpha * (penalización)
    modelo += term_asignados - alpha * term_penalizacion

    return modelo, T, P, x, y, z, c
