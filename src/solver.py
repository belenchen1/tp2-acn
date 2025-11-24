import csv
import os
import pulp

from lectura_de_datos import cargar_cursos, cargar_incompatibilidades
from modelo import (
    construir_modelo_ej1,
    construir_modelo_ej2,
    construir_modelo_ej3,
)


def resolver_y_guardar(nombre_ejercicio, construir_modelo_fn, cursos, incompatibles):
    print(f"\n=== Resolviendo {nombre_ejercicio} ===")

    modelo, timeslots, P, x, y = construir_modelo_fn(cursos, incompatibles)

    modelo.solve(pulp.PULP_CBC_CMD(msg=True))

    valor_obj = pulp.value(modelo.objective)

    print("Status:", pulp.LpStatus[modelo.status])
    print("Cantidad de cursos:", len(P))
    print("Valor objetivo (parciales programados):", valor_obj)

    if int(valor_obj) == len(P):
        print("Se programaron TODOS los parciales.")
    else:
        print("Hay parciales sin programar.")

    os.makedirs("output", exist_ok=True)

    ruta_csv = f"output/solucion_{nombre_ejercicio}.csv"

    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["curso", "dia", "hora"])
        for p in P:
            asignado = False
            for idx_t, (d, h) in enumerate(timeslots):
                valor = x[p][idx_t].varValue
                if valor is not None and valor > 0.5:
                    writer.writerow([p, d, h])
                    asignado = True
            if not asignado:
                writer.writerow([p, "-", "-"])

    print(f"Archivo generado: {ruta_csv}")


def resolver_y_guardar_ej3(cursos, incompatibles, pesos, alpha=0.001):
    print("\n=== Resolviendo ej3 (objetivo combinado) ===")

    modelo, timeslots, P, x, y, z, c = construir_modelo_ej3(
        cursos, incompatibles, pesos, capacidad_aulas=75, alpha=alpha
    )

    modelo.solve(pulp.PULP_CBC_CMD(msg=True))

    valor_obj = pulp.value(modelo.objective)

    total_asignados = sum(
        (y[p].varValue or 0.0) for p in P
    )

    print("Status:", pulp.LpStatus[modelo.status])
    print("Cantidad de cursos:", len(P))
    print("Parciales asignados (sum y[p]):", total_asignados)
    print("Valor objetivo combinado:", valor_obj)
    print(f"(recordá: obj = sum y[p] - {alpha} * penalización_dispersion)")

    if int(round(total_asignados)) == len(P):
        print("Se programaron TODOS los parciales (a pesar de la penalización).")
    else:
        print("Hay parciales sin programar (porque el modelo prefiere menos choques).")

    os.makedirs("output", exist_ok=True)

    ruta_csv = "output/solucion_ej3.csv"

    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["curso", "dia", "hora"])
        for p in P:
            asignado = False
            for idx_t, (d, h) in enumerate(timeslots):
                valor = x[p][idx_t].varValue
                if valor is not None and valor > 0.5:
                    writer.writerow([p, d, h])
                    asignado = True
            if not asignado:
                writer.writerow([p, "-", "-"])

    print(f"Archivo generado: {ruta_csv}")


def main():
    cursos = cargar_cursos("data/cursos.dat")
    incompatibles, pesos = cargar_incompatibilidades("data/estudiantes-en-comun.dat")

    # Ejercicio 1: modelo base
    resolver_y_guardar("ej1", construir_modelo_ej1, cursos, incompatibles)

    # Ejercicio 2: modelo con restricción de no 3 parciales el mismo día
    resolver_y_guardar("ej2", construir_modelo_ej2, cursos, incompatibles)

    # Ejercicio 3: objetivo combinado (max parciales, min choques ponderados)
    resolver_y_guardar_ej3(cursos, incompatibles, pesos, alpha=0.001)


if __name__ == "__main__":
    main()
