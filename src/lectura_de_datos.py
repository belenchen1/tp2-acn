import csv

def cargar_cursos(path="data/cursos.dat"):
    cursos = {}  
    with open(path, "r", encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()      
            if not linea:              
                continue

            parts = linea.split()        
            if len(parts) < 2:           
                print("Línea inválida en cursos.dat:", repr(linea))
                continue

            curso_id = parts[0]
            aulas = int(parts[1])
            cursos[curso_id] = aulas

    return cursos

def cargar_incompatibilidades(path="data/estudiantes-en-comun.dat"):
    incompatibles = set()
    pesos = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()          
            if not line:                 
                continue

            parts = line.split()         

            if len(parts) < 3:
                print("Línea inválida:", repr(line))
                continue    

            p = parts[0]
            q = parts[1]
            w = int(parts[2])

            key = tuple(sorted((p, q)))
            incompatibles.add(key)
            pesos[key] = w

    return incompatibles, pesos
