import pandas as pd
import random

# Configuración de muestras por combinación
muestras_por_comb = {
    ("Leve", "Inatento"): 20,
    ("Leve", "Hiperactivo"): 20,
    ("Leve", "Combinado"): 20,
    ("Moderado", "Inatento"): 40,
    ("Moderado", "Hiperactivo"): 40,
    ("Moderado", "Combinado"): 40,
    ("Severo", "Inatento"): 60,
    ("Severo", "Hiperactivo"): 60,
    ("Severo", "Combinado"): 60,
}

# Ítems
items_invertidos = [6, 7, 9, 10, 11, 12, 14, 16, 18]
items_hiperactividad = [0, 2, 4, 12, 16]
items_atencion = [1, 3, 6, 7, 18]
items_conducta = [5, 8, 9, 10, 11, 13, 14, 15, 17, 19]

# Simula la respuesta a un test EDAH (más realista)
def diagnosticar(respuestas):
    # Invertir los puntajes de ciertos ítems
    invertidas = [(3 - r if (i+1) in items_invertidos else r) for i, r in enumerate(respuestas)]
    
    # Calcular los puntajes para cada dimensión
    da_score = sum(invertidas[i] for i in items_atencion)
    hi_score = sum(invertidas[i] for i in items_hiperactividad)
    tc_score = sum(invertidas[i] for i in items_conducta)
    
    total_score = da_score + hi_score + tc_score

    # Determinación del tipo TDAH
    if da_score > hi_score + 2:
        tipo = "Inatento"
    elif hi_score > da_score + 2:
        tipo = "Hiperactivo"
    else:
        tipo = "Combinado"

    # Determinación del nivel de TDAH
    if total_score <= 20:
        nivel = "Leve"
    elif total_score <= 30:
        nivel = "Moderado"
    else:
        nivel = "Severo"

    return da_score, hi_score, tc_score, total_score, tipo, nivel

# Genera datos de respuesta realistas
def generar_respuesta_por_objetivo(nivel_deseado, tipo_deseado, max_intentos=2000):
    for _ in range(max_intentos):
        # Respuestas aleatorias, pero con cierta lógica por tipo
        if tipo_deseado == "Inatento":
            respuestas = [random.choice([0, 1, 2]) for _ in range(20)]  # Menos respuestas extremas
        elif tipo_deseado == "Hiperactivo":
            respuestas = [random.choice([1, 2, 3]) for _ in range(20)]  # Tendencia a ser más impulsivo
        else:
            respuestas = [random.randint(0, 3) for _ in range(20)]  # Combinado más balanceado
        
        da, hi, tc, total, tipo, nivel = diagnosticar(respuestas)
        
        # Asegurarse de que el tipo y nivel sean los deseados
        if tipo == tipo_deseado and nivel == nivel_deseado:
            return respuestas, da, hi, tc, total, tipo, nivel
        
    return None, None, None, None, None, None, None

# Generar dataset
data = []
id_actual = 1

# Agregar más variabilidad en las características demográficas
def generar_caracteristicas_demograficas():
    edad = random.randint(6, 12)
    sexo = random.choice(['M', 'F'])
    escuela = random.choice(['Escuela A', 'Escuela B', 'Escuela C'])
    return edad, sexo, escuela

for (nivel, tipo), cantidad in muestras_por_comb.items():
    print(f"🔄 Generando {cantidad} muestras para Nivel: {nivel} | Tipo: {tipo}")
    generadas = 0
    intentos_fallidos = 0
    while generadas < cantidad:
        respuestas, da, hi, tc, total, tipo_real, nivel_real = generar_respuesta_por_objetivo(nivel, tipo)
        
        if respuestas is not None:
            edad, sexo, escuela = generar_caracteristicas_demograficas()
            fila = [id_actual, edad, sexo, escuela] + respuestas + [da, hi, tc, total, tipo_real, nivel_real]
            data.append(fila)
            id_actual += 1
            generadas += 1
        else:
            intentos_fallidos += 1
            if intentos_fallidos > 20:
                print(f"⚠️ No se pudieron generar suficientes muestras para ({nivel}, {tipo})")
                break

# Crear el DataFrame y guardar el archivo CSV
columnas = ['Id_Evaluacion', 'Edad', 'Sexo', 'Escuela'] + [f'Respuesta_{i}' for i in range(1, 21)] + ['DA_Score', 'HI_Score', 'TC_Score', 'Total_Score', 'Tipo_TDAH', 'Nivel_TDAH']
df = pd.DataFrame(data, columns=columnas)

# Guardar el dataset
nombre_archivo = "edah_dataset_mejorado.csv"
df.to_csv(nombre_archivo, index=False)
print(f"✅ Dataset generado: {nombre_archivo} con {len(df)} muestras")
