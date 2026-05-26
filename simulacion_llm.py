import os
import glob
import getpass
from groq import Groq

def leer_archivos_vision(carpeta_origen):
    textos_crudos = []
    if not os.path.exists(carpeta_origen):
        print(f"Error: No se encontró la carpeta '{carpeta_origen}'")
        return textos_crudos
    rutas_archivos = glob.glob(os.path.join(carpeta_origen, '*.txt'))
    for ruta in rutas_archivos:
        try:
            with open(ruta, 'r', encoding='utf-8') as archivo:
                contenido = archivo.read().strip()
                textos_crudos.append({"archivo": os.path.basename(ruta), "glosas": contenido})
        except Exception as e:
            print(f"Error leyendo {ruta}: {e}")
    return textos_crudos

def procesar_con_llm(texto_entrada, cliente_ai):
    prompt_sistema = (
        "Eres un experto traductor de Lengua de Senas Mexicana (LSM) al espanol. "
        "Recibes glosas (palabras clave aisladas) extraidas por un modelo de vision. "
        "Genera la oracion en espanol mas natural y coherente. "
        "Devuelve UNICAMENTE la oracion traducida, sin comentarios ni explicaciones."
    )
    try:
        respuesta = cliente_ai.chat.completions.create(
            model="llama-3.1-8b-instant",
            temperature=0.2,
            messages=[
                {"role": "system", "content": prompt_sistema},
                {"role": "user", "content": texto_entrada}
            ]
        )
        return respuesta.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR]: {str(e)}"

if __name__ == "__main__":
    print("   MODULO LLM TEXTO LSM                         ")

    api_key = getpass.getpass("API Key de Groq (gsk_...): ").strip()
    if not api_key:
        print("ERROR: No se ingreso API Key.")
        exit(1)

    cliente = Groq(api_key=api_key)
    carpeta_vision = "outputs_vision"

    print("\nLeyendo archivos...")
    datos_vision = leer_archivos_vision(carpeta_vision)

    if not datos_vision:
        print("No hay archivos para procesar.")
    else:
        datos_vision_ordenados = sorted(datos_vision, key=lambda x: x['archivo'])
        print("\nProcesando con el LLM...\n")
        for dato in datos_vision_ordenados:
            print(f"[{dato['archivo']}] ENTRADA : '{dato['glosas']}'")
            resultado = procesar_con_llm(dato['glosas'], cliente)
            print(f"[{dato['archivo']}] SALIDA  : '{resultado}'\n")
            print("-" * 50)
