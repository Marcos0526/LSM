"""
launcher.py — Terminal UI para el pipeline de reconocimiento de señas
para usar:
    python launcher.py
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

class C:
    RESET   = "\033[0m"
    BOLD    = "\033[1m"
    DIM     = "\033[2m"
    GREEN   = "\033[38;5;82m"
    GREEN2  = "\033[38;5;46m"
    YELLOW  = "\033[38;5;226m"
    RED     = "\033[38;5;196m"
    CYAN    = "\033[38;5;51m"
    WHITE   = "\033[38;5;255m"
    GREY    = "\033[38;5;240m"
    BG_BLK  = "\033[40m"


def g(text):   return f"{C.GREEN}{text}{C.RESET}"
def g2(text):  return f"{C.GREEN2}{C.BOLD}{text}{C.RESET}"
def y(text):   return f"{C.YELLOW}{text}{C.RESET}"
def r(text):   return f"{C.RED}{text}{C.RESET}"
def c(text):   return f"{C.CYAN}{text}{C.RESET}"
def dim(text): return f"{C.DIM}{C.GREY}{text}{C.RESET}"
def w(text):   return f"{C.WHITE}{C.BOLD}{text}{C.RESET}"

# ── Helpers 
WIDTH = min(shutil.get_terminal_size((80, 24)).columns, 88)


def clear():
    os.system("cls" if os.name == "nt" else "clear")

def line(char="─"):
    print(dim(char * WIDTH))

def sep(char="═"):
    print(g(char * WIDTH))

def center(text, fill=" "):
    import re
    plain = re.sub(r"\033\[[0-9;]*m", "", text)
    pad   = max(0, (WIDTH - len(plain)) // 2)
    print(fill * pad + text)

def pause(msg="  Presiona ENTER para continuar..."):
    input(dim(msg))

# ── Definición de scripts
# Cada entrada: (clave_menu, nombre_display, script_path, descripcion,
#  args_extra)
SCRIPTS = [
    {
        "key":   "1",
        "name":  "Tracking en vivo",
        "file":  "Extraccion_landmarks.py",
        "desc":  "Visualiza landmarks en tiempo real con la cámara",
        "args":  [],
        "color": g2,
    },
    {
        "key":   "2",
        "name":  "Recolectar dataset",
        "file":  "Recolector_datos.py",
        "desc":  "Graba y etiqueta señas → hand_landmarks_dataset.csv",
        "args":  [],
        "color": g2,
    },

]

BANNER = r"""
  ██╗  ██╗ █████╗ ███╗   ██╗██████╗     ███████╗██╗ ██████╗ ███╗   ██╗
  ██║  ██║██╔══██╗████╗  ██║██╔══██╗    ██╔════╝██║██╔════╝ ████╗  ██║
  ███████║███████║██╔██╗ ██║██║  ██║    ███████╗██║██║  ███╗██╔██╗ ██║
  ██╔══██║██╔══██║██║╚██╗██║██║  ██║    ╚════██║██║██║   ██║██║╚██╗██║
  ██║  ██║██║  ██║██║ ╚████║██████╔╝    ███████║██║╚██████╔╝██║ ╚████║
  ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚═════╝     ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═══╝
"""

SUBTITLE = "  Pipeline de Reconocimiento de Lengua de Señas · MediaPipe + CV2"

# ── Verificar existencia de scripts
def script_status(script_path: str) -> str:
    if Path(script_path).exists():
        return g("  ✓ encontrado  ")
    return r("  ✗ no existe   ")

# ── Pantalla principal
def draw_menu():
    clear()
    sep("═")
    for line_txt in BANNER.splitlines():
        center(g(line_txt))
    center(dim(SUBTITLE))
    sep("═")
    print()

    # Scripts
    print(w("  MÓDULOS DISPONIBLES"))
    line()
    for s in SCRIPTS:
        status  = script_status(s["file"])
        key_str = g2(f" [{s['key']}]")
        name    = w(f" {s['name']:<22}")
        desc    = dim(s["desc"])
        print(f"  {key_str}  {name}  {status}  {desc}")
    line()
    print()

    # Utilidades
    print(w("  UTILIDADES"))
    line()
    print(f"  {g2('[V]')}  {w('Ver dataset actual'):<28}  {dim('Muestra estadísticas del CSV')}")
    print(f"  {g2('[Q]')}  {w('Salir'):<28}")
    line()
    print()
    print(dim("  ▸ Escribe el número o letra y presiona ENTER"))
    print()

# ── Ejecutar script
def run_script(entry: dict):
    path = entry["file"]
    if not Path(path).exists():
        print()
        print(r(f"  ✗ Script no encontrado: {path}"))
        print(y(f"  Asegúrate de que '{path}' esté en el mismo directorio que launcher.py"))
        print()
        pause()
        return

    print()
    sep()
    print(g2(f"  ▶  Ejecutando: {path}"))
    sep()
    print()

    cmd = [sys.executable, path] + entry.get("args", [])
    try:
        subprocess.run(cmd, check=False)
    except KeyboardInterrupt:
        print()
        print(y("  ⏹  Interrumpido por el usuario"))
    except Exception as e:
        print(r(f"  Error: {e}"))
    print()
    sep()
    print(dim(f"  {path} finalizó"))
    sep()
    print()
    pause()


# ── Ver dataset
DATASET_DIR = Path("/home/marcos/LSM/Dataset")   

def show_dataset_stats():
    clear()
    sep()
    print(g2("  ESTADÍSTICAS DEL DATASET"))
    sep()
    print()

    csv_files = list(DATASET_DIR.glob("*.csv"))

    if not csv_files:
        print(y("  No se encontró ningún archivo .csv en el directorio actual."))
        print()
        pause()
        return

    import csv
    from collections import Counter

    for csv_path in csv_files:
        print(w(f"  {csv_path}"))
        line("─")
        counts = Counter()
        total  = 0
        try:
            with open(csv_path, "r") as f:
                reader = csv.DictReader(f)
                if "label" not in (reader.fieldnames or []):
                    print(dim("  (sin columna 'label')"))
                    continue
                for row in reader:
                    counts[row["label"]] += 1
                    total += 1
        except Exception as e:
            print(r(f"  Error leyendo {csv_path}: {e}"))
            continue

        if not counts:
            print(dim("  Archivo vacío o sin datos"))
        else:
            max_n = max(counts.values())
            for label, n in sorted(counts.items()):
                bar_len = int((n / max_n) * 30)
                bar     = g("█" * bar_len) + dim("░" * (30 - bar_len))
                pct     = f"{n/total*100:5.1f}%"
                print(f"    {w(f'{label:<18}')} {bar}  {c(f'{n:>4}')} muestras  {dim(pct)}")
            print()
            print(f"  {dim('Total:')}  {g2(str(total))} muestras  ·  {g2(str(len(counts)))} clases")
        print()

    pause()

# ── Loop principal
def main():
    launcher_dir = str(Path(__file__).parent.resolve())
    if launcher_dir not in sys.path:
        sys.path.insert(0, launcher_dir)
    os.chdir(launcher_dir)

    while True:
        draw_menu()
        choice = input(g2("  › ")).strip().lower()
        print()

        if choice == "q":
            clear()
            print()
            center(g2("  Hasta luego  ◈"))
            print()
            break


        elif choice == "v":
            show_dataset_stats()

        else:
            match = next((s for s in SCRIPTS if s["key"] == choice), None)
            if match:
                run_script(match)
            else:
                print(r("  Opción no válida"))
                pause()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()
        print(dim("  Saliendo..."))
        sys.exit(0)