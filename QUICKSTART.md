# 🚀 Inicio Rápido - KMC-Óxidos

## Instalación en 3 Pasos

### 1. Asegúrate de tener uv instalado

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Instalar dependencias

```bash
cd c:\Users\fabca\Documents\proyectos\kmc_oxidos
uv sync
```

Esto instalará automáticamente:
- PyTorch, NumPy, SciPy
- Stable-Baselines3, Gymnasium
- Pydantic, Matplotlib
- Ruff, Mypy (dev)
- Y todas las demás dependencias

### 3. Configurar variables de entorno

```bash
cp .env.example .env
```

## ✅ Verificar Instalación

```bash
# Verificar que todo funciona
uv run python -c "from src.kmc.simulator import KMCSimulator; print('✅ Todo OK!')"
```

## 🎯 Primer Ejemplo: Simulación KMC

Crear un archivo `test_kmc.py`:

```python
from src.kmc.simulator import KMCSimulator
from src.settings import settings

# Configurar logging
logger = settings.setup_logging()

# Crear simulador pequeño
simulator = KMCSimulator(
    lattice_size=(10, 10, 5),
    temperature=600.0,
    deposition_rate=1.0,
    seed=42
)

# Ejecutar simulación corta
logger.info("Iniciando simulación...")
simulator.run(max_steps=1000)

# Ver resultados
composition = simulator.lattice.get_composition()
logger.info(f"Composición final: {composition}")
logger.info(f"Pasos: {simulator.step}, Tiempo: {simulator.time:.2e}s")
```

Ejecutar:
```bash
uv run python test_kmc.py
```

## 📊 Ejemplo con Análisis

```python
from src.kmc.simulator import KMCSimulator
from src.analysis import calculate_roughness, calculate_fractal_dimension
import matplotlib.pyplot as plt

# Simulación
simulator = KMCSimulator(
    lattice_size=(20, 20, 10),
    temperature=600.0,
    deposition_rate=1.0
)

# Storage para análisis
times = []
roughnesses = []

def snapshot(sim):
    heights = sim.lattice.get_height_profile()
    roughness = calculate_roughness(heights)
    times.append(sim.time)
    roughnesses.append(roughness)
    
# Ejecutar con callbacks
simulator.run(max_steps=5000, callback=snapshot, snapshot_interval=100)

# Análisis final
heights = simulator.lattice.get_height_profile()
fractal_dim = calculate_fractal_dimension(heights)

print(f"Dimensión fractal: {fractal_dim:.3f}")

# Plot
plt.figure(figsize=(8, 5))
plt.loglog(times, roughnesses, 'o-')
plt.xlabel('Tiempo (s)')
plt.ylabel('Rugosidad W(L,t)')
plt.title('Evolución de Rugosidad')
plt.grid(True, alpha=0.3)
plt.savefig('roughness.png', dpi=300)
print("Gráfico guardado en roughness.png")
```

## 🧪 Usar el Notebook de Ejemplo

```bash
uv run jupyter lab
```

Luego abrir: `notebooks/01_ejemplo_kmc_basico.ipynb`

## 🎓 Scripts de Experimentación

### Simulación Clásica KMC

```bash
uv run python experiments/run_simulations.py
```

Esto ejecutará una simulación completa y generará:
- Análisis de rugosidad
- Cálculo de dimensión fractal
- Gráficos en `results/`
- Logs en `logs/`

### Entrenamiento RL (SwarmThinkers)

```bash
uv run python experiments/train_policy.py
```

Esto entrenará una política PPO:
- Modelo guardado en `checkpoints/`
- Logs de entrenamiento
- Puede tomar varias horas

## 🔧 Comandos Útiles

### Desarrollo

```bash
# Formatear código
uv run ruff format src/

# Check linting
uv run ruff check src/

# Type checking
uv run mypy src/

# Ver configuración actual
uv run python -c "from src.settings import settings; import json; print(json.dumps(settings.model_dump_summary(), indent=2))"
```

### Modificar Configuración

Editar `.env`:

```env
# Simulación más grande
LATTICE_SIZE_X=100
LATTICE_SIZE_Y=100
LATTICE_SIZE_Z=30

# Temperatura más alta
TEMPERATURE=800.0

# Más pasos
SIMULATION_TIME=5000.0
```

## 📖 Documentación

- **README.md**: Documentación completa
- **FRAMEWORK_SUMMARY.md**: Resumen técnico
- **CONTRIBUTING.md**: Guía para contribuir
- **docs/avances.md**: Avances del proyecto

## 🆘 Solución de Problemas

### Error: "No module named 'src'"

```bash
# Reinstalar en modo editable
uv sync
```

### Error: "Unable to import gymnasium"

```bash
# Reinstalar dependencias
uv sync --refresh
```

### Error: "CUDA not available"

Si quieres usar GPU, asegúrate de tener CUDA instalado, o cambia en `.env`:
```env
DEVICE=cpu
```

## 🎉 ¡Listo!

Ya puedes:
- ✅ Ejecutar simulaciones KMC
- ✅ Analizar morfología de superficies
- ✅ Entrenar políticas RL
- ✅ Experimentar con parámetros

**Siguiente paso**: Revisar `notebooks/01_ejemplo_kmc_basico.ipynb` para ejemplos interactivos.

## 📞 Soporte

Si encuentras problemas:
1. Revisa la documentación en README.md
2. Verifica los logs en `logs/`
3. Abre un issue en GitHub

---

**¡Buena suerte con tu investigación!** 🔬🚀
