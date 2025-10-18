# KMC-Óxidos: Simulación Monte Carlo Cinético con Reinforcement Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

Simulación Monte Carlo Cinético del Crecimiento de Películas Delgadas de Óxidos Metálicos utilizando Reinforcement Learning basado en el enfoque **SwarmThinkers**.

## 📋 Descripción

Este proyecto implementa un simulador KMC (Kinetic Monte Carlo) para el crecimiento de películas delgadas de TiO₂, integrado con técnicas de aprendizaje por refuerzo (RL) inspiradas en SwarmThinkers. El objetivo es investigar:

- **Escalamiento dinámico** durante el crecimiento de películas
- **Morfología fractal** de superficies
- **Exponentes de Family-Vicsek** (α, β)
- Optimización del crecimiento mediante RL

## 🎯 Características Principales

- ✅ **Simulador KMC completo** para TiO₂ rutilo (110)
- ✅ **Integración con Gymnasium** para entornos RL
- ✅ **Arquitectura SwarmThinkers**: Actor-Crítico con PPO
- ✅ **Mecanismo de reponderación** con importance sampling
- ✅ **Análisis morfológico**: rugosidad, dimensión fractal, escalamiento
- ✅ **Configuración con Pydantic Settings**
- ✅ **Gestión profesional** con uv y ruff

## 🏗️ Arquitectura del Proyecto

```
kmc_oxidos/
├── src/
│   ├── settings/           # Configuración con Pydantic
│   │   └── config.py       # Settings, logging, parámetros
│   ├── kmc/                # Módulo de simulación KMC
│   │   ├── lattice.py      # Estructura de red 3D
│   │   ├── events.py       # Eventos atomísticos
│   │   ├── rates.py        # Cálculo de tasas de Arrhenius
│   │   └── simulator.py    # Motor KMC principal
│   ├── rl/                 # Módulo de Reinforcement Learning
│   │   ├── environment.py  # Ambiente Gymnasium
│   │   ├── policy.py       # Red Actor (MLP 5 capas, 256 units)
│   │   ├── critic.py       # Red Crítico
│   │   └── reweighting.py  # Mecanismo SwarmThinkers
│   ├── analysis/           # Análisis morfológico
│   │   ├── roughness.py    # W(L,t) y escalamiento
│   │   ├── fractal.py      # Dimensión fractal
│   │   └── visualization.py# Visualización
│   └── data/
│       └── tio2_parameters.py  # Parámetros físicos
├── experiments/            # Scripts de experimentación
│   ├── train_policy.py     # Entrenamiento PPO
│   └── run_simulations.py  # Ejecución y análisis
├── notebooks/              # Jupyter notebooks
├── docs/                   # Documentación del proyecto
├── results/                # Resultados de simulaciones
├── checkpoints/            # Modelos guardados
├── logs/                   # Archivos de log
├── pyproject.toml          # Configuración del proyecto (uv)
├── .env.example            # Variables de entorno ejemplo
└── README.md               # Este archivo
```

## 🚀 Instalación

### Requisitos Previos

- Python 3.10 o superior
- [uv](https://github.com/astral-sh/uv) (gestor de paquetes)

### Instalación con uv

```bash
# Clonar el repositorio
cd kmc_oxidos

# Instalar dependencias
uv sync

# Activar el entorno virtual
source .venv/bin/activate  # Linux/Mac
# o
.venv\Scripts\activate  # Windows
```

### Configuración

1. Copiar el archivo de configuración de ejemplo:
```bash
cp .env.example .env
```

2. Editar `.env` con tus parámetros:
```bash
# Parámetros de simulación KMC
LATTICE_SIZE_X=50
LATTICE_SIZE_Y=50
LATTICE_SIZE_Z=20
TEMPERATURE=600.0  # Kelvin
DEPOSITION_RATE=1.0  # ML/s

# Parámetros de entrenamiento RL
RL_LEARNING_RATE=0.0005
RL_TOTAL_TIMESTEPS=1000000
```

## 📖 Uso

### 1. Simulación KMC Clásica

```python
from src.kmc.simulator import KMCSimulator
from src.settings import settings

# Crear simulador
simulator = KMCSimulator(
    lattice_size=(50, 50, 20),
    temperature=600.0,
    deposition_rate=1.0
)

# Ejecutar simulación
simulator.run(max_steps=10000)

# Analizar resultados
height_profile = simulator.lattice.get_height_profile()
composition = simulator.lattice.get_composition()
```

### 2. Entrenamiento con RL (SwarmThinkers)

```bash
# Entrenar política PPO
uv run python experiments/train_policy.py
```

### 3. Ejecutar Simulaciones y Análisis

```bash
# Correr simulaciones con análisis
uv run python experiments/run_simulations.py
```

### 4. Análisis Morfológico

```python
from src.analysis import (
    calculate_roughness,
    calculate_fractal_dimension,
    fit_family_vicsek
)

# Calcular rugosidad
roughness = calculate_roughness(height_profile)

# Dimensión fractal
fractal_dim = calculate_fractal_dimension(height_profile)

# Exponentes de escalamiento
scaling = fit_family_vicsek(times, roughnesses, system_size)
print(f"α = {scaling['alpha']:.3f}, β = {scaling['beta']:.3f}")
```

## 🧪 Ejemplos Detallados

### Ejemplo 1: Configuración Personalizada

```python
from src.settings import Settings

# Crear configuración personalizada
config = Settings(
    kmc=KMCConfig(
        lattice_size_x=100,
        lattice_size_y=100,
        lattice_size_z=30,
        temperature=700.0,
    ),
    rl=RLConfig(
        learning_rate=1e-3,
        total_timesteps=2000000,
    )
)

# Configurar logging
logger = config.setup_logging()
```

### Ejemplo 2: Uso del Ambiente de RL

```python
from src.rl import TiO2GrowthEnv
from stable_baselines3 import PPO

# Crear ambiente
env = TiO2GrowthEnv(
    lattice_size=(10, 10, 10),
    temperature=600.0,
    max_steps=1000
)

# Entrenar modelo
model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=100000)

# Evaluar
obs, info = env.reset()
for _ in range(100):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
```

## 📊 Parámetros Físicos

### TiO₂ Rutilo (110)

| Parámetro | Valor | Unidad | Descripción |
|-----------|-------|--------|-------------|
| `lattice_constant_a` | 4.59 | Å | Parámetro de red a |
| `ea_diff_ti` | 0.6 | eV | Barrera de difusión Ti |
| `ea_diff_o` | 0.8 | eV | Barrera de difusión O |
| `ea_des_ti` | 2.0 | eV | Barrera de desorción Ti |
| `bond_energy_ti_o` | -4.5 | eV | Energía de enlace Ti-O |

Ver `src/data/tio2_parameters.py` para la lista completa.

## 🔧 Desarrollo

### Herramientas de Desarrollo

```bash
# Linting y formateo con ruff
uv run ruff check src/
uv run ruff format src/

# Type checking con mypy
uv run mypy src/

# Jupyter Lab
uv run jupyter lab
```

### Estructura de Código

- **Type hints** en todas las funciones
- **Docstrings** estilo Google
- **Configuración** centralizada con Pydantic
- **Logging** estructurado

## 📚 Fundamento Científico

### Algoritmo KMC (Bortz-Kalos-Lebowitz)

1. **Construir lista de eventos** con sus tasas $\Gamma_i$
2. **Seleccionar evento** proporcionalmente a las tasas
3. **Ejecutar evento** y actualizar sistema
4. **Avanzar tiempo**: $\Delta t = -\ln(r) / \Gamma_{\text{total}}$

### SwarmThinkers

Mecanismo de reponderación:

$$P(a) = \frac{\pi_\theta(a|o) \cdot \Gamma_a}{\sum_{a'} \pi_\theta(a'|o) \cdot \Gamma_{a'}}$$

Con importance sampling para mantener consistencia física.

### Escalamiento de Family-Vicsek

$$W(L,t) = L^\alpha f(t/L^z)$$

Donde:
- $\alpha$: Exponente de rugosidad
- $\beta$: Exponente de crecimiento
- $z = \alpha/\beta$: Exponente dinámico

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Por favor:

1. Fork del proyecto
2. Crear rama feature (`git checkout -b feature/AmazingFeature`)
3. Commit cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abrir Pull Request

Ver `CONTRIBUTING.md` para más detalles.

## 👥 Autores

- **Fabián Castro Contreras** - Investigador Principal
- **Vicente Diaz** - Colaborador
- **Marcos Flores** - Colaborador

## 📄 Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo `LICENSE` para detalles.

## 📖 Referencias

1. SwarmThinkers: Accelerating Kinetic Monte Carlo with Reinforcement Learning
2. Family-Vicsek scaling in thin film growth
3. TiO₂ surface science and thin film growth

## 🔗 Enlaces Útiles

- [Documentación de uv](https://github.com/astral-sh/uv)
- [Stable-Baselines3](https://stable-baselines3.readthedocs.io/)
- [Gymnasium](https://gymnasium.farama.org/)
- [Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)

## 📞 Contacto

Para preguntas o colaboraciones:
- Email: fabian@example.com
- Issues: [GitHub Issues](https://github.com/tu-usuario/kmc_oxidos/issues)

---

**Desarrollado con** ❤️ **en Chile** 🇨🇱
