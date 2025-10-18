# Avances del Roadmap

## Preparación y Adaptación del Modelo

### Revisión Profunda de SwarmThinkers

Esta sección proporciona un análisis detallado de la implementación de SwarmThinkers, examinando sus algoritmos, la estructura actor-crítico, el mecanismo de reponderación, y cómo se definen y utilizan sus componentes clave.

1. Estudiar en Detalle la Implementación

*   **Algoritmos (PPO):** SwarmThinkers utiliza la Optimización de Políticas Próximas (PPO) para optimizar su política. PPO es un algoritmo de aprendizaje por refuerzo que busca maximizar la recompensa esperada ajustando la política de los agentes. Se elige por su estabilidad y eficiencia en configuraciones de gradiente de política on-policy.[1] La optimización se realiza utilizando el optimizador Adam con una tasa de aprendizaje de $5\times10^{-4}$, y se aplican 10 épocas PPO con un tamaño de mini-lote de 256 para las actualizaciones de la política, con un recorte de gradiente para asegurar la estabilidad del entrenamiento.[1]

*   **Estructura Actor-Crítico:** El marco de SwarmThinkers emplea una arquitectura actor-crítico. El "actor" (política descentralizada) es una red neuronal compartida ($f_{\theta}$) que cada partícula difusora modelada como un agente autónomo utiliza para proponer transiciones localmente, basándose en su observación de vecindario.[1] El "crítico" (función de valor centralizada, $V_{\phi}(s)$) evalúa las acciones de los agentes a nivel de sistema, proporcionando una señal de crédito global que vincula las decisiones atómicas de corto plazo con los resultados estructurales a largo plazo.[1] Esta estructura permite un paradigma de entrenamiento centralizado y ejecución descentralizada (CTDE), donde los agentes reciben solo observaciones locales, mientras que el crítico aprovecha el contexto global para asignar crédito a largo plazo.[1]

*   **Mecanismo de Reponderación:** Para asegurar la consistencia termodinámica mientras se aprovecha la inteligencia aprendida, SwarmThinkers introduce un mecanismo de reponderación. Este mecanismo fusiona las preferencias aprendidas de la política ($\pi_{\theta}(a|o_{1:N})$) con las tasas de transición clásicas, derivadas físicamente ($\Gamma_{a}$), para construir una distribución de muestreo reponderada $P(a) = \frac{\pi_{\theta}(a|o_{1:N})\cdot\Gamma_{a}}{\sum_{a^{\prime}}\pi_{\theta}(a^{\prime}|o_{1:N})\cdot\Gamma_{a^{\prime}}}$.[1] Aunque esta distribución mejora la eficiencia del muestreo al dar mayor peso a las transiciones que la política considera importantes, se desvía de la verdadera distribución física. Para corregir este sesgo y garantizar la imparcialidad estadística, se emplea el muestreo por importancia, que utiliza pesos inversos de la política para estimar observables físicos de manera imparcial.[1]

2. Identificar Componentes Clave

*   **Observación Local del Agente:** Cada agente en SwarmThinkers percibe su entorno atómico a través de una "observación local" ($o_i$). Esta observación se construye como un vector de longitud fija que representa un vecindario de radio fijo alrededor del agente, típicamente incluyendo los primeros y segundos vecinos más cercanos.[1] La representación es $o_{i}=[\sigma_{ij}]_{j\in\mathcal{N}_{i}}$, donde $\sigma_{ij}\in\mathbb{Z}$ denota la especie atómica del vecino $j$.[1] Esta codificación minimalista y consciente de la simetría captura los factores relevantes para la transición, siendo invariante al tamaño del sistema, la densidad de defectos y la geometría, lo que contribuye a la escalabilidad y generalización del marco.[1]

*   **Cálculo de Recompensas:** La función de recompensa en SwarmThinkers se define de manera minimalista y basada en la física como el cambio negativo en la energía total del sistema: $r_{t}=-\Delta E_{t}$, donde $\Delta E_{t}=E_{t+1}-E_{t}$.[1] Esta recompensa refleja el impulso termodinámico hacia estados de menor energía, guiando a la política para descubrir secuencias de transición que aceleran la relajación estructural sin necesidad de una configuración de recompensa específica para la tarea.[1]

*   **Integración de las Tasas Físicas:** La integración de las tasas físicas ($\Gamma_a$) es fundamental para preservar la consistencia termodinámica. Aunque la política aprendida $\pi_{\theta}$ propone transiciones preferidas, la selección final de eventos se modula mediante estas tasas físicas a través del mecanismo de reponderación.[1] Esto asegura que, a pesar de la priorización inteligente, las propiedades estadísticas del sistema simulado se mantengan fieles a la física subyacente, utilizando el muestreo por importancia para corregir cualquier sesgo introducido por la política.[1] Además, el marco extiende esta estimación imparcial a nivel de trayectoria para observables dependientes de la ruta, como el factor de avance del Cu, utilizando un peso de importancia acumulativo.[1]


### Definición del Sistema Físico de Óxido Metálico

Esta sección detalla la selección del óxido metálico más simple para la simulación, los eventos atomísticos elementales que rigen su crecimiento y las metodologías para determinar las tasas de Arrhenius y las energías de activación asociadas.

1. Selección del Óxido Más Básico: Dióxido de Titanio (TiO2)

Para la simulación del crecimiento de películas delgadas de óxido metálico, el **Dióxido de Titanio (TiO2)**, específicamente su superficie de rutilo (110), se ha seleccionado como el sistema modelo más simple y adecuado. El TiO2 es un material ampliamente investigado con relevancia tecnológica, y la superficie de rutilo (110) es su faceta de menor energía superficial, lo que la hace comúnmente presente en estudios experimentales de ciencia de superficies.[1] Existe una base sólida de trabajos KMC sobre superficies de TiO2, particularmente en procesos de hidratación e hidroxilación, que son análogos a los eventos de adsorción y reacción en el crecimiento de películas delgadas.[1] La simulación apuntará a una película de TiO2 estequiométrica (relación Ti:O de 1:2), con un sustrato inicial de rutilo TiO2 (110) considerado rígido e inerte.[1]

2. Eventos Atomísticos Elementales para el Crecimiento de Películas Delgadas

Los modelos KMC representan el crecimiento de películas delgadas como una secuencia de eventos discretos que ocurren en una red predefinida.[2, 3, 4] Para el crecimiento de películas delgadas de óxido metálico, los eventos críticos incluyen:

*   **Adsorción de especies metálicas y de oxígeno:**
    *   **Adsorción de Ti:** Un átomo de titanio (Ti) de la fase de vapor se adhiere a un sitio disponible en la superficie del sustrato o de la película en crecimiento.[5, 6, 7]
    *   **Adsorción y Disociación de O2:** Las moléculas de oxígeno (O2) pueden adsorberse en la superficie y luego disociarse en átomos de oxígeno (O) reactivos. La disociación de O2 es crucial para la formación de enlaces Ti-O y la estequiometría de la película.[8] La velocidad de deposición (flujo de partículas entrantes) influye directamente en la frecuencia de los eventos de adsorción.[3, 9]

*   **Difusión superficial de átomos metálicos y de oxígeno:**
    *   **Difusión de Ti y O:** Los adátomos de Ti y O, o pequeños cúmulos, se mueven a través de la superficie de la película en crecimiento, saltando de un sitio de red a otro adyacente sobre una barrera de energía.[5, 3, 7, 10] Este proceso permite que los átomos encuentren sitios de menor energía, influyendo en la densificación y morfología de la película.[2, 9] La difusión puede ser anisotrópica, con velocidades que varían según la dirección cristalográfica o las características locales de la superficie.[11]

*   **Reacciones de formación del óxido:**
    *   **Formación de enlaces Ti-O:** Las especies de Ti y O adsorbidas se unen para formar la estructura estable del óxido metálico (TiO2).[6, 8] Esto implica la formación de enlaces Metal-O y O-Metal, contribuyendo al crecimiento de la red cristalina y al llenado de sitios vacantes.[6] Estas reacciones suelen activarse térmicamente, y el entorno local influye en la barrera de reacción.[4]

*   **Posibles eventos de desorción o reestructuración:**
    *   **Desorción (Ti, O):** Las especies adsorbidas pueden desprenderse de la superficie y regresar a la fase gaseosa, compitiendo con la adsorción y la difusión, especialmente a temperaturas más altas.[5, 7, 10]
    *   **Nucleación y Coalescencia de Islas:** Formación de cúmulos estables de átomos depositados y la fusión de islas más pequeñas en otras más grandes.[7, 12]
    *   **Recocido/Formación de Defectos:** Reordenamiento de átomos para minimizar la energía o acomodar la tensión, incluyendo la creación o aniquilación de defectos puntuales.[5, 8, 4]

3. Tasas de Arrhenius y Energías de Activación

La velocidad de un evento $X$ activado térmicamente en KMC se rige por la ecuación de Arrhenius: $\Gamma_{X}=\Gamma_{0}exp(-\frac{E_{a}^{X}}{kT})$.[5, 13, 10, 4]

*   **Energías de Activación ($E_a$):**
    *   **Potenciales Interatómicos (Enfoques Clásicos):** Se utilizan para calcular la energía total del sistema y estimar la barrera de energía para una transición. Los potenciales de pares son computacionalmente económicos, mientras que los potenciales de muchos cuerpos (como EAM o Tersoff) ofrecen mayor precisión al considerar el entorno local y los ángulos de enlace.[14, 4] La energía de activación $E_a$ puede relacionarse con la diferencia de energía entre los estados inicial y final, a menudo combinada con una barrera de referencia.[4]
    *   **Modelos de Conteo de Enlaces (Enfoques Clásicos):** Estiman $E_a$ basándose en el número y tipo de enlaces químicos rotos y formados durante una transición atómica.[15, 16] La contribución energética de cada tipo de enlace es un parámetro derivado de datos experimentales o cálculos *ab initio*.[15]
    *   **Aprovechamiento de Datos Previos (Experimentales/Teóricos):** Los valores de $E_a$ pueden obtenerse directamente de estudios computacionales de alta fidelidad (como la Teoría del Funcional de la Densidad, DFT, utilizando métodos como NEB) o inferirse de mediciones experimentales (como coeficientes de difusión).[13, 17, 15]

*   **Frecuencias de Intento ($\Gamma_0$):**
    *   Representa la velocidad a la que un átomo intenta cruzar la barrera de energía, relacionada con las frecuencias vibracionales de los átomos.[5, 17, 10]
    *   **Teoría del Estado de Transición (TST):** $\Gamma_0$ puede expresarse como una relación de funciones de partición o frecuencias vibracionales.[6] Una simplificación común es asumir un prefactor constante, típicamente en el rango de $10^{12}$ a $10^{14} \text{ s}^{-1}$.[5, 10]
    *   **Valores Empíricos:** A menudo se asume un prefactor universal para todos los eventos en modelos KMC más simples.[10]
    *   **Cálculos DFT/Fonones:** Métodos más avanzados calculan modos vibracionales para derivar $\Gamma_0$ con mayor precisión.[17]


## Elección del Framework e Implementación

### Framework Seleccionado

Se ha implementado un framework **modular custom en Python** con las siguientes características:

#### Stack Tecnológico

- **Base**: Python 3.11+ con `uv` (gestor de paquetes moderno de Astral)
- **Simulación KMC**: Implementación custom optimizada con algoritmo BKL
- **RL Framework**: Stable-Baselines3 (PPO) + PyTorch
- **Ambiente**: Gymnasium (entorno personalizado)
- **Configuración**: Pydantic + Pydantic Settings
- **Calidad**: Ruff (linting + formatting) + Mypy (type checking)
- **Análisis**: NumPy, SciPy, scikit-image
- **Visualización**: Matplotlib

#### Estructura del Proyecto Implementada

```
kmc_oxidos/
├── src/
│   ├── settings/           # Configuración con Pydantic Settings
│   ├── kmc/                # Simulador KMC (lattice, events, rates, simulator)
│   ├── rl/                 # Módulo RL (environment, policy, critic, reweighting)
│   ├── analysis/           # Análisis morfológico (roughness, fractal, visualization)
│   └── data/               # Parámetros físicos TiO₂
├── experiments/            # Scripts de entrenamiento y simulación
├── notebooks/              # Jupyter notebooks de ejemplo
├── docs/                   # Documentación completa
├── results/                # Resultados de simulaciones
├── checkpoints/            # Modelos entrenados
└── logs/                   # Archivos de log
```

### Razones de la Elección

1. **Control Total**: Implementación custom permite adaptar el mecanismo SwarmThinkers específicamente para óxidos metálicos
2. **Modularidad**: Fácil extensión a otros óxidos (V₂O₅, etc.) o eventos
3. **Type Safety**: Type hints + Mypy para código robusto
4. **Configuración Profesional**: Pydantic Settings para gestión de parámetros
5. **Integración RL Nativa**: Gymnasium + Stable-Baselines3 para PPO
6. **Calidad de Código**: Ruff para mantener estándares profesionales
7. **Ecosystem Python**: Acceso a toda la ciencia de datos y ML

### Módulos Implementados

#### 1. Sistema de Configuración (`src/settings/`)
- ✅ Pydantic Settings para variables de entorno
- ✅ Configuración centralizada (KMC, RL, Hardware, Paths, Logging)
- ✅ Validación automática de parámetros
- ✅ Logger integrado con múltiples niveles

#### 2. Simulador KMC (`src/kmc/`)
- ✅ **`lattice.py`**: Red 3D para TiO₂ con vecindarios y PBC
- ✅ **`events.py`**: Catálogo de eventos (adsorción, difusión, desorción, reacción)
- ✅ **`rates.py`**: Tasas de Arrhenius con dependencia de coordinación
- ✅ **`simulator.py`**: Motor KMC con algoritmo BKL (Bortz-Kalos-Lebowitz)

#### 3. Reinforcement Learning (`src/rl/`)
- ✅ **`environment.py`**: Ambiente Gymnasium para TiO₂
- ✅ **`policy.py`**: Red Actor (MLP 5 capas, 256 unidades)
- ✅ **`critic.py`**: Red Crítico para función de valor
- ✅ **`reweighting.py`**: Mecanismo de reponderación + importance sampling

Arquitectura SwarmThinkers:
- Actor: Política descentralizada con observaciones locales
- Crítico: Función de valor centralizada
- Reweighting: $P(a) = \frac{\pi_\theta(a|o) \cdot \Gamma_a}{\sum_{a'} \pi_\theta(a'|o) \cdot \Gamma_{a'}}$
- Importance Sampling: Corrección de sesgo para mantener física

#### 4. Análisis Morfológico (`src/analysis/`)
- ✅ **`roughness.py`**: Cálculo de $W(L,t)$ y exponentes $\alpha$, $\beta$ de Family-Vicsek
- ✅ **`fractal.py`**: Dimensión fractal usando box-counting
- ✅ **`visualization.py`**: Plots 3D y evolución temporal

#### 5. Parámetros Físicos (`src/data/`)
- ✅ **`tio2_parameters.py`**: Parámetros completos para TiO₂ rutilo (110)
  - Energías de activación: $E_a^{diff}$, $E_a^{des}$
  - Frecuencias de intento: $\Gamma_0$
  - Energías de enlace: Ti-O, Ti-Ti, O-O
  - Parámetros de red
  - Soporte para múltiples superficies (rutilo 110/100, anatasa 101)

#### 6. Scripts de Experimentación (`experiments/`)
- ✅ **`train_policy.py`**: Entrenamiento PPO con callbacks
- ✅ **`run_simulations.py`**: Ejecución KMC clásico y análisis

### Características del Framework

#### Gestión con uv
```bash
# Instalación de dependencias
uv sync

# Ejecución con entorno virtual automático  
uv run python experiments/run_simulations.py

# Añadir nuevas dependencias
uv add <paquete>
```

#### Configuración Centralizada
Todas las configuraciones en `.env`:
```env
# KMC
LATTICE_SIZE_X=50
TEMPERATURE=600.0

# RL
RL_LEARNING_RATE=0.0005
RL_TOTAL_TIMESTEPS=1000000
```

Acceso type-safe:
```python
from src.settings import settings

temp = settings.kmc.temperature  # Type-checked
logger = settings.setup_logging()
```

#### Calidad de Código
```bash
# Formateo automático
uv run ruff format src/

# Linting
uv run ruff check src/

# Type checking
uv run mypy src/
```

### Documentación Completa

- ✅ **README.md**: Instalación, uso, ejemplos, arquitectura
- ✅ **CONTRIBUTING.md**: Guías para contribuidores
- ✅ **FRAMEWORK_SUMMARY.md**: Resumen técnico detallado
- ✅ **Docstrings**: Todos los módulos documentados (estilo Google)
- ✅ **Notebook de ejemplo**: `notebooks/01_ejemplo_kmc_basico.ipynb`

### Estado Actual

**✅ Implementación Completa - Listo para Investigación**

El framework está completamente implementado y funcional con:
- Simulador KMC robusto para TiO₂
- Integración RL con arquitectura SwarmThinkers
- Análisis morfológico completo
- Configuración profesional
- Documentación exhaustiva
- Ejemplos de uso

**Próximos Pasos:**
1. Validar parámetros energéticos con literatura
2. Ejecutar simulaciones de referencia
3. Entrenar política RL inicial
4. Comparar KMC clásico vs KMC-RL
5. Análisis de escalamiento dinámico