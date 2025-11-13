# SwarmThinkers Implementation Plan for kmc_oxidos

**Branch**: `feature/swarmthinkers-integration`  
**Objetivo**: Integrar framework SwarmThinkers para acelerar simulaciones KMC de deposición TiO₂

---

## 📋 Resumen Ejecutivo

SwarmThinkers es un framework de RL que trata a cada átomo/sitio como un agente que propone transiciones localmente. Una política centralizada aprende a priorizar eventos estructuralmente importantes, acelerando la simulación sin perder fidelidad física mediante importance sampling.

### Diferencias Clave vs Paper Original

| Aspecto | Paper (Fe-Cu difusión) | kmc_oxidos (TiO₂ deposición) |
|---------|------------------------|------------------------------|
| Proceso | Difusión de vacancias (cerrado) | Deposición + crecimiento (abierto) |
| Eventos | Solo DIFFUSION | ADSORPTION, DIFFUSION, REACTION, DESORPTION |
| Especies | 1 (vacancias) | 3 (Ti, O, VACANT) |
| Reacciones | ❌ No | ✅ Ti + 2O → TiO₂ |
| Barrera ES | ❌ No | ✅ Difusión descendente |

---

## 🎯 Fases de Implementación

### **Fase 1: Prototipo Diffusion-Only** ⬅️ EMPEZAMOS AQUÍ

**Objetivo**: Validar mecánica SwarmThinkers básica sin complejidad de múltiples eventos.

**Scope**:
- Solo eventos de difusión (DIFFUSION_TI, DIFFUSION_O)
- Política simple: propone K direcciones de difusión por átomo adsorbido
- Validación: comparar distribuciones vs KMC clásico en lattice pequeño

**Componentes a Implementar**:

1. **Observaciones Locales** (`src/rl/observations.py`):
   ```python
   def get_local_observation(lattice, site_idx) -> np.ndarray:
       """
       Returns:
           - [0:36]: especies 1st neighbors (one-hot: VACANT, TI, O)
           - [36:48]: alturas relativas (z_neighbor - z_site) para ES barrier
           - [48:50]: composición local (n_Ti, n_O)
           - [50]: altura z absoluta
       """
   ```

2. **SwarmPolicy Simple** (`src/rl/swarm_policy.py`):
   ```python
   class DiffusionSwarmPolicy(nn.Module):
       """Policy que solo propone direcciones de difusión."""
       def forward(self, obs) -> torch.Tensor:
           # Returns: logits para K direcciones (12 vecinos)
   ```

3. **SwarmEngine Básico** (`src/rl/swarm_engine.py`):
   ```python
   class SwarmEngine:
       def generate_diffusion_proposals(policy, lattice, n_swarm):
           # 1. Get adsorbed atoms
           # 2. Policy propone direcciones
           # 3. Calcula tasas con RateCalculator (incluye ES barrier)
           # 4. Reweighting: P(a) = π(a)·Γ_a / Z
           # 5. Select + importance weight
   ```

4. **Experimento Validación** (`experiments/validate_swarmthinkers_phase1.py`):
   - Lattice 20×20×10, temperatura 180K
   - Ejecutar 10k steps con KMC clásico
   - Ejecutar 10k steps con SwarmThinkers
   - Comparar: roughness, coverage, distribución de especies
   - Test estadístico: Kolmogorov-Smirnov para unbiasedness

**Criterios de Éxito Fase 1**:
- ✅ Importance weights convergen (ESS > 0.5)
- ✅ Distribuciones finales indistinguibles (p-value > 0.05 en KS test)
- ✅ SwarmThinkers completa simulación sin crashes
- ✅ Código documentado y testeado

---

### **Fase 2: Eventos Completos**

**Objetivo**: Extender a todos los tipos de eventos del sistema.

**Scope**:
- Agregar ADSORPTION_TI, ADSORPTION_O, DESORPTION_TI, DESORPTION_O, REACTION_TIO2
- Policy con múltiples heads (uno por tipo de evento)
- Action masking robusto

**Componentes Nuevos**:

1. **MultiEventPolicy** (`src/rl/swarm_policy.py`):
   ```python
   class TiO2SwarmPolicy(nn.Module):
       """Policy con heads especializados por tipo de evento."""
       - head_diffusion: K direcciones
       - head_adsorption: 2 especies (Ti, O)
       - head_desorption: 1 probabilidad
       - head_reaction: 1 probabilidad
   ```

2. **Action Masking** (`src/rl/action_masking.py`):
   ```python
   def get_valid_actions(agent_idx, lattice) -> Dict[ActionType, bool]:
       # VACANT_SURFACE -> solo ADSORPTION
       # TI_ADSORBED -> DIFFUSION + DESORPTION + REACTION (si 2+ O vecinos)
       # O_ADSORBED -> DIFFUSION + DESORPTION
   ```

3. **SwarmEngine Completo** (`src/rl/swarm_engine.py`):
   - Dispatch de tasas según tipo de evento
   - Soporte para reacciones multi-site
   - Global softmax sobre todos (agente, acción) pairs

**Validación Fase 2**:
- Comparar formación de TiO₂ con KMC clásico
- Verificar que reacciones ocurren en configuraciones correctas
- Measure effective transition ratio (ETR)

---

### **Fase 3: Training con RL**

**Objetivo**: Entrenar política para maximizar eficiencia manteniendo física.

**Scope**:
- Setup entrenamiento PPO
- Recompensa: `r_t = -ΔE_t` (minimizar energía)
- Critic centralizado con estadísticas globales
- Generalization: entrenar en 10×10×10, evaluar en 40×40×20

**Componentes**:

1. **SwarmEnvironment** (`src/rl/swarm_environment.py`):
   ```python
   class TiO2SwarmEnv(gym.Env):
       """Gymnasium env con SwarmEngine en el loop."""
       - Observations: local obs para cada agente activo
       - Actions: selección de evento vía swarm
       - Rewards: -ΔE por step
       - Info: importance weights, ESS
   ```

2. **Training Script** (`experiments/train_swarm_policy.py`):
   - PPO con Stable-Baselines3
   - Entropy regularization para exploración
   - Checkpoints cada 10k steps
   - Tensorboard logging

3. **Métricas de Performance**:
   - Speedup ratio: steps_KMC / steps_swarm para igual evolución
   - Effective transition ratio (ETR): eventos productivos / total
   - Memory usage
   - Walltime per step

**Objetivos de Performance**:
- 🎯 Speedup > 10× en lattices grandes (>40×40×40)
- 🎯 ETR > 0.1 (vs < 0.001 en KMC clásico)
- 🎯 Memory < 2 GB para 50×50×30 lattice

---

## 🏗️ Arquitectura de Código

### Estructura de Archivos Nueva

```
src/rl/
├── __init__.py              # Actualizar exports
├── observations.py          # 🆕 Local observation extraction
├── swarm_policy.py          # 🆕 DiffusionSwarmPolicy + TiO2SwarmPolicy
├── swarm_engine.py          # 🆕 SwarmEngine core logic
├── action_masking.py        # 🆕 Valid actions per agent type (Fase 2)
├── swarm_environment.py     # 🆕 Gymnasium env (Fase 3)
├── policy.py                # ✅ Mantener para baseline
├── critic.py                # ✅ Mantener, usar en Fase 3
├── reweighting.py           # ✅ Ya existe, reutilizar
└── environment.py           # ✅ Mantener para comparación

experiments/
├── validate_swarmthinkers_phase1.py  # 🆕 Validación diffusion-only
├── validate_swarmthinkers_phase2.py  # 🆕 Validación multi-evento
├── train_swarm_policy.py             # 🆕 Training PPO (Fase 3)
└── compare_swarm_vs_classic.py       # 🆕 Benchmarks completos
```

### Principios de Diseño

1. **No modificar `src/kmc/`**: Módulo KMC permanece puro y clásico
2. **Composition over Inheritance**: SwarmEngine compone KMCSimulator, no hereda
3. **Separación de concerns**:
   - `swarm_engine.py`: Lógica de swarm (propuestas, reweighting, selection)
   - `swarm_policy.py`: Redes neuronales
   - `observations.py`: Feature engineering
   - `action_masking.py`: Validación de acciones
4. **Testabilidad**: Cada componente con unit tests
5. **Reproducibilidad**: Seeds fijos en experimentos de validación

---

## 📊 Validación y Métricas

### Correctness (Physics Fidelity)

**Test de Unbiasedness**:
```python
# Ejecutar N trials con ambos métodos
roughness_classic = [run_kmc_classic() for _ in range(50)]
roughness_swarm = [run_swarm() for _ in range(50)]

# Kolmogorov-Smirnov test
ks_statistic, p_value = ks_2samp(roughness_classic, roughness_swarm)
assert p_value > 0.05, "Distributions differ significantly"
```

**Métricas Físicas**:
- Roughness evolution W(t)
- Coverage θ(t)
- Composición (Ti/O ratio)
- Formación TiO₂ (# moléculas vs tiempo)
- Exponentes de scaling α, β

### Performance

**Speedup Ratio**:
```
SR = steps_classic_needed / steps_swarm_needed
```
donde ambos alcanzan misma configuración morfológica.

**Effective Sample Size (ESS)**:
```
ESS = (Σ w_i)² / Σ w_i²
```
donde w_i son importance weights. ESS > 0.5 indica buen sampling.

**Effective Transition Ratio (ETR)**:
```
ETR = eventos_productivos / total_eventos
```
donde evento productivo = causa cambio estructural (no reversible inmediato).

---

## 🚀 Ejecución Fase 1

### Checklist de Implementación

- [ ] Crear `src/rl/observations.py` con `get_local_observation()`
- [ ] Crear `src/rl/swarm_policy.py` con `DiffusionSwarmPolicy`
- [ ] Crear `src/rl/swarm_engine.py` con lógica básica de swarm
- [ ] Crear `experiments/validate_swarmthinkers_phase1.py`
- [ ] Ejecutar validación en lattice 20×20×10
- [ ] Analizar resultados: KS test, ESS, visual comparison
- [ ] Documentar findings en `docs/swarmthinkers_phase1_results.md`
- [ ] Commit y push a branch
- [ ] Si todo OK → merge a master y pasar a Fase 2

### Comandos de Ejecución

```powershell
# Validación Fase 1
uv run python -m experiments.validate_swarmthinkers_phase1 `
    --lattice-size 20 20 10 `
    --temperature 180 `
    --max-steps 10000 `
    --n-trials 50 `
    --swarm-size 32 `
    --seed 42

# Genera outputs:
# - results/swarmthinkers_phase1/comparison_plots.png
# - results/swarmthinkers_phase1/ks_test_results.json
# - results/swarmthinkers_phase1/importance_weights_evolution.png
```

---

## 📚 Referencias

- **Paper Original**: SwarmThinkers: Learning Physically Consistent Atomic KMC Transitions at Scale (Li et al., 2025)
- **Código Base**: `src/kmc/` (KMC clásico ya implementado)
- **RL Framework**: `src/rl/` (ActorNetwork, CriticNetwork, ReweightingMechanism ya existen)
- **Documentación Proyecto**: `docs/FRAMEWORK_SUMMARY.md`, `docs/agent/GUIDEMENT.md`

---

## ✅ Criterios de Éxito General

**Fase 1 (Prototipo)**:
- ✅ No crashes durante simulación
- ✅ Importance weights estables (ESS > 0.5)
- ✅ Distribuciones físicas correctas (KS p-value > 0.05)

**Fase 2 (Multi-Evento)**:
- ✅ Todos los tipos de eventos funcionan
- ✅ Action masking correcto (no propuestas inválidas)
- ✅ Reacciones TiO₂ ocurren en configuraciones esperadas

**Fase 3 (Training)**:
- ✅ Policy aprende (reward aumenta durante training)
- ✅ Speedup > 10× vs KMC clásico
- ✅ Generaliza a lattices más grandes sin reentrenar

---

**Notas**:
- Este plan es iterativo: ajustaremos según resultados de cada fase
- Prioridad = correctness > speedup
- Documentar decisiones de diseño en commits
