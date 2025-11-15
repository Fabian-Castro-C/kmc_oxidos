# Sistema Abierto vs Cerrado - Deposición Balística TiO₂

## Problema Identificado

El cálculo de energía original trataba el sistema como **CERRADO**, calculando:
```
E_total = E_bonds + E_surface + E_formation
```

Esto es **incorrecto** para deposición balística, que es un **SISTEMA ABIERTO** con reservorio de partículas.

## Corrección: Gran Ensemble Canónico

### Física Correcta

En deposición balística:
- **Átomos vienen de un reservorio** a potencial químico µ
- **Gran potencial**: Ω = E - µN (no solo E)
- **Cambio relevante**: ΔΩ, no ΔE_total

### Eventos y Energía

1. **Adsorción** (N aumenta):
   ```
   ΔΩ_ads = ΔE_bonds - µ
   ```
   - Costo de traer átomo del reservorio: -µ
   - Ganancia por formar enlaces: ΔE_bonds < 0
   - Con µ ≈ 0 (beam energético): ΔΩ ≈ ΔE_bonds

2. **Difusión** (N constante):
   ```
   ΔΩ_diff = ΔE_local
   ```
   - Solo cambio en enlaces locales
   - No entra/sale del reservorio

3. **Desorción** (N disminuye):
   ```
   ΔΩ_des = -ΔE_bonds + µ
   ```
   - Romper enlaces: +|ΔE_bonds|
   - Devolver átomo al reservorio: +µ

### Simplificación para Ballistic Deposition

En deposición balística con **beam energético**:
- µ_Ti ≈ 0 eV (referencia: átomos gas aislados)
- µ_O ≈ 0 eV

Por lo tanto: **ΔΩ ≈ ΔE_local**

## Implementación Actualizada

```python
class SystemEnergyCalculator:
    """
    Calcula ΔΩ para sistema abierto (deposición balística).
    
    Con µ ≈ 0: ΔΩ ≈ ΔE_local (solo energía de enlaces)
    """
    
    def __init__(self):
        self.mu_ti = 0.0  # Potencial químico Ti (eV)
        self.mu_o = 0.0   # Potencial químico O (eV)
        
    def calculate_local_energy(self, lattice, site_idx):
        """Energía LOCAL: suma de enlaces con vecinos."""
        local_E = 0.0
        for neighbor in site.neighbors:
            if neighbor.is_occupied():
                local_E += bond_energy(site, neighbor) / 2.0
        return local_E
    
    def calculate_system_energy(self, lattice):
        """Energía total: suma de energías locales."""
        return sum(self.calculate_local_energy(lattice, i) 
                   for i in range(len(lattice.sites)))
```

## Diferencias Clave

| Aspecto | Sistema Cerrado (❌ Incorrecto) | Sistema Abierto (✅ Correcto) |
|---------|--------------------------------|------------------------------|
| **Variable termodinámica** | E (energía) | Ω = E - µN (gran potencial) |
| **Componentes** | E_bonds + E_surface + E_formation | Solo E_bonds (con µ ≈ 0) |
| **Penalidad superficie** | Sí (dangling bonds) | No (siempre hay superficie en deposición) |
| **Bonus estequiometría** | Sí (global TiO₂ ratio) | No (composición determinada por reservorio) |
| **Adsorción** | Solo ΔE_bonds | ΔE_bonds - µ |
| **Físicamente representa** | Molécula aislada | Crecimiento desde fase vapor |

## Reward RL

Con la corrección:

```python
r_t = -ΔΩ_t ≈ -ΔE_local
```

**Interpretación física:**
- Formar enlace Ti-O (-4.5 eV) → reward +4.5 eV → ✅ favorable
- Romper enlace → reward negativo → ❌ desfavorable
- Adsorción sin vecinos → reward ≈ 0 (µ ≈ 0)
- Adsorción cerca de otros átomos → reward > 0 (forma enlaces)

## Bugs Corregidos

### Bug 1: Adsorción en sitio ocupado
```python
# ❌ ANTES: Intentaba ocupar el sitio del agente mismo
if action == ADSORB_TI:
    if not site.is_occupied():  # ¡Siempre False para agentes!
        site.species = TI

# ✅ AHORA: Adsorbe ENCIMA del sitio actual
if action == ADSORB_TI:
    target_site = lattice.get_site(x, y, z+1)  # Busca arriba
    if target_site and not target_site.is_occupied():
        target_site.species = TI
```

### Bug 2: Energía global vs local
```python
# ❌ ANTES: Energía total del sistema (sistema cerrado)
E_total = bond_energy + surface_penalty + formation_bonus

# ✅ AHORA: Solo energía de enlaces (sistema abierto)
E_total = sum(local_bond_energies)
```

## Referencias

- Gran ensemble canónico: Ω(T, V, µ) = E - µN
- Deposición balística: µ ≈ 0 (beam energético, no equilibrio térmico)
- KMC rates: Γ_ads = ν₀ * exp(-E_barrier/kT) con E_barrier ≈ 0 para adsorción
