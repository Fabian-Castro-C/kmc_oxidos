# Contributing to KMC-Óxidos

¡Gracias por tu interés en contribuir al proyecto! Este documento proporciona pautas para contribuciones.

## 🎯 Código de Conducta

Se espera que todos los contribuidores mantengan un ambiente respetuoso y profesional.

## 🚀 Cómo Contribuir

### Reportar Bugs

1. Verificar que el bug no haya sido reportado previamente
2. Incluir información detallada:
   - Versión de Python
   - Sistema operativo
   - Pasos para reproducir
   - Comportamiento esperado vs actual

### Proponer Nuevas Características

1. Abrir un issue describiendo la característica
2. Explicar el caso de uso
3. Esperar feedback antes de implementar

### Pull Requests

1. Fork del repositorio
2. Crear rama desde `main`: `git checkout -b feature/mi-feature`
3. Hacer commits con mensajes descriptivos
4. Asegurar que el código:
   - Pasa ruff: `uv run ruff check src/`
   - Pasa mypy: `uv run mypy src/`
   - Está documentado con docstrings
   - Incluye type hints
5. Push y crear PR con descripción detallada

## 📝 Estándares de Código

### Python Style

- Seguir PEP 8
- Usar ruff para formateo: `uv run ruff format src/`
- Line length: 100 caracteres

### Type Hints

```python
def calculate_rate(
    energy: float,
    temperature: float,
) -> float:
    """
    Calculate rate using Arrhenius equation.

    Args:
        energy: Activation energy in eV.
        temperature: Temperature in Kelvin.

    Returns:
        Rate in Hz.
    """
    ...
```

### Docstrings

Usar Google style:

```python
def function(arg1: int, arg2: str) -> bool:
    """
    Short description.

    Longer description if needed.

    Args:
        arg1: Description of arg1.
        arg2: Description of arg2.

    Returns:
        Description of return value.

    Raises:
        ValueError: When something is wrong.
    """
    ...
```

## 🔧 Desarrollo Local

```bash
# Clonar
git clone https://github.com/tu-usuario/kmc_oxidos.git
cd kmc_oxidos

# Instalar dependencias de desarrollo
uv sync

# Activar entorno
source .venv/bin/activate

# Verificar código
uv run ruff check src/
uv run mypy src/
```

## 📦 Estructura de Commits

```
tipo(alcance): descripción breve

Descripción detallada opcional.

Fixes #123
```

Tipos:
- `feat`: Nueva característica
- `fix`: Bug fix
- `docs`: Documentación
- `style`: Formateo
- `refactor`: Refactorización
- `perf`: Mejora de rendimiento
- `test`: Tests

## 🙏 Agradecimientos

Tu contribución es muy valiosa. ¡Gracias por hacer este proyecto mejor!
