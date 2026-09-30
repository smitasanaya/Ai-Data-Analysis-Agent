"""
Phase (new): Dynamic Model Selection
Instead of hardcoding a Gemini model name (which keeps breaking as Google
retires/restricts models), this asks the API directly which models this
specific key can access, and picks the best one that supports the
features we need (generateContent + function calling).

Caches the result for the lifetime of the process so we're not calling
list_models() on every single request.
"""

from google.genai import types

# Preference order: try these first if available, since newer/flash
# models are cheaper and faster. Falls through to whatever IS available
# if none of these match.
PREFERRED_ORDER = [
    "gemini-flash-latest",
    "gemini-3.8-flash",
    "gemini-3.7-flash",
    "gemini-3.6-flash",
    "gemini-2.5-flash",
    "gemini-2.0-flash",
]

_cached_model = {}  # keyed by id(client) to support multiple clients/keys


def get_best_available_model(client) -> str:
    """
    Query the API for models this key can access, and return the name of
    the best one available (preferring flash models, since they're
    fastest/cheapest and sufficient for tool-calling + text generation).
    """
    cache_key = id(client)
    if cache_key in _cached_model:
        return _cached_model[cache_key]

    try:
        available = []
        for model in client.models.list():
            # Only consider models that support generateContent
            actions = getattr(model, "supported_actions", None) or getattr(model, "supported_generation_methods", None) or []
            name = model.name.replace("models/", "")
            available.append(name)

        # Try preferred models in order
        for candidate in PREFERRED_ORDER:
            if candidate in available:
                _cached_model[cache_key] = candidate
                return candidate

        # None of our preferred names matched — grab any flash model as a reasonable default
        flash_models = [m for m in available if "flash" in m.lower() and "image" not in m.lower() and "live" not in m.lower()]
        if flash_models:
            _cached_model[cache_key] = flash_models[0]
            return flash_models[0]

        # Last resort: whatever's first in the list
        if available:
            _cached_model[cache_key] = available[0]
            return available[0]

    except Exception:
        pass

    # If discovery itself fails (e.g. network issue), fall back to the alias
    # that Google keeps pointed at their current stable model.
    return "gemini-flash-latest"
