"""finder/context_processors.py

Injects the active language ('en' or 'hi') into every template context
so that base.html navbar / footer can switch language without each view
having to pass it explicitly.
"""

from __future__ import annotations


def lang(request) -> dict:
    """Return the active UI language from the session (defaults to 'en')."""
    return {"lang": request.session.get("lang", "en")}
