"""Custom Django template filters for the finder app."""

from django import template

register = template.Library()


@register.filter(name="split")
def split_filter(value, sep="·"):
    """Split a string by a separator and return a list.
    
    Usage: {{ value|split:"·" }}
    """
    if not value:
        return []
    return [part for part in str(value).split(sep)]


@register.filter(name="strip")
def strip_filter(value):
    """Strip leading/trailing whitespace from a string.
    
    Usage: {{ value|strip }}
    """
    if value is None:
        return ""
    return str(value).strip()
