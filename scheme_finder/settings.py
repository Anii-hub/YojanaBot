"""
Django settings for scheme_finder project.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ── Detect environment (Render or Railway) ────────────────────────────────────
RENDER_EXTERNAL_HOSTNAME = os.environ.get("RENDER_EXTERNAL_HOSTNAME", "")
RAILWAY_PUBLIC_DOMAIN = os.environ.get("RAILWAY_PUBLIC_DOMAIN", "")
RAILWAY_STATIC_URL = os.environ.get("RAILWAY_STATIC_URL", "")

# Determine production hostname
PROD_HOSTNAME = RENDER_EXTERNAL_HOSTNAME or RAILWAY_PUBLIC_DOMAIN or RAILWAY_STATIC_URL

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = os.environ.get(
    "DJANGO_SECRET_KEY",
    "django-insecure-rag-scheme-finder-dev-key-change-in-production",
)

DEBUG = os.environ.get("DJANGO_DEBUG", "True") == "True"

# Allow localhost in dev; in production allow all hosts (Render's proxy
# handles real host validation — restricting ALLOWED_HOSTS here just causes
# spurious 400/500 errors when the injected RENDER_EXTERNAL_HOSTNAME doesn't
# match the Host header exactly).
_allowed = ["localhost", "127.0.0.1"]
if PROD_HOSTNAME:
    clean_host = PROD_HOSTNAME.replace("https://", "").replace("http://", "")
    _allowed.append(clean_host)
# In production fall back to ["*"] so Render proxy headers never cause 400s
if os.environ.get("ALLOWED_HOSTS"):
    ALLOWED_HOSTS = os.environ["ALLOWED_HOSTS"].split(",")
elif not DEBUG:
    ALLOWED_HOSTS = ["*"]
else:
    ALLOWED_HOSTS = _allowed

# Required for Django 4.x CSRF protection with HTTPS
CSRF_TRUSTED_ORIGINS = []
if PROD_HOSTNAME:
    clean_host = PROD_HOSTNAME.replace("https://", "").replace("http://", "")
    CSRF_TRUSTED_ORIGINS.append(f"https://{clean_host}")
    CSRF_TRUSTED_ORIGINS.append(f"http://{clean_host}")
# Broad fallback: trust all onrender.com and railway.app origins
if not CSRF_TRUSTED_ORIGINS and not DEBUG:
    CSRF_TRUSTED_ORIGINS = [
        "https://*.onrender.com",
        "https://*.railway.app",
        "https://*.up.railway.app",
    ]

# Security settings (only active when DEBUG=False)
SECURE_PROXY_SSL_HEADER = ("HTTP_X_FORWARDED_PROTO", "https")
SECURE_SSL_REDIRECT = not DEBUG
SESSION_COOKIE_SECURE = not DEBUG
CSRF_COOKIE_SECURE = not DEBUG

# ---------------------------------------------------------------------------
# Application definition
# ---------------------------------------------------------------------------

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "finder",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    "whitenoise.middleware.WhiteNoiseMiddleware",   # serve static files in dev+prod
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "scheme_finder.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "finder" / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.debug",
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "finder.context_processors.lang",
            ],
        },
    },
]

WSGI_APPLICATION = "scheme_finder.wsgi.application"

# ---------------------------------------------------------------------------
# Database — SQLite is fine (we don't store user data)
# ---------------------------------------------------------------------------

DATABASES = {
    "default": {
        "ENGINE": "django.db.backends.sqlite3",
        "NAME": BASE_DIR / "db.sqlite3",
    }
}

# ---------------------------------------------------------------------------
# Internationalization
# ---------------------------------------------------------------------------

LANGUAGE_CODE = "en-us"
TIME_ZONE = "Asia/Kolkata"
USE_I18N = True
USE_TZ = True

# ---------------------------------------------------------------------------
# Static files
# ---------------------------------------------------------------------------

STATIC_URL = "/static/"
STATIC_ROOT = BASE_DIR / "staticfiles"
STATICFILES_DIRS = [BASE_DIR / "finder" / "static"]
# CompressedStaticFilesStorage (no manifest) avoids KeyError 500s when a
# static file referenced in a template doesn't appear in the manifest.
# Switch back to CompressedManifestStaticFilesStorage once all assets are stable.
STATICFILES_STORAGE = "whitenoise.storage.CompressedStaticFilesStorage"

# ---------------------------------------------------------------------------
# RAG pipeline settings (read from .env)
# ---------------------------------------------------------------------------

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.1-8b-instant")
CHROMA_DIR = BASE_DIR / "data" / "chroma"
RAG_TOP_K = 5

# ── HuggingFace / sentence-transformers model cache ──────────────────────────
# Default to /tmp so downloads work on ephemeral filesystems (e.g. Render free
# tier which has no persistent disk). On a paid tier with a persistent disk,
# override HF_HOME and SENTENCE_TRANSFORMERS_HOME via environment variables
# (already done in render.yaml for the starter/standard plans).
_hf_cache = os.environ.get("HF_HOME", "/tmp/huggingface")
os.environ.setdefault("HF_HOME", _hf_cache)
os.environ.setdefault("TRANSFORMERS_CACHE", _hf_cache)
os.environ.setdefault(
    "SENTENCE_TRANSFORMERS_HOME",
    os.environ.get("SENTENCE_TRANSFORMERS_HOME", "/tmp/sentence_transformers"),
)

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

# ---------------------------------------------------------------------------
# Logging — show errors in Render/Railway log stream even with DEBUG=False
# ---------------------------------------------------------------------------

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "verbose": {
            "format": "{levelname} {asctime} {module} {process:d} {thread:d} {message}",
            "style": "{",
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "verbose",
        },
    },
    "root": {
        "handlers": ["console"],
        "level": "WARNING",
    },
    "loggers": {
        "django": {
            "handlers": ["console"],
            "level": "INFO" if DEBUG else "WARNING",
            "propagate": False,
        },
        "django.request": {
            "handlers": ["console"],
            "level": "ERROR",
            "propagate": False,
        },
        "finder": {
            "handlers": ["console"],
            "level": "DEBUG",
            "propagate": False,
        },
    },
}
