from django.apps import AppConfig
import logging
import os
import sys

log = logging.getLogger(__name__)


class FinderConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'finder'

    def ready(self):
        """Pre-load the vector store before gunicorn forks workers.

        With 'gunicorn --preload', ready() runs in the master process.
        Workers inherit the loaded _store via Linux copy-on-write fork,
        so the 400 MB PyTorch model is loaded exactly ONCE instead of
        once per worker per cold-start request.

        Skipped during manage.py commands (migrate, collectstatic, etc.)
        to avoid slow startups and memory spikes during the build phase.
        """
        # Only pre-load under the actual WSGI server, not manage.py commands.
        # manage.py always passes a sub-command as argv[1].
        _is_manage_cmd = (
            len(sys.argv) > 1
            and not sys.argv[0].endswith('gunicorn')
            and 'wsgi' not in ''.join(sys.argv)
        )
        if _is_manage_cmd:
            return

        # Also skip if explicitly disabled (useful in CI / testing).
        if os.environ.get('YOJANA_SKIP_PRELOAD', '').lower() == 'true':
            return

        log.info("[finder.ready] Pre-loading vector store...")
        try:
            from finder import rag_service  # noqa: PLC0415
            store = rag_service._get_store()
            if store is None:
                log.warning("[finder.ready] Store pre-load failed: %s", rag_service.store_error())
            else:
                log.info("[finder.ready] Store ready — %d chunks.", store.collection.count())
        except Exception as exc:
            log.warning("[finder.ready] Store pre-load error (non-fatal): %s", exc)
