from django.apps import AppConfig
import logging
import os
import sys
import threading

log = logging.getLogger(__name__)


class FinderConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'finder'

    def ready(self):
        """Kick off a background thread to pre-warm the vector store.

        The thread loads the embedding model + ChromaDB index in the worker
        process AFTER gunicorn has already bound to the port, so Render's
        port-scan check succeeds immediately.

        The _store_lock inside rag_service guarantees that any /results/
        request arriving before the thread finishes will simply wait at the
        lock rather than crashing or loading the model a second time.

        Skipped during manage.py commands (migrate, collectstatic, warmup_store)
        to avoid slow build-phase startups.
        """
        # Skip for manage.py sub-commands (migrate, collectstatic, etc.)
        _is_manage_cmd = (
            len(sys.argv) > 1
            and not sys.argv[0].endswith('gunicorn')
            and 'wsgi' not in ' '.join(sys.argv)
        )
        if _is_manage_cmd:
            return

        # Allow explicit opt-out (e.g. during build or testing)
        if os.environ.get('YOJANA_SKIP_PRELOAD', '').lower() == 'true':
            return

        def _load():
            log.info('[finder.ready] Background preload: loading vector store...')
            try:
                from finder import rag_service  # noqa: PLC0415
                store = rag_service._get_store()
                if store is None:
                    log.warning('[finder.ready] Preload failed: %s', rag_service.store_error())
                else:
                    log.info('[finder.ready] Preload done — %d chunks ready.', store.collection.count())
            except Exception as exc:
                log.warning('[finder.ready] Preload error (non-fatal): %s', exc)

        t = threading.Thread(target=_load, daemon=True, name='store-preload')
        t.start()
        log.info('[finder.ready] Store preload thread started.')
