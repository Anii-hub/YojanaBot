"""
finder/management/commands/warmup_store.py

Pre-warms the configured retriever during Render build phase so the first
user request does not pay initialization cost.

Called automatically from build.sh:
    python manage.py warmup_store
"""

from __future__ import annotations

import time

from django.core.management.base import BaseCommand


class Command(BaseCommand):
    help = "Pre-warm the configured scheme retriever."

    def handle(self, *args, **options):
        self.stdout.write("=" * 60)
        self.stdout.write("  YojanaBot - Retriever Warmup")
        self.stdout.write("=" * 60)
        start = time.time()
        self.stdout.write("[warmup] Loading configured retriever...")

        try:
            from finder import rag_service  # noqa: PLC0415
            store = rag_service._get_store()

            if store is None:
                err = rag_service.store_error() or "unknown error"
                self.stderr.write("[warmup] FAILED: " + err)
                # Non-fatal: app will still start; first request will be slow.
                return

            count = store.collection.count()
            elapsed = time.time() - start
            self.stdout.write(
                self.style.SUCCESS(
                    "[warmup] Done in %.1fs - %d chunks indexed." % (elapsed, count)
                )
            )

        except Exception as exc:
            elapsed = time.time() - start
            self.stderr.write("[warmup] Error after %.1fs: %s" % (elapsed, exc))
            self.stderr.write("[warmup] App will still start; first request may be slow.")
