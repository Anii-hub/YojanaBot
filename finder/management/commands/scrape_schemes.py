from django.core.management.base import BaseCommand

from data_pipeline.step0_auto_scraper import run


class Command(BaseCommand):
    help = "Scrape and index new government scheme PDFs"

    def add_arguments(self, parser):
        parser.add_argument("--dry-run", action="store_true")
        parser.add_argument("--manual-url", action="append", dest="manual_urls")

    def handle(self, *args, **options):
        result = run(manual_urls=options.get("manual_urls"), dry_run=options["dry_run"])
        self.stdout.write(result.summary())
