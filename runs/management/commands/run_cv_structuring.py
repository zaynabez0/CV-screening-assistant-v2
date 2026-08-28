"""CV structuring as a management command, launched by the trigger view via subprocess.

Structuring makes one LLM call per uploaded CV. Run inline it would hold the request open
for the whole batch, with the browser showing nothing — which is what it used to do. Same
thin-command pattern as the three stages.
"""

from django.core.management.base import BaseCommand, CommandError

from runs.models import ScreeningRun
from runs.services import StageNotReady, run_cv_structuring


class Command(BaseCommand):
    help = "Structure every validated-but-unstructured CV in a run."

    def add_arguments(self, parser):
        parser.add_argument("run_id", type=int)

    def handle(self, *args, **options):
        run_id = options["run_id"]
        try:
            run = ScreeningRun.objects.get(pk=run_id)
        except ScreeningRun.DoesNotExist as exc:
            raise CommandError(f"No ScreeningRun with id {run_id}") from exc

        try:
            summary = run_cv_structuring(run)
        except StageNotReady as exc:
            # A precondition failure, not a crash — record it where the UI can show it.
            run.mark_job_failed(str(exc))
            raise CommandError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            # The subprocess has no other way to report; without this the UI would
            # poll a "running" job forever. Note a single CV failing does NOT land here:
            # the service absorbs those and reports them in its finish message.
            run.mark_job_failed(f"{type(exc).__name__}: {exc}")
            raise

        self.stdout.write(self.style.SUCCESS(
            f"Structuring done: {summary['structured']} structured, {summary['failed']} failed."
        ))
