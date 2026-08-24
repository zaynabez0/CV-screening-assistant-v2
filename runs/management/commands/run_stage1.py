"""Stage 1 as a management command, launched by the trigger view via subprocess.

Deliberately thin: it exists so the LLM work happens outside the request/response
cycle. Everything it does lives in runs.services, which the tests call directly.
"""

from django.core.management.base import BaseCommand, CommandError

from runs.models import ScreeningRun
from runs.services import StageNotReady, run_stage1


class Command(BaseCommand):
    help = "Run Stage 1 (hard filter) across every structured candidate in a run."

    def add_arguments(self, parser):
        parser.add_argument("run_id", type=int)

    def handle(self, *args, **options):
        run_id = options["run_id"]
        try:
            run = ScreeningRun.objects.get(pk=run_id)
        except ScreeningRun.DoesNotExist as exc:
            raise CommandError(f"No ScreeningRun with id {run_id}") from exc

        try:
            summary = run_stage1(run)
        except StageNotReady as exc:
            # A precondition failure, not a crash — record it where the UI can show it.
            run.mark_job_failed(str(exc))
            raise CommandError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001
            # The subprocess has no other way to report; without this the UI would
            # poll a "running" job forever.
            run.mark_job_failed(f"{type(exc).__name__}: {exc}")
            raise

        self.stdout.write(self.style.SUCCESS(
            f"Stage 1 done: {summary['passed']} passed, {summary['failed']} failed "
            f"of {summary['evaluated']} evaluated."
        ))
