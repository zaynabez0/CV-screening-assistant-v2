"""Stage 2 as a management command, launched by the trigger view via subprocess.

Stage 2 makes one LLM call per survivor, so running it inside the request/response
cycle would block the browser for the whole batch. Same thin-command pattern as Stage 1.
"""

from django.core.management.base import BaseCommand, CommandError

from runs.models import ScreeningRun
from runs.services import StageNotReady, run_stage2


class Command(BaseCommand):
    help = "Run Stage 2 (ranking) across every Stage 1 survivor in a run."

    def add_arguments(self, parser):
        parser.add_argument("run_id", type=int)

    def handle(self, *args, **options):
        run_id = options["run_id"]
        try:
            run = ScreeningRun.objects.get(pk=run_id)
        except ScreeningRun.DoesNotExist as exc:
            raise CommandError(f"No ScreeningRun with id {run_id}") from exc

        try:
            summary = run_stage2(run)
        except StageNotReady as exc:
            run.mark_job_failed(str(exc))
            raise CommandError(str(exc)) from exc
        except Exception as exc:  # noqa: BLE001 — the subprocess has no other way to report
            run.mark_job_failed(f"{type(exc).__name__}: {exc}")
            raise

        self.stdout.write(self.style.SUCCESS(
            f"Stage 2 done: {summary['ranked']} ranked, {summary['shortlisted']} above the cutoff."
        ))
