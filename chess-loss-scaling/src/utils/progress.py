"""Progress bar utilities using rich."""
from typing import Optional

from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def create_progress_bar() -> Progress:
    """
    Create a rich progress bar with useful columns.

    Returns:
        Configured Progress instance
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TextColumn("•"),
        TimeElapsedColumn(),
        TextColumn("•"),
        TimeRemainingColumn(),
    )


class ProgressTracker:
    """Context manager for progress tracking."""

    def __init__(self):
        self.progress: Optional[Progress] = None
        self.tasks: dict[str, TaskID] = {}

    def __enter__(self):
        self.progress = create_progress_bar()
        self.progress.__enter__()
        return self

    def __exit__(self, *args):
        if self.progress:
            self.progress.__exit__(*args)

    def add_task(self, description: str, total: int) -> str:
        """Add a new task to track."""
        if not self.progress:
            raise RuntimeError("ProgressTracker not initialized")
        task_id = self.progress.add_task(description, total=total)
        self.tasks[description] = task_id
        return description

    def update(self, task_name: str, advance: int = 1):
        """Update progress for a task."""
        if not self.progress:
            raise RuntimeError("ProgressTracker not initialized")
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} not found")
        self.progress.update(self.tasks[task_name], advance=advance)

    def complete(self, task_name: str):
        """Mark a task as complete."""
        if not self.progress:
            raise RuntimeError("ProgressTracker not initialized")
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} not found")
        self.progress.update(self.tasks[task_name], completed=True)
