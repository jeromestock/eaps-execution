from __future__ import annotations

import asyncio

from apscheduler import Job, Schedule

from eaps_execution.src.Execution.util import setup_logger

logger = setup_logger("production.log")


class ExecutionHandler:
    """
    The ExecutionHandler is used to handle the data that is received for the execution.
    """

    def __init__(self) -> None:
        #: The event that is set when the node value for the step is executed.
        self.event: asyncio.Event = asyncio.Event()

        #: The schedule states that are received from the scheduler.
        self.schedule_data_store: list[Schedule] = []

        #: Jobs that are executed.
        self.jobs_executed: dict = {}

    def datachange_scheduler_notification(
        self,
        data_store: list,
        current_job: Job | None = None,
    ) -> None:
        """
        Callback for the scheduler subscription.
        This method will be called when the scheduler executed a process step.

        :param data_store (List[Schedule]): The new data store from the scheduler
        :param current_job (Job | None): The current job that is executed in the machine, defaults to None

        :return: None
        """
        self.schedule_data_store = data_store

        if current_job.args[1] in self.jobs_executed:
            self.jobs_executed[current_job.args[1]].append(
                (
                    current_job.args[2],
                    current_job.args[3],
                    current_job.args[4],
                )
            )
        else:
            self.jobs_executed[current_job.args[1]] = [
                (
                    current_job.args[2],
                    current_job.args[3],
                    current_job.args[4],
                )
            ]
