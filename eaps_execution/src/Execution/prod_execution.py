from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

import pandas as pd
from apscheduler import (
    AsyncScheduler,
    current_job,
)
from apscheduler.triggers.date import DateTrigger
from apscheduler.triggers.interval import (
    IntervalTrigger,
)
from tzlocal import get_localzone

from eaps_execution.src.Execution.prod_connection import ExecutionHandler

if TYPE_CHECKING:
    from datetime import datetime


logging.basicConfig()
logging.getLogger("apscheduler").setLevel(logging.DEBUG)


class ExecutionProcessJobError(Exception):
    def __init__(
        self,
        message="An error occured in the machines during the execution of the production schedule",
    ) -> None:
        super().__init__(message)


class ExecuteProductionSchedule:
    def __init__(
        self,
        prod_schedule_file: str | Path | None = None,
        handler=ExecutionHandler,
    ) -> None:
        #: File in which the production schedule is stored.
        self._prod_schedule_file = prod_schedule_file

        #: Production schedule as DataFrame if the production schedule is defined, otherwise None.
        self.production_schedule: pd.DataFrame | None = self._setup_production_schedule()

        #: Beginning time of production schedule.
        self.episode_time_start: datetime | int

        #: Ending time of the production schedule.
        self.episode_time_end: datetime

        #: Schedule duration of the production schedule.
        self.schedule_duration: int

        #: Scheduler to execute the production schedule
        self.scheduler = AsyncScheduler()

        #: Handler for the subscription
        self.handler = handler

    def _setup_production_schedule(
        self,
        production_schedule: pd.DataFrame | None = None,
    ) -> pd.DataFrame | None:
        local_tz = get_localzone()

        production_schedule: pd.DataFrame | None = (
            pd.read_csv(
                self._prod_schedule_file,
                sep=",",
                parse_dates=[
                    "working_starttime",
                    "origin_starttime",
                    "origin_endtime",
                ],
            )
            if isinstance(self._prod_schedule_file, Path) and not isinstance(production_schedule, pd.DataFrame)
            else production_schedule
        )

        # Function for adding a time zone and converting to datetime.datetime
        def convert_to_datetime_with_tz(column):
            # Edit only the columns that contain dates
            if pd.api.types.is_datetime64_any_dtype(column):
                # Only localize if tz-naive, do nothing if tz-aware
                return column.dt.tz_localize(local_tz) if column.dt.tz is None else column
            return column

        return (
            production_schedule.apply(convert_to_datetime_with_tz)
            if isinstance(production_schedule, pd.DataFrame)
            else production_schedule
        )

    def _init_times(
        self,
        episode_time_start: datetime | str | None = None,
        episode_time_end: datetime | str | None = None,
        is_index: bool = False,
    ) -> None:
        """Initialize the episode_time_start, episode_end_time and the schedule_duration.

        :param episode_time_start: Start time of the scenario as datetime or string.
        :param episode_time_end: Endtime of the scenario as datetime or string.
        :param is_index: If True, the production schedule is assumed to have datetime
            indices, otherwise it is assumed to have strings.
        """
        assert self.production_schedule is not pd.DataFrame, "Production schedule is not initialized."

        def _parse_episode_time(
            default_time,
            input_time: datetime | str | None = None,
        ) -> datetime:
            """Convert input_time to dateime object.

            :param input_time: Input_time to be converted.
            :param default_time: Fallback time from the DataFrame.
            :return: Converted datetime
            """
            if isinstance(input_time, datetime):
                return input_time

            if isinstance(input_time, str):
                return datetime.strptime(input_time, "%Y-%m-%d %H:%M").replace(
                    tzinfo=ZoneInfo(time.tzname[time.localtime().tm_isdst])
                )
            return pd.to_datetime(default_time).to_pydatetime()

        self.episode_time_end = _parse_episode_time(
            self.production_schedule["endtime"].iloc[-1],
            episode_time_end,
        )

        self.episode_time_start = _parse_episode_time(
            self.production_schedule["starttime"].iloc[0],
            episode_time_start,
        )

        self.schedule_duration = (
            (self.episode_time_end - self.episode_time_start).total_seconds
            if not is_index
            else (self.episode_time_end - self.episode_time_start)
        )

        # Check if episode begin and end times make sense.
        if self.episode_time_start > self.episode_time_end:
            raise ValueError("Start time of the scenario should be smaller than or equal to endtime.")

        # Check if is_index is set correctly and make sense.
        if (
            not isinstance(self.episode_time_start, datetime)
            and not isinstance(self.episode_time_end, datetime)
            and is_index is False
        ):
            raise ValueError("Bool 'is_index' cannot be false if production schedule has indices for start times")

    def set_production_schedule(self, production_schedule: pd.DataFrame) -> None:
        """
        Set the production schedule DataFrame directly.

        :param production_schedule: The production schedule as a DataFrame.
        """
        self.production_schedule = self._setup_production_schedule(production_schedule)

        self._init_times()

    async def execute_production_process(
        self,
        job,
        machine,
        endtime,
        capacity,
        *args,
    ) -> None:
        """Execute one process step from the production schedule.

        This function is called by the scheduler after the time for the current process step has
        come. It removes the current job from the scheduler, executes the process step and adds
        the next job to the scheduler if there is one. Additionally, it sends the updated
        scheduler to the ExecutionHandler.

        :param job: ID of the job to be executed.
        :param machine: ID of the machine to execute the job on.
        :param endtime: Endtime of the process step.
        :param capacity: Capacity of products during the execution of the job on the machine.
        :param args: Additional arguments that are not used in this function.
        """
        self.handler.datachange_scheduler_notification(
            data_store=self.scheduler.data_store._schedules,
            current_job=current_job.get(),
        )

        await self.scheduler.remove_schedule(current_job.get().id)

    async def init_scheduler(
        self,
        episode_time_start: datetime | str | None = None,
        episode_time_end: datetime | str | None = None,
        use_index_trigger: bool = False,
        *args,
    ) -> None:
        """Initialize the scheduler by iterating the production schedule. Choose between 'DateTrigger' and
        'IntervallTrigger'. The 'DateTrigger' is used when your time in the production schedule is a date,
        otherwise 'IntervallTrigger' is used.

        :param episode_time_start: Start time of the scenario as datetime or string.
        :param episode_time_end: Endtime of the scenario as datetime or string.
        :param use_index_trigger: Set the scheduler trigger to an interval or date trigger.
        """
        assert self.production_schedule is not pd.DataFrame, "Production schedule is not initialized."

        self._init_times(
            episode_time_start,
            episode_time_end,
            use_index_trigger,
        )

        async with AsyncScheduler() as self.scheduler:
            for (
                _,
                row,
            ) in self.production_schedule.iterrows():
                if pd.isna(row["working_starttime"]):
                    raise ValueError("The production schedule has missing values in the 'working_starttime' column.")
                trigger: DateTrigger | IntervalTrigger = (
                    DateTrigger(row["working_starttime"])
                    if not use_index_trigger
                    else IntervalTrigger(row["working_starttime"])
                )

                try:
                    await self.scheduler.add_schedule(
                        func_or_task_id=self.execute_production_process,
                        trigger=trigger,
                        args=[
                            row["job"],
                            row["machine"],
                            row["endtime"],
                            row["capacity"],
                            *args,
                        ],
                    )
                except Exception as e:
                    raise Exception(
                        f"Could not add schedule from row {row} to the scheduler, got exception {e}."
                    ) from e

                self.handler.schedule_data_store = self.scheduler.data_store._schedules

    async def start_scheduler(self) -> None:
        """Start the initialized scheduler."""
        assert self.scheduler is not AsyncScheduler, "Scheduler is not initialized."

        async with self.scheduler:
            await self.scheduler.start_in_background()
            try:
                while datetime.now(get_localzone()) < self.episode_time_end:
                    # Create task for waiting for the event and the sleep interval
                    event_task = asyncio.create_task(self.handler.event.wait())
                    sleep_task = asyncio.create_task(asyncio.sleep(1))

                    # Wait for the first completed task
                    done, _ = await asyncio.wait(
                        [event_task, sleep_task],
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    # Cancel when the event has been triggered
                    if event_task in done:
                        logging.info("The scheduler will be stopped")
                        raise ExecutionProcessJobError
            finally:
                # Make sure that all tasks are canceled
                for task in [
                    event_task,
                    sleep_task,
                ]:
                    if not task.done():
                        task.cancel()
                    try:
                        await task  # Wait for the task to catch any cancel exceptions
                    except asyncio.CancelledError:
                        pass
                await self.stop_scheduler()

    async def stop_scheduler(self) -> None:
        """
        Stop the initialized scheduler.

        :param event_triggered: Boolean indicating if the event was triggered during the execution of the scheduler.
        """
        assert self.scheduler is not AsyncScheduler, "Scheduler is not initialized."

        await self.scheduler.stop()
