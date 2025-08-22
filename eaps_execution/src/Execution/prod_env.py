from __future__ import annotations

import abc
import asyncio
from datetime import datetime
from typing import TYPE_CHECKING

from eta_nexus.connection_manager import ConnectionManager
from eta_nexus.util.type_annotations import Path
from eta_utility.eta_x.envs import StateConfig

from eaps_execution.src.Execution.prod_execution import (
    ExecuteProductionSchedule,
    ExecutionProcessJobError,
)
from eaps_execution.src.Execution.util import setup_logger

if TYPE_CHECKING:
    from typing import ClassVar

logger = setup_logger("production.log")


class BaseExecutionEnv(
    ExecuteProductionSchedule,
    abc.ABC,
):
    NODE_TEMPLATE: ClassVar[dict[str, str]] = {
        "working_state": "b{machine}WorkingState",
        "job": "s{machine}Job",
        "capacity": "i{machine}Capacity",
    }

    def __init__(
        self,
        machines: tuple[str, ...] | dict[str, str | int],
        connection_name: str,
        connection_manager_file: Path,
        prod_schedule_file: str | None = None,
        handler=None,
        max_errors: int = 10,
    ) -> None:
        super().__init__(
            prod_schedule_file=prod_schedule_file,
            handler=handler,
        )

        # Initialize ConnectionManager
        if not isinstance(connection_manager_file, Path):
            raise TypeError("connection_manager_file must be of type Path (str or PathLike)")

        self.connection_manager = ConnectionManager.from_config(
            connection_manager_file,
            max_error_count=max_errors,
        )

        self._prod_schedule_file = prod_schedule_file

        self._connection_manager_file = connection_manager_file

        self.machines: tuple[str] | dict[str | int, str] = machines

        machine_names = machines.values() if isinstance(machines, dict) else machines

        def generate_state_vars():
            for machine in machine_names:
                yield {
                    "name": self.NODE_TEMPLATE["working_state"].format(machine=machine),
                    "ext_id": f"{connection_name}.{self.NODE_TEMPLATE['working_state'].format(machine=machine)}",
                    "is_ext_input": True,
                }
                yield {
                    "name": self.NODE_TEMPLATE["job"].format(machine=machine),
                    "ext_id": f"{connection_name}.{self.NODE_TEMPLATE['job'].format(machine=machine)}",
                    "is_ext_input": True,
                }

        self.state_config = StateConfig.from_dict(generate_state_vars())

    @property
    def config_name(self) -> str:
        """Returns the name of the Connector Manager file."""
        return self._connection_manager_file

    @property
    def prod_schedule_file(self) -> str:
        """Returns the production schedule file."""
        return self._prod_schedule_file

    async def execute_production_process(
        self,
        job: str,
        machine: int,
        endtime: datetime | str,
        capacity: int,
        *args,
    ) -> None:
        """
        Executes the production process on the specified machine.

        :param job: ID of the job to be executed.
        :param machine: ID of the machine to execute the job on (972 for GMD, 971 for PBC, 973 for GMB, 974 for GIC).
        :param endtime: Endtime of the process step.
        :param capacity: Capacity of products during the execution of the job on the machine.
        :param args: Additional arguments that are not used in this function.
        """
        assert self.state_config is not None, "Set state_config before calling step function."

        assert self.connection_manager is not ConnectionManager, "Connector Manager is not initialized."

        await super().execute_production_process(job, machine, endtime, capacity, *args)

        # Preparation for writing the process step to the machine
        node_in = {}

        # Mapping from int to string
        if isinstance(self.machines, dict):
            try:
                node_in.update({str(self.NODE_TEMPLATE["working_state"].format(machine=self.machines[machine])): True})
                node_in.update({str(self.NODE_TEMPLATE["job"].format(machine=self.machines[machine])): job})
            except (KeyError, TypeError):
                logger.error("Error updating node_in state for machine %s and job %s", machine, job)
        else:
            try:
                node_in.update({str(self.NODE_TEMPLATE["working_state"].format(machine=machine)): True})
                node_in.update({str(self.NODE_TEMPLATE["job"].format(machine=machine)): job})
            except (KeyError, TypeError):
                logger.error("Error updating node_in state for machine %s and job %s", machine, job)

        self.connection_manager.write(node_in)

    async def experiment(self) -> None:
        """
        Starts the experiment by starting the scheduler.
        """
        # Sleep time before starting to initialize the scheduler.
        await asyncio.sleep(1)

        # Initialize the scheduler.
        await self.init_scheduler()

        # Sleep time before starting the execution and the subscription.
        await asyncio.sleep(1)

        try:
            # Sleep time before starting the scheduler.
            await asyncio.sleep(1)

            # Start the scheduler.
            scheduler_task = asyncio.create_task(self.start_scheduler())

            # Wait for either the scheduler to finish or an error in any task
            done, _ = await asyncio.wait(
                [scheduler_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # If the scheduler is done, cancel the scheduler
            if scheduler_task in done:
                if scheduler_task.exception():
                    raise scheduler_task.exception()
                logger.info("Scheduler task completed successfully.")
            # Wait until all tasks are completed or cancelled
            results = await asyncio.gather(
                scheduler_task,
                return_exceptions=True,
            )

            for result in results:
                if isinstance(result, Exception):
                    raise result

            self.connection_manager.close()
        except ExecutionProcessJobError as e:
            logger.error("The scheduler was stopped: %s", e)
        except Exception:
            pass
