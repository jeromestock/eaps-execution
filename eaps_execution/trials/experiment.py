from __future__ import annotations

import asyncio
import pathlib

import pandas as pd
from eta_utility.eta_x import ConfigOpt

from eaps_execution.src.Execution.prod_connection import ExecutionHandler
from eaps_execution.src.Execution.prod_env import (
    BaseExecutionEnv,
)
from eaps_execution.src.Execution.util import (
    transform_production_plan,
)


def execute(
    env: BaseExecutionEnv,
    production_plan: pd.DataFrame,
):
    env.set_production_schedule(production_plan)

    asyncio.run(env.experiment())


def main():
    series_name_base = "schedulemodels3001"
    file = pathlib.Path(__file__).parent / "production.json"
    config = ConfigOpt.from_config_file(file, pathlib.Path(__file__).parent)

    path_folder = pathlib.Path(__file__).parent

    # Initialize modules with current config file
    env = BaseExecutionEnv(
        machines={
            972: "GMD",
            971: "PBC",
            973: "GMB",
            974: "GIC",
        },
        connection_name="EAPS",
        connection_manager_file=path_folder / "environments/connection.toml",
        handler=ExecutionHandler(),
    )

    # Transform production schedule to minor target duration
    production_plan, _, _, _ = transform_production_plan(
        folder_path=config.path_results,
        file_name=f"{series_name_base}/{config.settings.environment['export_schedule_file']}",
        export_file_name=f"{series_name_base}/schedule_rescheduling.csv",
        target_duration_minutes=1,
    )

    # Start executing the energy aware production schedule
    execute(env, production_plan)


if __name__ == "__main__":
    main()
