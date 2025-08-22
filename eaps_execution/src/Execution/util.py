from __future__ import annotations

import logging
import pathlib
import random
from datetime import datetime, timedelta, timezone

import pandas as pd


def setup_logger(log_file):
    """
    Generate a logger to log in log-file.

    :param log_file: File where logs are written in.
    """
    # Configure logger
    logger = logging.getLogger("ScheduleLogger")
    logger.setLevel(logging.DEBUG)

    # Create formatter
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # Create FileHandler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    # Add FileHandler to the logger, if not yet available
    if not logger.hasHandlers():
        logger.addHandler(file_handler)

    return logger


def generate_random_start_time(
    current_time: datetime,
    end_time: datetime,
) -> datetime:
    """
    Generate a random start time between the current time and end time.

    This function generates a random datetime between the current time
    and the end time. The datetime will be in the same timezone as the
    current time.

    :param current_time: The current time.
    :param end_time: The end time.

    :return datetime: A random datetime between the current time and end time.
    """
    # Generate a random number between 0 and the total number of seconds
    # between the current time and end time.
    delta = end_time - current_time
    total_seconds = delta.total_seconds()
    random_seconds = random.uniform(0, total_seconds)

    # Add the random number of seconds to the current time to get the
    # random start time.
    return current_time + timedelta(seconds=random_seconds)


def generate_random_working_duration() -> float:
    """
    Generate a random working duration between 10 and 40 seconds.

    This function generates a random float between 10 and 40 seconds.
    The duration is in seconds.

    :return int: A random float between 10 and 40 seconds.
    """
    return random.uniform(10, 40)


def generate_random_job() -> int:
    """
    Generate a random job ID.

    This function generates a random integer between 1000 and 9999,
    which can be used as a job ID. The function returns an integer.

    The valid range of job IDs is from 1000 to 9999, inclusive.

    :return int: A random integer between 1000 and 9999, inclusive.
    """
    return random.randint(1000, 9999)


def generate_random_delay_between_jobs() -> timedelta:
    """
    Generate a random delay between jobs.

    This function generates a random timedelta between 2 and 8 seconds,
    inclusive. The function returns a timedelta object.

    The valid range of delays is from 2 to 8 seconds, inclusive.

    :return timedelta: A random timedelta between 2 and 8 seconds, inclusive.
    """
    return timedelta(seconds=random.uniform(2, 8))


def generate_random_machine(
    machines: list[int],
) -> int:
    """
    Generate a random machine from the given list of machines.

    This function generates a random machine from the given list of machines. The function
    returns an integer representing the machine ID.

    :param machines (List[int]): The list of machines from which to generate a random machine.

    :return int: A random machine ID from the given list of machines.
    """
    return random.choice(machines)


def generate_random_bool() -> bool:
    """
    Generate a random boolean value.

    This function generates a random boolean value. The function takes no arguments and
    returns a boolean value. The possible boolean values are `True` and `False`.

    :return bool: A random boolean value. Possible values are `True` or `False`.
    """
    return random.choice([True, False])


def generate_job(
    start_time: datetime,
    working_starttime: datetime,
    end_time: datetime,
    machine: int,
) -> dict:
    """
    Generate a random job based on the given parameters.

    Generates a random job based on the given start time, working start time, end time, and
    machine ID. The function returns a dictionary containing the job details. The dictionary
    contains the following keys:

    - `starttime`: The start time of the job
    - `working_starttime`: The start time of the working phase of the job
    - `endtime`: The end time of the job
    - `machine`: The machine ID
    - `job`: A random job number
    - `capacity`: The job capacity (always set to 1)

    :param start_time (datetime): The start time of the job
    :param working_starttime (datetime): The start time of the working phase of the job
    :param end_time (datetime): The end time of the job
    :param machine (int): The machine ID

    :return dict: A dictionary containing the job details
    """
    # add job to the production schedule
    return {
        "starttime": start_time,
        "working_starttime": working_starttime,
        "endtime": end_time,
        "machine": machine,
        "job": generate_random_job(),  # random job number
        "capacity": 1,
    }


def generate_random_production_plan(
    schedule_length: int,
    machines: list[int] | None = None,
) -> pd.DataFrame:
    """
    Generates a random production schedule for the given machines.

    This function generates a production schedule for the given machines. The schedule length
    is defined by the `schedule_length` parameter. The function randomly chooses a machine
    from the given list of machines and schedules a job on that machine. The working duration
    of the job is randomly generated between 10 and 60 seconds. The start time of the job is
    randomly chosen within the next 10 seconds. The function checks if there is already a job
    running at the machine at the scheduled start time. If there is, the function skips this
    scheduled time and generates a new start time.

    The function returns a Pandas DataFrame containing the generated production schedule.

    :param schedule_length (int): The desired length of the production schedule.
    :param machines (list[int]): The list of machines to be used in the production schedule.

    :return pd.DataFrame: A Pandas DataFrame containing the generated production schedule.
    """
    # set machines to default
    if machines is None:
        machines = [971, 974]

    # current time
    current_time = datetime.now(timezone.utc) + timedelta(seconds=5)

    # end time of the schedule horizon
    end_time = current_time + timedelta(seconds=30)

    # list of the production schedule
    production_plan = []

    # generate production schedule
    while current_time < end_time:
        # random machine choice
        machine = generate_random_machine(machines)

        # random working duration
        working_duration = generate_random_working_duration()

        # random start time
        start_time = generate_random_start_time(current_time, end_time)

        # check whether a job is already running on the machine at this time
        if any(row["machine"] == machine and row["endtime"] > start_time for _, row in production_plan):
            # skip this point, as a job is already running on the machine
            continue

        # random working start time
        working_starttime = random.uniform(
            start_time,
            min(
                start_time + timedelta(seconds=10),
                end_time,
            ),
        )

        # end time based on the start time and the working duration
        end_time = start_time + timedelta(seconds=working_duration)

        # generate job
        production_plan.append(
            generate_job(
                start_time,
                working_starttime,
                end_time,
                machine,
            )
        )

        # delay time between jobs
        current_time = end_time + generate_random_delay_between_jobs()  # more than two seconds between two jobs

        # Check, if at least eight jobs are generated

    while len(production_plan) <= schedule_length:
        # generate random job
        machine = generate_random_machine(machines)
        start_time = generate_random_start_time(current_time, end_time)
        working_duration = generate_random_working_duration()
        end_time = start_time + timedelta(seconds=working_duration)
        flag = generate_random_bool()

        # random working start time during the next 10 seconds
        working_starttime = random.uniform(
            start_time,
            min(
                start_time + timedelta(seconds=10),
                end_time,
            ),
        )

        if flag is True:
            for machine in machines:
                production_plan.append(
                    generate_job(
                        start_time,
                        working_starttime,
                        end_time,
                        machine,
                    )
                )
        else:
            production_plan.append(
                generate_job(
                    start_time,
                    working_starttime,
                    end_time,
                    machine,
                )
            )

        current_time = end_time + generate_random_delay_between_jobs()  # more than two seconds between two jobs

    # convert list to DataFrame
    return pd.DataFrame(production_plan)


def transform_production_plan(
    folder_path: str | pathlib.Path,
    file_name: str | pathlib.Path,
    export_file_name: str | pathlib.Path,
    target_duration_minutes: int = 5,
) -> tuple[pd.DataFrame, float, datetime, datetime]:
    """
    Transforms the production schedule in the given CSV file by scaling it to a new
    duration(default 5 minutes). The function returns the adjusted data, the scaling factor,
    the new base start time, and the adjusted start time.

    The function loads the CSV file, renames the columns, converts the time columns to
    datetime objects, determines the earliest start time and the latest end time, calculates
    the total duration in seconds, calculates the scaling factor, and adjusts the times.

    The function then saves the adjusted data to a new CSV file, including all original
    columns, and returns the adjusted data, the scaling factor, the new base start time,
    and the adjusted start time.

    :param folder_path (str | pathlib.Path): The path to the folder containing the CSV file
    :param file_name (str | pathlib.Path): The name of the CSV file
    :param export_file_name (str | pathlib.Path): The name of the new CSV file containing the adjusted data
    :param target_duration_minutes (int | optional): The target duration in minutes, by default 5

    :return tuple[pd.DataFrame, float, datetime, datetime]: The adjusted data, the scaling factor, the new base start
    time, and the adjusted start time
    """
    # Convert to Path objects
    folder_path = pathlib.Path(folder_path)
    file_name = pathlib.Path(file_name)
    export_file_name = pathlib.Path(export_file_name)

    # Load the CSV file
    fullpath = folder_path / file_name
    export_path = folder_path / export_file_name

    data = pd.read_csv(fullpath)

    # Rename the columns
    data = data.rename(
        columns={
            "starttime": "origin_starttime",
            "working_starttime": "origin_working_starttime",
            "endtime": "origin_endtime",
        }
    )

    # Convert time columns to datetime objects
    data["origin_starttime"] = pd.to_datetime(data["origin_starttime"])
    data["origin_working_starttime"] = pd.to_datetime(data["origin_working_starttime"])
    data["origin_endtime"] = pd.to_datetime(data["origin_endtime"])

    # Determine the earliest start time and the latest end time
    min_start = data["origin_starttime"].min()
    max_end = data["origin_endtime"].max()

    # Calculate the total duration in seconds
    original_duration_seconds = (max_end - min_start).total_seconds()

    # Target duration in seconds (5 minutes)
    target_duration_seconds = target_duration_minutes * 60

    # Calculate the scaling factor
    scaling_factor = target_duration_seconds / original_duration_seconds

    # New base start time one minute into the future
    new_base_start_time = datetime.now(timezone.utc) + timedelta(minutes=0.5)

    # Adjust the times
    data["starttime"] = new_base_start_time + (data["origin_starttime"] - min_start) * scaling_factor
    data["working_starttime"] = new_base_start_time + (data["origin_working_starttime"] - min_start) * scaling_factor
    data["endtime"] = new_base_start_time + (data["origin_endtime"] - min_start) * scaling_factor

    # Save the adjusted data to a new CSV file, including all original columns
    data.to_csv(export_path, index=False)

    return (
        data,
        scaling_factor,
        min_start,
        new_base_start_time,
    )


def transform_time(
    time: datetime | str,
    base_start_time: datetime | str,
    min_start: datetime | str,
    scaling_factor: float,
) -> datetime:
    """
    Transforms a given time to its original time based on the base start time, min start time, and scaling factor.

    :param time (datetime | str): The time to be transformed, either a datetime object or a string in the
    format "%Y-%m-%d %H:%M"
    :param base_start_time (datetime | str): The base start time, either a datetime object or a string in the
    format "%Y-%m-%d %H:%M"
    :param min_start (datetime | str): The min start time, either a datetime object or a string in the
    format "%Y-%m-%d %H:%M"
    :param scaling_factor (float): The scaling factor calculated in `transform_production_plan`

    :return datetime: The original time, a datetime object
    """
    # Convert input parameters to datetime objects if necessary
    time = (
        time
        if isinstance(time, datetime)
        else datetime.strptime(time, "%Y-%m-%d %H:%M").replace(tzinfo=datetime.now().astimezone().tzinfo)
    )
    base_start_time = (
        base_start_time
        if isinstance(base_start_time, datetime)
        else datetime.strptime(base_start_time, "%Y-%m-%d %H:%M").replace(tzinfo=datetime.now().astimezone().tzinfo)
    )
    min_start = (
        min_start
        if isinstance(min_start, datetime)
        else datetime.strptime(min_start, "%Y-%m-%d %H:%M").replace(tzinfo=datetime.now().astimezone().tzinfo)
    )

    return (time - base_start_time) / scaling_factor + min_start
