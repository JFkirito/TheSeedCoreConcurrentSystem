# -*- coding: utf-8 -*-
"""
TheSeedCore ConcurrentSystem

Copyright (c) 2024 疾风Kirito <1453882193@qq.com>
All rights reserved.

Author: 疾风Kirito
Version: 1.0.0
Date: 2024-09-27

Description:
    TheSeedCore ConcurrentSystem is an advanced, high-performance framework designed for concurrent task execution in environments that demand high throughput, low latency, and efficient resource management.
    It provides a robust and flexible architecture to manage and execute tasks asynchronously, leveraging both multi-threading and multi-processing paradigms.
    The system is engineered to scale dynamically based on system load and task requirements, offering optimized CPU and GPU resource utilization.
    With built-in support for GPU acceleration, intelligent process and thread pool management, and sophisticated task scheduling mechanisms,
    ConcurrentSystem is ideal for applications requiring intensive computational tasks, distributed workloads, or real-time processing.

    This system emphasizes flexibility, allowing users to configure various operational parameters to adapt to different workloads and hardware environments.
    It is designed to be highly adaptable, providing seamless integration with PyTorch for GPU-accelerated tasks and offering platform-agnostic compatibility with Linux, Windows, and macOS.
    Through its advanced resource management strategies, such as automatic scaling and load balancing,
    the system ensures that system resources are utilized optimally, preventing bottlenecks and ensuring sustained high performance even under heavy load conditions.

Key Features:
    - **Dynamic Resource Management**: Automatically adjusts the number of processes and threads based on system load and task demands,
        ensuring efficient resource utilization and flexible scaling.

    - **Cross-platform Compatibility**: Fully supports Linux, Windows, and macOS, providing consistent behavior and interfaces across platforms.

    - **GPU Acceleration Support**: Integrated with PyTorch to leverage GPU acceleration for computationally intensive tasks, significantly improving performance.

    - **Advanced Task Scheduling**: Supports task prioritization, locking mechanisms, timeout controls, and retry strategies, ensuring critical tasks are handled promptly.

    - **Asynchronous Callback Mechanism**: Provides callback executors based on both Qt event loops and core asynchronous event loops,
        ensuring efficient task result propagation and processing.

    - **Flexible Configuration**: Offers a wide range of configuration options, allowing users to customize system behavior,
        including process and thread counts, priority settings, expansion and shrinkage policies, and more.

    - **Robustness and Fault Tolerance**: Built-in exception handling and resource cleanup mechanisms ensure system stability under high load and unexpected conditions.

Module Components:
    1. **Dependency Checker (_checkDependencies)**:
        Automatically detects critical dependencies in the system environment, such as PyTorch, PySide6, PyQt6, and PyQt5.
        It dynamically adjusts module functionality based on availability to ensure optimal compatibility.

    2. **System Monitors (_LinuxMonitor, _WindowsMonitor, _MacOSMonitor)**:
        Platform-specific resource monitors that provide precise system information, such as CPU core count, memory usage, and CPU utilization across different operating systems.

    3. **Configuration Manager (_ConfigManager)**:
        Handles the parsing and validation of user-provided configuration parameters, ensuring the system operates within reasonable resource limits.
        Provides default values and warnings to help users optimize system configuration.

    4. **Synchronization Manager (_SynchronizationManager)**:
        Manages shared resources and synchronization mechanisms in the concurrent system, including task result queues, process and thread status pools, and task locks,
        ensuring thread-safe task scheduling and result handling.

    5. **Task Object (_TaskObject)**:
        Encapsulates task execution logic and properties, including argument serialization, GPU acceleration support, retry mechanisms, and timeout settings,
        ensuring tasks can be safely executed in a multi-process or multi-threaded environment.

    6. **Process and Thread Objects (_ProcessObject, _ThreadObject)**:
        Represents individual working processes and threads responsible for fetching and executing tasks from task queues.
        Supports task prioritization, resource cleanup, and exception handling, ensuring stable and efficient task execution.

    7. **Load Balancer (_LoadBalancer)**:
        Monitors system resource usage and task load in real-time.
        Based on predefined expansion and shrinkage policies, it dynamically adjusts the scale of process and thread pools, optimizing system performance.

    8. **Task Schedulers (_ProcessTaskScheduler, _ThreadTaskScheduler)**:
        Distributes tasks to the appropriate processes or threads for execution.
        Utilizes optimized scheduling strategies based on task priority, current load, and resource availability, ensuring tasks are executed efficiently.

    9. **Callback Executors (_QtCallbackExecutor, _CoreCallbackExecutor)**:
        Manages the execution of callbacks upon task completion.
        Depending on the system environment,
        it selects the appropriate event loop mechanism (Qt or core asynchronous) to ensure that callbacks are executed in the correct thread context,
        supporting both synchronous and asynchronous callbacks.

    10. **Task Future (TaskFuture)**:
        Provides access to the results of asynchronous tasks, supporting blocking wait, timeout control, and result retrieval,
        allowing users to conveniently obtain the execution results after task submission.

    11. **Concurrent System Core (ConcurrentSystem)**:
        The core manager of the module, providing a unified task submission interface.
        It manages the process and thread pools, schedulers, and load balancer.
        Users can submit tasks and retrieve results easily through this class.

    12. **System Connector (ConnectConcurrentSystem)**:
        A function that initializes and configures the concurrent system.
        It allows users to create and customize a `ConcurrentSystem` instance by accepting various optional parameters,
        such as core process count, core thread count, maximum process and thread count, expansion policies, shrinkage policies, and task thresholds.
        This method enables users to flexibly set the behavior of the concurrent system based on actual needs.

Notes:
    - **Task Serialization**:
        Submitted task functions and arguments must be serializable, especially when using a multi-process environment, to ensure that tasks can be transmitted between processes.

    - **GPU Acceleration**:
        To use GPU acceleration features, ensure that PyTorch is installed and that a CUDA-enabled GPU is available.

    - **Resource Configuration**:
        It is recommended to adjust configuration parameters based on actual hardware resources (such as CPU core count and memory size) to avoid overloading system resources and reducing performance.

License:
    This software is licensed under the MIT License. For the full text, see below:

MIT License

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

from __future__ import annotations

__author__ = "疾风Kirito"
__version__ = "1.0.0"
__date__ = "2024-09-27"
__all__ = [
    "ConcurrentSystem",
    "TaskFuture",
    "ConnectConcurrentSystem",
    "PyTorchSupport",
    "AvailableCUDADevicesID",
    "PySide6Support",
    "PyQt6Support",
    "PyQt5Support",
    "QApplication"
]

import asyncio
import ctypes
import logging
import multiprocessing
import os
import pickle
import platform
import queue
import subprocess
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from ctypes import wintypes
from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, TYPE_CHECKING, Tuple, Union

if TYPE_CHECKING:
    pass

MainProcess = True if multiprocessing.current_process().name == 'MainProcess' else False
PyTorchSupport: bool = False
AvailableCUDADevicesID: list = []
PySide6Support: bool = False
PyQt6Support: bool = False
PyQt5Support: bool = False
QtMode = False


def _checkDependencies() -> None:
    """
    Checks and initializes the availability of required dependencies for PyTorch and various Qt frameworks.

    This function imports necessary libraries and sets global flags indicating support for PyTorch and the following Qt frameworks:
    - PySide6
    - PyQt6
    - PyQt5

    steps:
        1. Try to import the PyTorch library:
            - If successful, check if CUDA is available:
                - If CUDA is available, populate AvailableCUDADevicesID with the IDs of available CUDA devices and set PyTorchSupport to True.
                - If CUDA is not available, set AvailableCUDADevicesID to an empty list and PyTorchSupport to False.
            - If the import fails (ImportError), set AvailableCUDADevicesID to an empty list and PyTorchSupport to False.
        2. Try to import the PySide6 framework:
            - If successful, set PySide6Support to True.
            - If the import fails (ImportError), set PySide6Support to False.
        3. Try to import the PyQt6 framework:
            - If successful, set PyQt6Support to True.
            - If the import fails (ImportError), set PyQt6Support to False.
        4. Try to import the PyQt5 framework:
            - If successful, set PyQt5Support to True.
            - If the import fails (ImportError), set PyQt5Support to False.

    Notes:
        - This function is crucial for determining the capabilities of the current environment regarding GPU support and the availability of GUI frameworks.
        - The global variables indicate whether the respective libraries are available for use, allowing for conditional logic in the application.
    """

    global PyTorchSupport, AvailableCUDADevicesID, PySide6Support, PyQt6Support, PyQt5Support
    try:
        # noinspection PyUnresolvedReferences
        import torch

        if torch.cuda.is_available():
            AvailableCUDADevicesID = [cuda_device_id for cuda_device_id in range(torch.cuda.device_count())]
            PyTorchSupport = True
        else:
            AvailableCUDADevicesID = []
            PyTorchSupport = False
    except ImportError as _:
        AvailableCUDADevicesID = []
        PyTorchSupport = False

    try:
        import qasync
        from PySide6.QtWidgets import QApplication

        PySide6Support = True
    except ImportError as _:
        PySide6Support = False

    try:
        import qasync
        from PyQt6.QtWidgets import QApplication
        PyQt6Support = True
    except ImportError as _:
        PyQt6Support = False

    try:
        import qasync
        from PyQt5.QtWidgets import QApplication
        PyQt5Support = True
    except ImportError as _:
        PyQt5Support = False


_checkDependencies()

if PyTorchSupport:
    # noinspection PyUnresolvedReferences
    import torch
if PySide6Support:
    # noinspection PyUnresolvedReferences
    from PySide6.QtCore import Signal, QThread
    # noinspection PyUnresolvedReferences
    from PySide6.QtWidgets import QApplication

    QtMode = True
elif PyQt6Support:
    # noinspection PyUnresolvedReferences
    from PyQt6.QtCore import pyqtSignal, QThread
    # noinspection PyUnresolvedReferences
    from PyQt6.QtWidgets import QApplication

    QtMode = True
elif PyQt5Support:
    # noinspection PyUnresolvedReferences
    from PyQt5.QtCore import pyqtSignal, QThread
    # noinspection PyUnresolvedReferences
    from PyQt5.QtWidgets import QApplication

    QtMode = True
else:
    QtMode = False
    QThread = None
    QApplication = None

if QtMode:
    # noinspection PyUnresolvedReferences
    import qasync


    class _QtCallbackExecutor(QThread):
        """
        Handles the execution of callback functions in a Qt-based application.

        This class extends `QThread` and is designed to manage callbacks associated with completed tasks while integrating with the Qt event loop.

        Attributes:
            SynchronizationManager: Manages synchronization across task results.
            CloseEvent: Event to signal the executor to stop.
            ExecuteSignal: Signal to emit when executing a callback.

        Methods:
            startExecutor: Starts the callback executor by beginning the thread's execution.
            closeExecutor: Signals the executor to stop and waits for it to finish.
            run: Continuously checks for completed task results and emits their callbacks.
            callbackExecutor: Executes a given callback function with the result of the task, supporting both synchronous and asynchronous functions.

        Notes:
            - The executor is designed to work seamlessly with the Qt framework, allowing for responsive UI updates upon task completion.
            - Proper management of the callback execution ensures that results are handled efficiently without blocking the main thread.
        """

        if PySide6Support:
            ExecuteSignal = Signal(tuple)
        elif PyQt6Support or PyQt5Support:
            ExecuteSignal = pyqtSignal(tuple)

        def __init__(self, SM: _SynchronizationManager, parent=None):
            super().__init__(parent)
            self.SynchronizationManager = SM
            self.CloseEvent = threading.Event()
            self.ExecuteSignal.connect(self.callbackExecutor)
            # noinspection PyUnresolvedReferences
            self.setPriority(QThread.Priority.HighestPriority)

        def startExecutor(self):
            """
            Starts the executor by invoking the start method, which begins the execution of the associated thread.

            Notes:
                - This method is essential for initializing the executor's operation, allowing it to begin processing tasks and managing callbacks.
                - By calling the start method, the executor runs asynchronously, enabling non-blocking execution in the application.
            """

            self.start()

        def closeExecutor(self):
            """
            Signals the executor to close and waits for the termination of the associated thread.

            steps:
                1. Set the CloseEvent to indicate that the executor should terminate its operations.
                2. Call the wait method to block the current thread until the executor's thread has finished executing.

            Notes:
                - This method is crucial for ensuring a graceful shutdown of the executor, allowing it to complete any ongoing tasks and clean up resources.
                - The wait mechanism ensures that the calling thread will not proceed until the executor has fully terminated, maintaining proper synchronization.
            """

            self.CloseEvent.set()
            self.wait(2)

        def run(self):
            """
            Processes completed task results and executes associated callbacks.

            This method continuously checks for task results from the result storage queue,
            storing the results and executing any registered callbacks.

            steps:
                1. Enter a loop that continues until the CloseEvent is set:
                    - Attempt to retrieve callback data from the ResultStorageQueue without blocking:
                        - If successful, unpack the retrieved data into task_result and task_id.
                        - Store the task result in the global _FutureResult dictionary using the task_id.
                        - Check if the task_id exists in the global _CallbackObject dictionary:
                            - If it does, retrieve the associated callback object.
                            - Emit a signal to execute the callback with the task result and remove the callback from _CallbackObject.
                    - If a queue.Empty exception occurs, sleep briefly (0.001 seconds) before continuing to the next iteration.

            Notes:
                - This method is essential for managing the execution of callbacks related to completed tasks.
                - Proper handling of results ensures that tasks are processed efficiently and any associated actions are taken without delay.
                - The use of signals allows for seamless integration with the event-driven architecture of the application.
            """

            global _CallbackObject, _FutureResult
            while not self.CloseEvent.is_set():
                try:
                    callback_data = self.SynchronizationManager.ResultStorageQueue.get_nowait()
                    task_result, task_id = callback_data
                    _FutureResult[task_id] = task_result
                    if task_id in _CallbackObject:
                        callback_object = _CallbackObject[task_id]
                        self.ExecuteSignal.emit((callback_object, task_result))
                        del _CallbackObject[task_id]
                except queue.Empty:
                    time.sleep(0.001)

        @qasync.asyncSlot(tuple)
        async def callbackExecutor(self, callback_data: tuple):
            """
            Executes a callback function with the provided task result, handling both asynchronous and synchronous callbacks.

            :param callback_data: A tuple containing the callback function and the task result to be passed to it.

            steps:
                1. Unpack the callback_data tuple into callback_object and task_result.
                2. Check if the callback_object is an asynchronous coroutine function:
                    - If it is, await the execution of the callback function with the task result.
                3. If the callback_object is not a coroutine function, call it directly with the task result.

            Notes:
                - This method allows for flexible handling of callback functions, whether they are asynchronous or synchronous.
                - The use of `@qasync.asyncSlot` decorator ensures that the method is compatible with Qt's event loop, enabling smooth integration with asynchronous operations in a Qt application.
            """

            callback_object, task_result = callback_data
            if asyncio.iscoroutinefunction(callback_object):
                await callback_object(task_result)
                return
            callback_object(task_result)

_MainEventLoop: Optional[asyncio.AbstractEventLoop] = None


class _CoreCallbackExecutor:
    """
    Handles the execution of callback functions in response to completed tasks.

    This class manages a separate event loop for executing callbacks associated with tasks that have completed processing.

    Attributes:
        SynchronizationManager: Manages synchronization across task results.
        CloseEvent: Event to signal the executor to stop.
        MainEventLoop: The main event loop for handling asynchronous operations.

    Methods:
        startExecutor: Starts the callback executor by initiating the main run loop.
        closeExecutor: Signals the executor to stop running.
        run: Continuously checks for completed task results and executes their callbacks.
        callbackExecutor: Executes a given callback function with the result of the task.

    Notes:
        - The callback executor allows for asynchronous handling of task results, supporting both synchronous and asynchronous callback functions.
        - Proper management of the callback execution loop is essential for responsiveness in the system.
    """

    def __init__(self, SM: _SynchronizationManager):
        self.SynchronizationManager = SM
        self.CloseEvent = threading.Event()
        self.MainEventLoop: asyncio.AbstractEventLoop = asyncio.get_event_loop()

    def startExecutor(self):
        """
        Starts the executor by creating a new asynchronous task for the run method.

        steps:
            1. Use the MainEventLoop to create a new task that runs the run method.

        Notes:
            - This method is essential for initiating the executor's operation, allowing it to begin processing tasks and handling callbacks.
            - By creating a task in the event loop, the executor can operate asynchronously without blocking the main execution flow.
        """

        self.MainEventLoop.create_task(self.run())

    def closeExecutor(self):
        """
        Signals the executor to close by setting the CloseEvent.

        steps:
            1. Set the CloseEvent to indicate that the executor should terminate its operations.

        Notes:
            - This method is essential for gracefully shutting down the executor, allowing it to finish any ongoing tasks and clean up resources.
            - Proper signaling ensures that the executor can perform necessary cleanup operations before exiting.
        """

        self.CloseEvent.set()

    async def run(self):
        """
        Processes completed task results and executes associated callbacks.

        This asynchronous method continuously checks for task results from the result storage queue,
        storing the results and executing any registered callbacks.

        steps:
            1. Enter a loop that continues until the CloseEvent is set:
                - Attempt to retrieve callback data from the ResultStorageQueue without blocking:
                    - If successful, unpack the retrieved data into task_result and task_id.
                    - Store the task result in the global _FutureResult dictionary using the task_id.
                    - Check if the task_id exists in the global _CallbackObject dictionary:
                        - If it does, retrieve the associated callback object.
                        - Create a new asynchronous task to execute the callback with the result and remove the callback from _CallbackObject.
                - If a queue.Empty exception occurs, sleep briefly (0.001 seconds) before continuing to the next iteration.

        Notes:
            - This method is essential for managing the execution of callbacks related to completed tasks.
            - Proper handling of results ensures that tasks are processed efficiently and any associated actions are taken without delay.
            - The asynchronous design allows for responsiveness in the application while waiting for results.
        """

        global _CallbackObject, _FutureResult
        while not self.CloseEvent.is_set():
            try:
                callback_data = self.SynchronizationManager.ResultStorageQueue.get_nowait()
                task_result, task_id = callback_data
                _FutureResult[task_id] = task_result
                if task_id in _CallbackObject:
                    callback_object = _CallbackObject[task_id]
                    self.MainEventLoop.create_task(self.callbackExecutor((callback_object, task_result)))
                    del _CallbackObject[task_id]
            except queue.Empty:
                await asyncio.sleep(0.001)

    @staticmethod
    async def callbackExecutor(callback_data: tuple):
        """
        Executes a callback function with the provided task result.

        :param callback_data: A tuple containing the callback function and the task result to be passed to it.

        steps:
            1. Unpack the callback_data tuple into callback_object and task_result.
            2. Check if the callback_object is an asynchronous coroutine function:
                - If it is, await the execution of the callback function with the task result.
            3. If the callback_object is not a coroutine function, call it directly with the task result.

        Notes:
            - This method allows for flexible handling of callback functions, whether they are synchronous or asynchronous.
            - Proper execution of the callback ensures that the task result is processed as intended, regardless of the callback's nature.
        """

        callback_object, task_result = callback_data
        if asyncio.iscoroutinefunction(callback_object):
            await callback_object(task_result)
            return
        callback_object(task_result)


class _ColoredFormatter(logging.Formatter):
    """
    Custom logging formatter that adds color to log messages based on their severity level.

    This class extends `logging.Formatter` to enhance the readability of log outputs by using ANSI escape sequences for colored text.

    Attributes:
        COLORS: A dictionary mapping log levels to their corresponding ANSI color codes.
        RESET: ANSI code to reset text formatting.

    Methods:
        format: Overrides the default format method to apply color to log messages based on their severity level.

    Notes:
        - The formatter improves log visibility in terminal outputs, making it easier to distinguish between different log levels.
        - The colors used correspond to standard practices for log severity, with blue for DEBUG, green for INFO, yellow for WARNING, and red for ERROR.
    """

    COLORS = {
        logging.DEBUG: "\033[1;34m",
        logging.INFO: "\033[1;32m",
        logging.WARNING: "\033[1;33m",
        logging.ERROR: "\033[0;31m",
    }
    RESET = "\033[0m"

    def format(self, record):
        """
        Formats log messages with color based on their severity level.

        :param record: The log record containing information about the log message.

        :return: A formatted string representing the log message, with color applied based on the log level.

        steps:
            1. Call the superclass's format method to get the base message from the log record.
            2. Retrieve the appropriate color for the log level from the COLORS dictionary:
                - If the log level is not found in COLORS, use the default reset color.
            3. Return the formatted message wrapped in the corresponding color codes, followed by a reset code.

        Notes:
            - This method enhances log visibility by adding color coding to different levels of log messages (e.g., DEBUG, WARNING, ERROR).
            - It improves readability and helps quickly identify the severity of log messages in the console output.
        """

        message = super().format(record)
        color = self.COLORS.get(record.levelno, self.RESET)
        return f"{color}{message}{self.RESET}"


class _LinuxMonitor:
    """
    The LinuxMonitor class

    This class provides methods to retrieve various system metrics, such as CPU usage, memory statistics, and process-specific resource consumption.

    Attributes:
        _INSTANCE: Singleton instance of the _LinuxMonitor class.
        _INITIALIZED: Indicates whether the monitor has been initialized.
        _stat_path: Path to the /proc/stat file for CPU statistics.
        _cpu_info_path: Path to the /proc/cpuinfo file for CPU information.
        _mem_info_path: Path to the /proc/meminfo file for memory information.

    Methods:
        total_cpu_usage: Returns the total CPU usage as a percentage over a specified interval.
        physical_cpu_cores: Returns the number of physical CPU cores in the system.
        logical_cpu_cores: Returns the number of logical CPU cores in the system.
        total_physical_memory: Returns the total physical memory available in bytes.
        total_virtual_memory: Returns the total virtual memory available in bytes.
        process_cpu_usage: Returns the CPU usage of a specific process given its PID.
        process_memory_usage: Returns the memory usage of a specific process given its PID.

    Notes:
        - This class is essential for monitoring and diagnosing resource usage on Linux systems.
        - Proper initialization and resource management are important to ensure accurate monitoring.
    """

    _INSTANCE: _LinuxMonitor = None
    _INITIALIZED: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._INSTANCE is None:
            cls._INSTANCE = super().__new__(cls)
        return cls._INSTANCE

    def __init__(self):
        self._stat_path = "/proc/stat"
        self._cpu_info_path = "/proc/cpuinfo"
        self._mem_info_path = "/proc/meminfo"

    @classmethod
    def physicalCpuCores(cls) -> int:
        """
        Retrieves the number of physical CPU cores available on the Linux system.

        This class method reads the `/proc/cpuinfo` file to count the number of distinct physical CPU cores by checking lines that start with 'physical id'.

        :return: An integer representing the number of physical CPU cores.

        raises:
            RuntimeError: If the LinuxMonitor class instance is not initialized.
            FileNotFoundError: If the cpu_info_path does not point to a valid file.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the LinuxMonitor is not initialized.
            2. Open the CPU information file specified by _cpu_info_path:
                - This file contains various statistics about the CPUs on the system.
            3. Read the file and split its contents into lines:
                - Count the number of lines that start with 'physical id', which indicates a distinct physical CPU core.
            4. Return the total count of physical CPU cores.

        Notes:
            - This method is crucial for understanding the hardware capabilities of the system, especially for applications requiring multi-core processing.
            - Proper error handling should be in place to ensure the file can be read correctly and that the count is accurate.
        """

        if not cls._INSTANCE:
            raise RuntimeError("LinuxMonitor is not initialized.")
        with open(cls._INSTANCE._cpu_info_path, 'r') as f:
            cpuinfo = f.read().splitlines()
            return sum(1 for line in cpuinfo if line.startswith('physical id'))

    @classmethod
    def logicalCpuCores(cls) -> int:
        """
        Retrieves the number of logical CPU cores available on the Linux system.

        This class method utilizes the `os.cpu_count()` function to obtain the total number of logical processors.

        :return: An integer representing the number of logical CPU cores.

        raises:
            RuntimeError: If the LinuxMonitor class instance is not initialized.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the LinuxMonitor is not initialized.
            2. Call `os.cpu_count()` to retrieve the number of logical CPU cores.
            3. Return the number of logical CPU cores.

        Notes:
            - This method is important for determining the parallel processing capabilities of the system.
            - The `os.cpu_count()` function returns `None` if the number of CPUs cannot be determined; handle this case if necessary.
        """

        if not cls._INSTANCE:
            raise RuntimeError("LinuxMonitor is not initialized.")
        return os.cpu_count()

    @classmethod
    def totalPhysicalMemory(cls) -> int:
        """
        Retrieves the total physical memory (RAM) on a Linux system.

        This class method reads the `/proc/meminfo` file to extract the total amount of physical memory available on the system.

        :return: An integer representing the total physical memory in bytes.

        raises:
            RuntimeError: If the LinuxMonitor class instance is not initialized.
            FileNotFoundError: If the mem_info_path does not point to a valid file.
            ValueError: If the memory value cannot be converted to an integer.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the LinuxMonitor is not initialized.
            2. Open the memory information file specified by _mem_info_path:
                - This file contains various statistics about memory usage on the system.
            3. Read through each line of the file:
                - Look for the line that starts with 'MemTotal', which indicates the total physical memory available.
            4. Extract the physical memory value from the line, converting it from kilobytes (KB) to bytes by multiplying by 1024.
            5. Return the total physical memory in bytes.

        Notes:
            - This method is essential for understanding the total available RAM on a Linux system.
            - Proper error handling should be in place to ensure the file can be read correctly and that the memory value is valid.
        """

        if not cls._INSTANCE:
            raise RuntimeError("LinuxMonitor is not initialized.")
        with open(cls._INSTANCE._mem_info_path, 'r') as f:
            for line in f:
                if line.startswith('MemTotal'):
                    return int(line.split()[1]) * 1024

    @classmethod
    def totalVirtualMemory(cls) -> int:
        """
        Retrieves the total virtual memory (swap space) on a Linux system.

        This class method reads the `/proc/meminfo` file to extract the total amount of swap memory available on the system.

        :return: An integer representing the total virtual memory in bytes.

        raises:
            RuntimeError: If the LinuxMonitor class instance is not initialized.
            FileNotFoundError: If the mem_info_path does not point to a valid file.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the LinuxMonitor is not initialized.
            2. Open the memory information file specified by _mem_info_path:
                - This file contains various statistics about memory usage on the system.
            3. Read through each line of the file:
                - Look for the line that starts with 'SwapTotal', which indicates the total swap memory available.
            4. Extract the swap memory value from the line, converting it from kilobytes (KB) to bytes by multiplying by 1024.
            5. Return the total virtual memory (swap space) in bytes.

        Notes:
            - This method is critical for understanding the available swap space on a Linux system.
            - Proper error handling should be in place to ensure the file can be read correctly.
        """

        if not cls._INSTANCE:
            raise RuntimeError("LinuxMonitor is not initialized.")
        with open(cls._INSTANCE._mem_info_path, 'r') as f:
            swap_total = 0
            for line in f:
                if line.startswith('SwapTotal'):
                    swap_total = int(line.split()[1]) * 1024
            return swap_total

    @classmethod
    def totalCpuUsage(cls, interval: float = 1.0) -> float:
        """
        Calculates the total CPU usage percentage over a specified interval on a Linux system.

        This class method reads the CPU statistics from the `/proc/stat` file to compute the CPU usage based on the difference in CPU times before and after a sleep interval.

        :param interval: The time in seconds to wait before calculating the CPU usage. Defaults to 1.0 seconds.

        :return: A float representing the total CPU usage percentage.

        raises:
            RuntimeError: If the LinuxMonitor class instance is not initialized.
            FileNotFoundError: If the cpu_info_path does not point to a valid file.
            ZeroDivisionError: If total_time is zero, leading to a division error when calculating CPU usage.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the LinuxMonitor is not initialized.
            2. Open the CPU information file specified by _cpu_info_path and read the first line:
                - Split the line to extract the CPU times and convert them to integers.
            3. Sleep for the specified interval to allow for CPU usage change.
            4. Open the CPU information file again and read the first line to get the updated CPU times.
            5. Calculate the differences in idle time and total CPU time before and after the interval:
                - Calculate idle time as the difference between the initial and final idle CPU times.
                - Calculate total time as the difference between the total CPU times.
            6. Calculate CPU usage as a percentage based on the formula:
                - `cpu_usage = (1 - idle_time / total_time) * 100`
            7. Return the calculated CPU usage.

        Notes:
            - This method is important for monitoring the CPU load of the system, which can be useful for performance tuning and resource management.
            - Proper error handling should be implemented to deal with potential issues in reading the CPU information file.
        """

        if not cls._INSTANCE:
            raise RuntimeError("LinuxMonitor is not initialized.")
        with open(cls._INSTANCE._cpu_info_path, 'r') as f:
            cpu_times_start = f.readline().split()[1:8]
            cpu_times_start = [int(x) for x in cpu_times_start]
        time.sleep(interval)
        with open(cls._INSTANCE._cpu_info_path, 'r') as f:
            cpu_times_end = f.readline().split()[1:8]
            cpu_times_end = [int(x) for x in cpu_times_end]
        idle_start = cpu_times_start[3]
        idle_end = cpu_times_end[3]
        total_start = sum(cpu_times_start)
        total_end = sum(cpu_times_end)
        idle_time = idle_end - idle_start
        total_time = total_end - total_start
        cpu_usage = (1 - idle_time / total_time) * 100
        return cpu_usage

    @classmethod
    def processCpuUsage(cls, pid: int, interval: float = 1.0) -> int:
        """
        Calculates the CPU usage of a specified process on Linux over a given interval.

        This class method reads from the `/proc` filesystem to obtain the CPU time used by the specified process and calculates its CPU usage percentage.

        :param pid: An integer representing the process ID for which to calculate CPU usage.
        :param interval: A float representing the time interval (in seconds) over which to measure CPU usage (default is 1 second).

        :return: An integer representing the CPU usage percentage for the specified process.

        raises:
            RuntimeError: If the LinuxMonitor class instance is not initialized.
            FileNotFoundError: If the specified process does not exist or cannot be accessed.
            ValueError: If the CPU time cannot be converted to an integer.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the LinuxMonitor is not initialized.
            2. Define the path to the process's stat file in the `/proc` filesystem.
            3. Open the stat file and read the initial CPU times:
                - Extract user time (utime) and system time (stime) from the stat file.
                - Calculate the total CPU time at the start.
            4. Read the total CPU times from the system's stat file to obtain the initial total CPU time.
            5. Sleep for the specified interval to allow for time to pass.
            6. Open the stat file again to read the final CPU times:
                - Extract user time (utime) and system time (stime) from the stat file.
                - Calculate the total CPU time at the end.
            7. Read the total CPU times from the system's stat file again to obtain the final total CPU time.
            8. Calculate the difference in CPU times for the process and the system:
                - Determine the CPU usage percentage based on the process and system CPU time deltas.
            9. Return the calculated CPU usage percentage, ensuring it falls within the range of 0 to 100.

        Notes:
            - This method is crucial for monitoring the CPU consumption of individual processes on Linux.
            - Proper error handling ensures that the function behaves predictably in case of issues during file access.
        """

        if not cls._INSTANCE:
            raise RuntimeError("LinuxMonitor is not initialized.")

        proc_stat_path = f'/proc/{pid}/stat'
        with open(proc_stat_path, 'r') as f:
            proc_stat_start = f.readline().split()
        utime_start = int(proc_stat_start[13])
        stime_start = int(proc_stat_start[14])
        total_start = utime_start + stime_start
        with open(cls._INSTANCE._stat_path, 'r') as f:
            cpu_times_start = f.readline().split()[1:8]
        total_cpu_start = sum([int(x) for x in cpu_times_start])
        time.sleep(interval)
        with open(proc_stat_path, 'r') as f:
            proc_stat_end = f.readline().split()
        utime_end = int(proc_stat_end[13])
        stime_end = int(proc_stat_end[14])
        total_end = utime_end + stime_end
        with open(cls._INSTANCE._stat_path, 'r') as f:
            cpu_times_end = f.readline().split()[1:8]
        total_cpu_end = sum([int(x) for x in cpu_times_end])
        proc_time = total_end - total_start
        total_time = total_cpu_end - total_cpu_start
        num_cores = os.cpu_count()
        cpu_usage = (proc_time / total_time) * num_cores * 100
        return max(0, min(int(cpu_usage), 100))

    @classmethod
    def processMemoryUsage(cls, pid) -> int:
        """
        Retrieves the memory usage of a specified process on Linux.

        This class method reads the `/proc` filesystem to obtain the resident set size (RSS) of the specified process.

        :param pid: An integer representing the process ID for which to retrieve memory usage.

        :return: An integer representing the memory usage in bytes.

        raises:
            RuntimeError: If the LinuxMonitor class instance is not initialized.
            FileNotFoundError: If the specified process does not exist or cannot be accessed.
            ValueError: If the memory usage cannot be converted to an integer.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the LinuxMonitor is not initialized.
            2. Open the status file for the specified process in the `/proc` filesystem:
                - The file contains information about the process, including memory usage.
            3. Read through each line of the file:
                - Look for the line that starts with 'VmRSS', which indicates the resident set size of the process.
            4. Extract the memory value from the line, converting it from kilobytes (KB) to bytes by multiplying by 1024.
            5. Return the memory usage in bytes.

        Notes:
            - This method is critical for monitoring the memory consumption of individual processes on Linux.
            - Ensure proper error handling is in place for situations where the process does not exist or cannot be accessed.
        """

        if not cls._INSTANCE:
            raise RuntimeError("LinuxMonitor is not initialized.")
        with open(f'/proc/{pid}/status', 'r') as f:
            for line in f:
                if line.startswith('VmRSS'):
                    return int(line.split()[1]) * 1024


class _WindowsMonitor:
    """
    The WindowsMonitor class

    Provides a set of methods for monitoring CPU and memory usage on Windows systems. This class handles
    the interaction with Windows API to retrieve system performance metrics, including CPU and memory statistics.

    This class implements the singleton design pattern, ensuring that only one instance of _WindowsMonitor exists.
    It is responsible for initializing necessary handles and settings for performance monitoring.

    Attributes:
        _INSTANCE: Singleton instance of the _WindowsMonitor class.
        _INITIALIZED: Indicates whether the monitor has been initialized.
        _ULONG_PTR: Data type for handling pointer-sized values based on architecture.
        Pdh: Handle to the Performance Data Helper (PDH) library.
        Kernel32: Handle to the Kernel32 library for system calls.
        PdhFmtDouble: Format specifier for PDH double values.
        RelationProcessorCore: Specifies the processor relationship type for CPU information.
        QueryHandle: Handle for the PDH query.
        CpuCounterHandle: Handle for the CPU counter.

    Methods:
        __new__: Controls instantiation to enforce the singleton pattern.
        __init__: Initializes the Windows monitor and its components.
        total_cpu_usage: Retrieves the total CPU usage percentage over a specified interval.
        physical_cpu_cores: Retrieves the number of physical CPU cores in the system.
        logical_cpu_cores: Retrieves the number of logical CPU cores in the system.
        total_physical_memory: Retrieves the total physical memory of the system.
        total_virtual_memory: Retrieves the total virtual memory available in the system.
        process_cpu_usage: Calculates the CPU usage percentage of a specified process.
        process_memory_usage: Retrieves the memory usage of a specified process.

    Notes:
        - This class is essential for monitoring system performance on Windows.
        - Proper error handling is included to ensure that functions behave predictably in case of API failures.
    """

    _INSTANCE: _WindowsMonitor = None
    _INITIALIZED: bool = False
    _ULONG_PTR = ctypes.c_ulonglong if platform.architecture()[0] == '64bit' else ctypes.c_ulong

    class PdhFmtCounterValue(ctypes.Structure):
        """
        Represents the formatted counter value for a Performance Data Helper (PDH) counter.

        This class is a ctypes structure used to store the status and formatted value of a PDH counter.

        Fields:
            CStatus: A DWORD representing the status of the counter, indicating whether the value is valid or an error code.
            doubleValue: A double representing the formatted counter value, typically as a floating-point number.

        Notes:
            - This structure is primarily used in conjunction with PDH functions to retrieve and format performance data.
            - Proper interpretation of the CStatus field is crucial for understanding the validity of the returned value.
        """

        _fields_ = [('CStatus', wintypes.DWORD),
                    ('doubleValue', ctypes.c_double)]

    class ProcessorRelationship(ctypes.Structure):
        """
        Represents the relationship of processors in the system.

        This class is a ctypes structure that holds information about the relationship between
        logical processors and their respective groups.

        Fields:
            Flags: A byte indicating the characteristics of the processor (e.g., whether it is part of a core).
            EfficiencyClass: A byte indicating the efficiency class of the processor.
            Reserved: A byte array reserved for future use or alignment.
            GroupCount: A WORD indicating the number of groups this processor is part of.
            GroupMask: A pointer to an array of ULONG_PTR values representing the processor group mask.

        Notes:
            - This structure is used in conjunction with system calls to gather detailed information about the CPU architecture and logical processor relationships in a Windows environment.
            - Proper alignment and size specifications are essential for correct interoperability with the Windows API.
        """
        pass

    class SystemLogicalProcessorInformationEx(ctypes.Structure):
        """
        Represents the logical processor information in the system.

        This class is a ctypes structure that holds detailed information about logical processors
        and their relationships to physical processors.

        Fields:
            Relationship: A DWORD indicating the relationship of the logical processor (e.g., core, cache).
            Size: A DWORD representing the size of this structure.
            Processor: An instance of ProcessorRelationship that holds details about the processor's relationship.

        Notes:
            - This structure is used when retrieving logical processor information through system calls in a Windows environment.
            - It provides essential information for understanding the logical-to-physical processor mapping, which is useful for performance optimization and resource management.
        """
        pass

    class MemoryStatusEx(ctypes.Structure):
        """
        Represents the memory status of the system.

        This class is a ctypes structure that holds information about the current memory usage and availability in a Windows environment.

        Fields:
            dwLength: A DWORD representing the size of this structure.
            dwMemoryLoad: A DWORD indicating the current memory load as a percentage of total physical memory.
            ullTotalPhys: A ULONG64 representing the total amount of physical memory in bytes.
            ullAvailPhys: A ULONG64 representing the amount of physical memory currently available in bytes.
            ullTotalPageFile: A ULONG64 representing the total size of the page file in bytes.
            ullAvailPageFile: A ULONG64 representing the amount of page file space currently available in bytes.
            ullTotalVirtual: A ULONG64 representing the total size of the virtual memory in bytes.
            ullAvailVirtual: A ULONG64 representing the amount of virtual memory currently available in bytes.
            ullAvailExtendedVirtual: A ULONG64 representing the amount of extended virtual memory available in bytes.

        Notes:
            - This structure is used with the GlobalMemoryStatusEx function to retrieve information about the current state of memory usage in the system.
            - It is essential for monitoring memory resources and optimizing application performance.
        """

        _fields_ = [
            ('dwLength', wintypes.DWORD),
            ('dwMemoryLoad', wintypes.DWORD),
            ('ullTotalPhys', ctypes.c_ulonglong),
            ('ullAvailPhys', ctypes.c_ulonglong),
            ('ullTotalPageFile', ctypes.c_ulonglong),
            ('ullAvailPageFile', ctypes.c_ulonglong),
            ('ullTotalVirtual', ctypes.c_ulonglong),
            ('ullAvailVirtual', ctypes.c_ulonglong),
            ('ullAvailExtendedVirtual', ctypes.c_ulonglong),
        ]

    ProcessorRelationship._fields_ = [
        ("Flags", ctypes.c_byte),  # type: ignore
        ("EfficiencyClass", ctypes.c_byte),  # type: ignore
        ("Reserved", ctypes.c_byte * 20),  # type: ignore
        ("GroupCount", wintypes.WORD),  # type: ignore
        ("GroupMask", ctypes.POINTER(_ULONG_PTR))
    ]

    SystemLogicalProcessorInformationEx._fields_ = [
        ("Relationship", wintypes.DWORD),
        ("Size", wintypes.DWORD),
        ("Processor", ProcessorRelationship)
    ]

    def __new__(cls, *args, **kwargs):
        if cls._INSTANCE is None:
            cls._INSTANCE = super().__new__(cls)
        return cls._INSTANCE

    def __init__(self):
        self.Pdh = ctypes.WinDLL('pdh')
        self.Kernel32 = ctypes.WinDLL('kernel32')
        self.PdhFmtDouble = 0x00000200
        self.RelationProcessorCore = 0
        self.QueryHandle = ctypes.c_void_p()
        self.CpuCounterHandle = ctypes.c_void_p()
        self.Pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.QueryHandle))
        self.Pdh.PdhAddCounterW(self.QueryHandle, r'\Processor(_Total)\% Processor Time', 0, ctypes.byref(self.CpuCounterHandle))
        self._INITIALIZED = True

    @classmethod
    def physicalCpuCores(cls) -> int:
        """
        Retrieves the number of physical CPU cores in the system on Windows.

        This class method utilizes the Windows API to obtain the count of physical CPU cores.

        :return: An integer representing the number of physical CPU cores.

        raises:
            RuntimeError: If the WindowsMonitor class instance is not initialized.
            OSError: If the call to GetLogicalProcessorInformationEx fails.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the WindowsMonitor is not initialized.
            2. Initialize a buffer size variable to store the size required for the buffer.
            3. Call `GetLogicalProcessorInformationEx` with a null buffer to get the required buffer size.
            4. Create a buffer of the required size using ctypes.
            5. Call `GetLogicalProcessorInformationEx` again to fill the buffer with information about logical processors.
                - If the call fails, retrieve the last error code and raise an OSError with the corresponding error message.
            6. Initialize a counter for the number of physical cores and an offset for buffer traversal.
            7. Iterate through the buffer:
                - Cast the current offset in the buffer to the appropriate structure to read processor information.
                - If the relationship indicates a processor core, increment the physical core count by the number of groups in that core.
                - Update the offset by the size of the information structure.
            8. Return the total number of physical cores.

        Notes:
            - This method is essential for obtaining information about the CPU architecture of the system on Windows.
            - Proper error handling ensures that the function behaves predictably when the instance is not available or when API calls fail.
        """

        if not cls._INSTANCE:
            raise RuntimeError("WindowsMonitor is not initialized.")
        buffer_size = wintypes.DWORD(0)
        ctypes.windll.kernel32.GetLogicalProcessorInformationEx(cls._INSTANCE.RelationProcessorCore, None, ctypes.byref(buffer_size))
        buffer = (ctypes.c_byte * buffer_size.value)()
        result = ctypes.windll.kernel32.GetLogicalProcessorInformationEx(cls._INSTANCE.RelationProcessorCore, ctypes.byref(buffer), ctypes.byref(buffer_size))
        if not result:
            error_code = ctypes.windll.kernel32.GetLastError()
            raise OSError(f"GetLogicalProcessorInformationEx failed with error code {error_code}")
        num_physical_cores = 0
        offset = 0
        while offset < buffer_size.value:
            info = ctypes.cast(ctypes.byref(buffer, offset), ctypes.POINTER(cls._INSTANCE.SystemLogicalProcessorInformationEx)).contents
            if info.Relationship == cls._INSTANCE.RelationProcessorCore:
                num_physical_cores += info.Processor.GroupCount
            offset += info.Size
        return num_physical_cores

    @classmethod
    def logicalCpuCores(cls) -> int:
        """
        Retrieves the number of logical CPU cores in the system on Windows.

        This class method utilizes the Windows API to obtain the count of active logical CPU cores.

        :return: An integer representing the number of logical CPU cores.

        raises:
            RuntimeError: If the WindowsMonitor class instance is not initialized.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the WindowsMonitor is not initialized.
            2. Call the `GetActiveProcessorCount` function from the kernel32 DLL to retrieve the number of active logical CPU cores.
            3. Return the count of logical CPU cores.

        Notes:
            - This method is important for monitoring the CPU capabilities of the system on Windows.
            - Proper error handling ensures that the function behaves predictably when the instance is not available.
        """

        if not cls._INSTANCE:
            raise RuntimeError("WindowsMonitor is not initialized.")
        return ctypes.windll.kernel32.GetActiveProcessorCount(0)

    @classmethod
    def totalPhysicalMemory(cls) -> int:
        """
        Retrieves the total physical memory of the system on Windows.

        This class method uses the Windows API to get the total physical memory available in the system.

        :return: An integer representing the total physical memory in bytes.

        raises:
            RuntimeError: If the WindowsMonitor class instance is not initialized.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the WindowsMonitor is not initialized.
            2. Create an instance of the MemoryStatusEx structure to hold memory information.
            3. Set the dwLength attribute of the memory status structure to the size of the structure.
            4. Call the `GlobalMemoryStatusEx` function from the kernel32 DLL to fill the structure with memory data.
            5. Return the total physical memory from the ullTotalPhys attribute of the memory status structure.

        Notes:
            - This method is crucial for monitoring the physical memory status of the system on Windows.
            - Proper error handling ensures that the function behaves predictably when the instance is not available or when API calls fail.
        """

        if not cls._INSTANCE:
            raise RuntimeError("WindowsMonitor is not initialized.")
        memory_status = cls._INSTANCE.MemoryStatusEx()
        memory_status.dwLength = ctypes.sizeof(cls._INSTANCE.MemoryStatusEx)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
        return memory_status.ullTotalPhys

    @classmethod
    def totalVirtualMemory(cls) -> int:
        """
        Retrieves the total virtual memory of the system on Windows.

        This class method uses the Windows API to get the total virtual memory available in the system.

        :return: An integer representing the total virtual memory in bytes.

        raises:
            RuntimeError: If the WindowsMonitor class instance is not initialized.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the WindowsMonitor is not initialized.
            2. Create an instance of the MemoryStatusEx structure to hold memory information.
            3. Set the dwLength attribute of the memory status structure to the size of the structure.
            4. Call the `GlobalMemoryStatusEx` function from the kernel32 DLL to fill the structure with memory data.
            5. Return the total virtual memory from the ullTotalPageFile attribute of the memory status structure.

        Notes:
            - This method is essential for monitoring the virtual memory status of the system on Windows.
            - Proper error handling ensures that the function behaves predictably when the instance is not available or when API calls fail.
        """

        if not cls._INSTANCE:
            raise RuntimeError("WindowsMonitor is not initialized.")
        memory_status = cls._INSTANCE.MemoryStatusEx()
        memory_status.dwLength = ctypes.sizeof(cls._INSTANCE.MemoryStatusEx)
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory_status))
        return memory_status.ullTotalPageFile

    @classmethod
    def totalCpuUsage(cls, interval: float = 0.1) -> float:
        """
        Retrieves the total CPU usage of the system on Windows over a specified interval.

        This class method utilizes the Performance Data Helper (PDH) to measure CPU usage.

        :param interval: A float representing the time interval (in seconds) for which to measure CPU usage (default is 0.1 seconds).

        :return: A float representing the total CPU usage as a percentage.

        raises:
            RuntimeError: If the WindowsMonitor class instance is not initialized.
            OSError: If retrieving the formatted counter value fails.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the WindowsMonitor is not initialized.
            2. Call `PdhCollectQueryData` to collect the initial query data from the PDH.
            3. Sleep for the specified interval to allow for data collection.
            4. Call `PdhCollectQueryData` again to collect the updated query data.
            5. Retrieve the formatted counter value for the CPU usage:
                - Call `PdhGetFormattedCounterValue` to get the formatted CPU usage.
                - If the call fails, raise an OSError with the corresponding error code.
            6. Return the CPU usage as a double value.

        Notes:
            - This method is critical for monitoring the CPU usage of the system on Windows.
            - Proper error handling ensures that the function behaves predictably when the instance is not available or when PDH calls fail.
        """

        if not cls._INSTANCE:
            raise RuntimeError("WindowsMonitor is not initialized.")
        cls._INSTANCE.Pdh.PdhCollectQueryData(cls._INSTANCE.QueryHandle)
        time.sleep(interval)
        cls._INSTANCE.Pdh.PdhCollectQueryData(cls._INSTANCE.QueryHandle)
        counter_value = cls._INSTANCE.PdhFmtCounterValue()
        status = cls._INSTANCE.Pdh.PdhGetFormattedCounterValue(cls._INSTANCE.CpuCounterHandle, cls._INSTANCE.PdhFmtDouble, None, ctypes.byref(counter_value))
        if status != 0:
            raise OSError(f"PdhGetFormattedCounterValue failed with error code {status}")
        return counter_value.doubleValue

    @classmethod
    def processCpuUsage(cls, pid: int, interval: float = 1.0) -> int:
        """
        Calculates the CPU usage of a specified process on Windows over a given interval.

        This class method retrieves the CPU time used by the specified process and calculates its CPU usage percentage.

        :param pid: An integer representing the process ID for which to calculate CPU usage.
        :param interval: A float representing the time interval (in seconds) over which to measure CPU usage (default is 1.0 seconds).

        :return: An integer representing the CPU usage percentage for the specified process.

        raises:
            RuntimeError: If the WindowsMonitor class instance is not initialized.
            OSError: If there are issues with opening the process or retrieving process times.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the WindowsMonitor is not initialized.
            2. Define constants for process access rights.
            3. Attempt to open the process using OpenProcess with the specified access rights:
                - If unsuccessful, retrieve the last error code and raise an OSError.
            4. Create FILETIME structures for recording the kernel and user times at the start and end of the measurement.
            5. Call `GetProcessTimes` to retrieve the initial kernel and user times for the process:
                - If unsuccessful, retrieve the last error code and raise an OSError.
            6. Sleep for the specified interval to allow for time to pass.
            7. Call `GetProcessTimes` again to retrieve the final kernel and user times:
                - If unsuccessful, retrieve the last error code and raise an OSError.
            8. Calculate the total kernel and user time elapsed:
                - Convert the FILETIME structures to elapsed time in 100-nanosecond intervals.
            9. Calculate the total CPU usage percentage based on elapsed time and the number of active processor cores.
            10. Return the calculated CPU usage percentage, ensuring it falls within the range of 0 to 100.
            11. Ensure that the process handle is closed in the finally block to prevent resource leaks.

        Notes:
            - This method is crucial for monitoring the CPU consumption of individual processes on Windows.
            - Proper error handling ensures that the function behaves predictably in case of issues during API calls.
        """

        if not cls._INSTANCE:
            raise RuntimeError("WindowsMonitor is not initialized.")
        PROCESS_QUERY_INFORMATION = 0x0400
        PROCESS_VM_READ = 0x0010
        handle = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_INFORMATION | PROCESS_VM_READ, False, pid)
        if not handle:
            error_code = ctypes.windll.kernel32.GetLastError()
            raise OSError(f"Failed to open process {pid}. Error code: {error_code}")
        kernel_time_start = wintypes.FILETIME()
        user_time_start = wintypes.FILETIME()
        kernel_time_end = wintypes.FILETIME()
        user_time_end = wintypes.FILETIME()
        try:
            result = ctypes.windll.kernel32.GetProcessTimes(
                handle,
                ctypes.byref(wintypes.FILETIME()),
                ctypes.byref(wintypes.FILETIME()),
                ctypes.byref(kernel_time_start),
                ctypes.byref(user_time_start)
            )
            if not result:
                error_code = ctypes.windll.kernel32.GetLastError()
                raise OSError(f"Failed to get process times for start. Error code: {error_code}")
            time.sleep(interval)
            result = ctypes.windll.kernel32.GetProcessTimes(
                handle,
                ctypes.byref(wintypes.FILETIME()),
                ctypes.byref(wintypes.FILETIME()),
                ctypes.byref(kernel_time_end),
                ctypes.byref(user_time_end)
            )
            if not result:
                error_code = ctypes.windll.kernel32.GetLastError()
                raise OSError(f"Failed to get process times for end. Error code: {error_code}")
            kernel_time_elapsed = (kernel_time_end.dwLowDateTime - kernel_time_start.dwLowDateTime) + \
                                  (kernel_time_end.dwHighDateTime - kernel_time_start.dwHighDateTime) * (2 ** 32)
            user_time_elapsed = (user_time_end.dwLowDateTime - user_time_start.dwLowDateTime) + \
                                (user_time_end.dwHighDateTime - user_time_start.dwHighDateTime) * (2 ** 32)
            total_time_elapsed = kernel_time_elapsed + user_time_elapsed
            num_cores = ctypes.windll.kernel32.GetActiveProcessorCount(0)
            cpu_usage_percentage = (total_time_elapsed / (interval * 10 ** 7 * num_cores)) * 100
            return max(0, min(int(cpu_usage_percentage), 100))
        finally:
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)

    @classmethod
    def processMemoryUsage(cls, pid) -> int:
        """
        Retrieves the memory usage of a specified process on Windows.

        This class method uses the Windows API to obtain the amount of committed memory for the specified process.

        :param pid: An integer representing the process ID for which to retrieve memory usage.

        :return: An integer representing the committed memory usage in megabytes (MB).

        raises:
            RuntimeError: If the WindowsMonitor class instance is not initialized.
            OSError: If there is an issue opening the process or retrieving memory information.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the WindowsMonitor is not initialized.
            2. Open the process using OpenProcess with the necessary access rights (PROCESS_QUERY_INFORMATION and PROCESS_VM_READ):
                - If unsuccessful, raise an OSError indicating that the process could not be opened.
            3. Create a buffer to hold the memory information for the process.
            4. Call `GetProcessMemoryInfo` to fill the buffer with the process's memory information:
                - If the call fails, appropriate error handling should be added (currently missing).
            5. Extract the committed memory size from the buffer by accessing the appropriate bytes and casting them to a ULONG64 pointer.
            6. Close the process handle to release the resource.
            7. Return the committed memory size converted to megabytes (MB).

        Notes:
            - This method is essential for monitoring the memory consumption of individual processes on Windows.
            - Proper error handling is crucial to ensure that resources are managed effectively and errors are reported clearly.
        """

        if not cls._INSTANCE:
            raise RuntimeError("WindowsMonitor is not initialized.")

            # 打开进程
        process_handle = ctypes.windll.kernel32.OpenProcess(0x0400 | 0x0010, False, pid)
        if not process_handle:
            raise OSError(f"Unable to open process {pid}")
        process_memory_counters = ctypes.create_string_buffer(72)
        ctypes.windll.psapi.GetProcessMemoryInfo(process_handle, process_memory_counters, 72)
        commit_size = ctypes.cast(process_memory_counters[16:24], ctypes.POINTER(ctypes.c_ulonglong)).contents.value
        ctypes.windll.kernel32.CloseHandle(process_handle)
        return commit_size / (1024 * 1024)


class _MacOSMonitor:
    """
    The MacOSMonitor class

    A singleton class that provides methods to monitor system performance metrics specifically for macOS.
    It retrieves information about CPU usage, memory usage, and the number of CPU cores in a structured manner.

    This class implements a singleton design pattern, ensuring that only one instance of _MacOSMonitor exists.
    It offers various methods to access system statistics, enabling efficient performance monitoring in applications.

    Attributes:
        _INSTANCE: Singleton instance of the _MacOSMonitor class.
        _INITIALIZED: Indicates whether the monitor has been initialized.

    Methods:
        __new__: Controls instantiation to enforce the singleton pattern.
        __init__: Initializes the _MacOSMonitor instance if not already initialized.
        total_cpu_usage: Retrieves the total CPU usage of the system over a specified interval.
        physical_cpu_cores: Retrieves the number of physical CPU cores in the system.
        logical_cpu_cores: Retrieves the number of logical CPU cores in the system.
        total_physical_memory: Retrieves the total physical memory available on the system.
        total_virtual_memory: Retrieves the total virtual memory usage on the system.
        process_cpu_usage: Calculates the CPU usage of a specific process by its process ID (PID).
        process_memory_usage: Retrieves the memory usage of a specific process by its process ID (PID).

    Notes:
        - This class is essential for applications that require monitoring of system resources on macOS.
        - Proper error handling is implemented to ensure predictable behavior when the instance is not available.
    """

    _INSTANCE: _MacOSMonitor = None
    _INITIALIZED: bool = False

    def __new__(cls, *args, **kwargs):
        if cls._INSTANCE is None:
            cls._INSTANCE = super().__new__(cls)
        return cls._INSTANCE

    def __init__(self):
        ...

    @classmethod
    def physicalCpuCores(cls) -> int:
        """
        Retrieves the number of physical CPU cores in the system on macOS.

        This class method uses the sysctl command to obtain the count of physical CPU cores available.

        :return: An integer representing the number of physical CPU cores.

        raises:
            RuntimeError: If the MacOSMonitor class instance is not initialized.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the MacOSMonitor is not initialized.
            2. Execute the command `sysctl -n hw.physicalcpu` to retrieve the number of physical CPU cores.
            3. Decode the output of the command and convert it to an integer.
            4. Return the number of physical CPU cores.

        Notes:
            - This method is important for understanding the hardware capabilities of the system on macOS.
            - Proper error handling ensures that the function behaves predictably when the instance is not available.
        """

        if not cls._INSTANCE:
            raise RuntimeError("MacOSMonitor is not initialized.")
        return int(subprocess.check_output(['sysctl', '-n', 'hw.physicalcpu']).decode())

    @classmethod
    def logicalCpuCores(cls) -> int:
        """
        Retrieves the number of logical CPU cores in the system on macOS.

        This class method uses the sysctl command to obtain the count of logical CPU cores available.

        :return: An integer representing the number of logical CPU cores.

        raises:
            RuntimeError: If the MacOSMonitor class instance is not initialized.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the MacOSMonitor is not initialized.
            2. Execute the command `sysctl -n hw.logicalcpu` to retrieve the number of logical CPU cores.
            3. Decode the output of the command and convert it to an integer.
            4. Return the number of logical CPU cores.

        Notes:
            - This method is crucial for obtaining information about the CPU architecture of the system on macOS.
            - Proper error handling ensures that the function behaves predictably when the instance is not available.
        """

        if not cls._INSTANCE:
            raise RuntimeError("MacOSMonitor is not initialized.")
        return int(subprocess.check_output(['sysctl', '-n', 'hw.logicalcpu']).decode())

    @classmethod
    def totalPhysicalMemory(cls) -> int:
        """
        Retrieves the total physical memory of the system on macOS.

        This class method uses the sysctl command to get the total physical memory available.

        :return: An integer representing the total physical memory in bytes.

        raises:
            RuntimeError: If the MacOSMonitor class instance is not initialized.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the MacOSMonitor is not initialized.
            2. Execute the command `sysctl -n hw.memsize` to retrieve the total physical memory.
            3. Decode the output of the command and convert it to an integer.
            4. Return the total physical memory in bytes.

        Notes:
            - This method is essential for monitoring the physical memory status of the system on macOS.
            - Proper error handling ensures that the function behaves predictably when the instance is not available.
        """

        if not cls._INSTANCE:
            raise RuntimeError("MacOSMonitor is not initialized.")
        return int(subprocess.check_output(['sysctl', '-n', 'hw.memsize']).decode())

    @classmethod
    def totalVirtualMemory(cls) -> int:
        """
        Retrieves the total virtual memory usage on macOS.

        This class method calculates the total virtual memory by checking the number of swapped-out pages.

        :return: An integer representing the total virtual memory in bytes.

        raises:
            RuntimeError: If the MacOSMonitor class instance is not initialized.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the MacOSMonitor is not initialized.
            2. Execute the `vm_stat` command to retrieve virtual memory statistics.
            3. Decode the output of the command and split it into lines.
            4. Iterate through each line in the output:
                - Check for the line that contains 'Pages swapped out'.
                - Extract the number of swapped-out pages from that line.
                - Convert the extracted value to an integer, removing any decimal points.
            5. Multiply the number of swapped-out pages by 4096 (the page size in bytes) to get the total virtual memory usage.
            6. Return the calculated total virtual memory in bytes.

        Notes:
            - This method is crucial for monitoring the virtual memory status of the system on macOS.
            - Proper error handling ensures that the function behaves predictably when the instance is not available.
        """

        if not cls._INSTANCE:
            raise RuntimeError("MacOSMonitor is not initialized.")
        vm_stat_output = subprocess.check_output(['vm_stat']).decode()
        for line in vm_stat_output.splitlines():
            if 'Pages swapped out' in line:
                swap_out_pages = int(line.split()[3].replace('.', ''))
                return swap_out_pages * 4096

    @classmethod
    def totalCpuUsage(cls, interval: float = 1.0) -> float:
        """
        Retrieves the total CPU usage of the system on macOS over a specified interval.

        This class method executes the `top` command to obtain the current CPU usage percentage.

        :param interval: A float representing the time interval (in seconds) for which to measure CPU usage (default is 1.0 seconds).

        :return: A float representing the total CPU usage as a percentage.

        raises:
            RuntimeError: If the MacOSMonitor class instance is not initialized.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the MacOSMonitor is not initialized.
            2. Execute the `top` command with the specified interval to retrieve the CPU usage statistics.
            3. Decode the output of the command and split it into lines.
            4. Iterate through each line in the output:
                - Look for the line that contains 'CPU usage'.
                - Extract the CPU usage percentage from that line, converting it to a float.
            5. Return the CPU usage percentage.

        Notes:
            - This method is essential for monitoring the overall CPU usage of the system on macOS.
            - Proper error handling ensures that the function behaves predictably when the instance is not available.
        """

        if not cls._INSTANCE:
            raise RuntimeError("MacOSMonitor is not initialized.")
        top_output = subprocess.check_output(['top', '-l', f"{str(int(interval))}"]).decode()
        for line in top_output.splitlines():
            if 'CPU usage' in line:
                cpu_usage = float(line.split()[2].replace('%', ''))
                return cpu_usage

    @classmethod
    def processCpuUsage(cls, pid, interval=1) -> int:
        """
        Calculates the CPU usage of a process specified by its process ID (PID).

        This class method retrieves the CPU usage percentage of a given process on macOS over a specified interval.

        :param pid: The process ID of the target process whose CPU usage is to be retrieved.
        :param interval: An optional integer specifying the time interval (in seconds) over which to measure CPU usage (default is 1 second).

        :return: An integer representing the CPU usage percentage, clamped between 0 and 100.

        raises:
            RuntimeError: If the MacOSMonitor class instance is not initialized.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the MacOSMonitor is not initialized.
            2. Retrieve the number of logical CPU cores using the `sysctl` command.
            3. Use the `ps` command to retrieve the initial CPU usage percentage for the specified PID:
                - Execute the command `ps -p <pid> -o %cpu`, which outputs the CPU usage.
            4. Sleep for the specified interval to allow for CPU usage measurement.
            5. Retrieve the CPU usage percentage again for the same PID after the interval:
                - Execute the same `ps` command as before.
            6. Calculate the CPU usage over the interval:
                - Compute the difference between the end and start CPU usage, divide by the interval and the number of cores.
            7. Clamp the result between 0 and 100 to ensure valid CPU usage percentage.
            8. Return the CPU usage percentage as an integer.

        Notes:
            - This method is essential for monitoring the CPU consumption of processes on macOS.
            - Proper error handling ensures that the function behaves predictably when the instance is not available.
        """

        if not cls._INSTANCE:
            raise RuntimeError("MacOSMonitor is not initialized.")
        num_cores = int(subprocess.check_output(['sysctl', '-n', 'hw.logicalcpu']).decode().strip())
        ps_output = subprocess.check_output(['ps', '-p', str(pid), '-o', '%cpu']).decode()
        cpu_usage_start = float(ps_output.splitlines()[1].strip())
        time.sleep(interval)
        ps_output = subprocess.check_output(['ps', '-p', str(pid), '-o', '%cpu']).decode()
        cpu_usage_end = float(ps_output.splitlines()[1].strip())
        cpu_usage = (cpu_usage_end - cpu_usage_start) / interval / num_cores
        return max(0, min(int(cpu_usage), 100))

    @classmethod
    def processMemoryUsage(cls, pid) -> int:
        """
        Retrieves the memory usage of a process specified by its process ID (PID).

        This class method checks the RSS (Resident Set Size) memory usage of a given process on macOS.

        :param pid: The process ID of the target process whose memory usage is to be retrieved.

        :return: An integer representing the memory usage in bytes.

        raises:
            RuntimeError: If the MacOSMonitor class instance is not initialized.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the MacOSMonitor is not initialized.
            2. Use the `ps` command to retrieve the RSS memory usage for the specified PID:
                - Execute the command `ps -p <pid> -o rss`, which outputs the memory usage in kilobytes.
            3. Decode the command output and extract the memory usage value:
                - Convert the value from kilobytes to bytes by multiplying by 1024.
            4. Return the memory usage in bytes.

        Notes:
            - This method is crucial for monitoring the memory consumption of processes on macOS.
            - Proper error handling ensures that the function behaves predictably when the instance is not available.
        """

        if not cls._INSTANCE:
            raise RuntimeError("MacOSMonitor is not initialized.")
        ps_output = subprocess.check_output(['ps', '-p', str(pid), '-o', 'rss']).decode()
        memory_usage = int(ps_output.splitlines()[1].strip())
        return memory_usage * 1024


if MainProcess:
    _CallbackObject: Dict[str, callable] = {}
    _FutureResult: Dict[str, Any] = {}
    _CoreProcessPool: Dict[str, _ProcessObject] = {}
    _ExpandProcessPool: Dict[str, _ProcessObject] = {}
    _ExpandProcessSurvivalTime: Dict[str, float] = {}
    _CoreThreadPool: Dict[str, _ThreadObject] = {}
    _ExpandThreadPool: Dict[str, _ThreadObject] = {}
    _ExpandThreadSurvivalTime: Dict[str, float] = {}
    _system = platform.system()
    if _system == 'Linux':
        _Monitor = _LinuxMonitor()
    elif _system == 'Windows':
        _Monitor = _WindowsMonitor()
    elif _system == "Darwin":
        _Monitor = _MacOSMonitor()
    else:
        raise NotImplementedError(f"Monitor for {_system} is not implemented.")


@dataclass
class _ConcurrentSystemConfig:
    """
    Configuration settings for the concurrent system.

    This class serves as a data structure for holding various configuration parameters related to process and thread management.

    Attributes:
        CoreProcessCount: Number of core processes to be used (optional).
        CoreThreadCount: Number of core threads to be used (optional).
        MaximumProcessCount: Maximum allowable number of processes (optional).
        MaximumThreadCount: Maximum allowable number of threads (optional).
        IdleCleanupThreshold: Threshold for triggering idle cleanup (optional).
        ProcessPriority: Priority level for processes (default is "NORMAL").
        TaskThreshold: Threshold for task management (optional).
        GlobalTaskThreshold: Global threshold for task management (optional).
        ExpandPolicy: Policy for resource expansion (optional).
        ShrinkagePolicy: Policy for resource shrinkage (optional).
        ShrinkagePolicyTimeout: Timeout duration for shrinkage policy operations (optional).

    Notes:
        - The configuration class provides a clear and structured way to manage settings that influence the behavior of the concurrent system.
        - Using `dataclass` simplifies the initialization and representation of configuration objects.
    """

    CoreProcessCount: Optional[int] = None
    CoreThreadCount: Optional[int] = None
    MaximumProcessCount: Optional[int] = None
    MaximumThreadCount: Optional[int] = None
    IdleCleanupThreshold: Optional[int] = None
    ProcessPriority: Literal["IDLE", "BELOW_NORMAL", "NORMAL", "ABOVE_NORMAL", "HIGH", "REALTIME"] = "NORMAL"
    TaskThreshold: Optional[int] = None
    GlobalTaskThreshold: Optional[int] = None
    ExpandPolicy: Optional[Literal["NoExpand", "AutoExpand", "BeforehandExpand"]] = None
    ShrinkagePolicy: Optional[Literal["NoShrink", "AutoShrink", "TimeoutShrink"]] = None
    ShrinkagePolicyTimeout: Optional[int] = None


class _ConfigManager:
    """
    Manages the configuration settings for the concurrent system.

    This class initializes and validates various configuration parameters, ensuring they are within acceptable ranges and logging any issues or defaults applied.

    Attributes:
        PhysicalCores: Number of physical CPU cores available.
        DebugMode: Flag indicating whether debug mode is active.
        Logger: Logger for the configuration manager.
        CoreProcessCount: Number of core processes configured.
        CoreThreadCount: Number of core threads configured.
        MaximumProcessCount: Maximum number of processes allowed.
        MaximumThreadCount: Maximum number of threads allowed.
        IdleCleanupThreshold: Threshold for idle cleanup operations.
        TaskThreshold: Threshold for task management.
        GlobalTaskThreshold: Global threshold for task management across the system.
        ProcessPriority: Priority level for process execution.
        ExpandPolicy: Policy for expanding resources.
        ShrinkagePolicy: Policy for shrinking resources.
        ShrinkagePolicyTimeout: Timeout duration for shrinkage policy operations.

    Methods:
        _setLogger: Configures logging for the configuration manager.
        _validateCoreProcessCount: Validates and returns the configured core process count.
        _validateCoreThreadCount: Validates and returns the configured core thread count.
        _validateMaximumProcessCount: Validates and returns the configured maximum process count.
        _validateMaximumThreadCount: Validates and returns the configured maximum thread count.
        _validateIdleCleanupThreshold: Validates and returns the configured idle cleanup threshold.
        _validateTaskThreshold: Validates and returns the configured task threshold.
        _validateGlobalTaskThreshold: Validates and returns the configured global task threshold.
        _validateProcessPriority: Validates and returns the configured process priority.
        _validateExpandPolicy: Validates and returns the configured expand policy.
        _validateShrinkagePolicy: Validates and returns the configured shrinkage policy.
        _validateShrinkagePolicyTimeout: Validates and returns the configured shrinkage policy timeout.
        calculateTaskThreshold: Calculates an appropriate task threshold based on system resources.

    Notes:
        - The configuration manager is essential for ensuring that the concurrent system operates efficiently and within the resource limits of the hardware.
        - Proper validation of settings helps to avoid runtime issues and optimizes performance.
    """

    def __init__(self, SharedObjectManager: multiprocessing.Manager, Config: _ConcurrentSystemConfig, DebugMode: bool = False):
        self.PhysicalCores = _Monitor.physicalCpuCores()
        self.DebugMode = DebugMode
        self.Logger = self._setLogger()
        self.CoreProcessCount = SharedObjectManager.Value("i", self._validateCoreProcessCount(Config.CoreProcessCount))
        self.CoreThreadCount = SharedObjectManager.Value("i", self._validateCoreThreadCount(Config.CoreThreadCount))
        self.MaximumProcessCount = SharedObjectManager.Value("i", self._validateMaximumProcessCount(Config.MaximumProcessCount))
        self.MaximumThreadCount = SharedObjectManager.Value("i", self._validateMaximumThreadCount(Config.MaximumThreadCount))
        self.IdleCleanupThreshold = SharedObjectManager.Value("i", self._validateIdleCleanupThreshold(Config.IdleCleanupThreshold))
        self.TaskThreshold = SharedObjectManager.Value("i", self._validateTaskThreshold(Config.TaskThreshold))
        self.GlobalTaskThreshold = SharedObjectManager.Value("i", self._validateGlobalTaskThreshold(Config.GlobalTaskThreshold))
        self.ProcessPriority = SharedObjectManager.Value("c", self._validateProcessPriority(Config.ProcessPriority))
        self.ExpandPolicy = SharedObjectManager.Value("c", self._validateExpandPolicy(Config.ExpandPolicy))
        self.ShrinkagePolicy = SharedObjectManager.Value("c", self._validateShrinkagePolicy(Config.ShrinkagePolicy))
        self.ShrinkagePolicyTimeout = SharedObjectManager.Value("i", self._validateShrinkagePolicyTimeout(Config.ShrinkagePolicyTimeout))

    def _setLogger(self):
        """
        Sets up and configures a logger for the ConcurrentSystem.

        :return: The configured logger instance.

        steps:
            1. Create a logger instance with the name '[ConcurrentSystem]'.
            2. Set the logging level of the logger to DEBUG.
            3. Create a console handler for outputting log messages to the console.
            4. Determine the log level for the console handler:
                - If DebugMode is enabled, set the log level to DEBUG.
                - If DebugMode is disabled, set the log level to the maximum of DEBUG and WARNING.
            5. Create a formatter to define the format of the log messages, including timestamp, logger name, level, and message.
            6. Set the formatter for the console handler.
            7. Add the console handler to the logger.

        Notes:
            - This method ensures that the logging for the ConcurrentSystem is well-structured and easily readable.
            - Configuring the logger based on the DebugMode allows for flexible logging verbosity, aiding in debugging and monitoring.
        """

        logger = logging.getLogger('[ConcurrentSystem]')
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        if self.DebugMode:
            console_handler.setLevel(logging.DEBUG)
        else:
            console_handler.setLevel(max(logging.DEBUG, logging.WARNING))

        formatter = _ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        return logger

    def _validateCoreProcessCount(self, core_process_count: Optional[int]) -> int:
        """
        Validates the core process count and returns a valid integer.

        :param core_process_count: An optional integer representing the desired core process count.

        :return: The validated core process count, which may be the provided value or a default.

        steps:
            1. Calculate the default value for the core process count as half of the number of physical CPU cores (default_value).
            2. Check if the provided core_process_count is None:
                - If it is, log a warning indicating that the default value will be used and return the default value.
            3. Check if the provided core_process_count is not an integer:
                - If it is of an invalid type, log a warning message about the invalid type and return the default value.
            4. Check if the provided core_process_count is out of valid range:
                - If it is less than 0 or greater than twice the number of physical CPU cores, log a warning and return the default value.
            5. If the provided core_process_count is set to 0:
                - Log a warning indicating that the process pool will be unavailable.
            6. If the provided core_process_count is valid, return it as is.

        Notes:
            - This method is essential for ensuring that the core process count configuration is appropriate based on the system's capabilities.
            - Proper logging provides visibility into potential misconfigurations and ensures that defaults are clearly communicated.
        """

        default_value = self.PhysicalCores // 2
        if core_process_count is None:
            self.Logger.warning(f"Core process count not set, using default value {default_value}.")
            return default_value

        if not isinstance(core_process_count, int):
            self.Logger.warning(f"Invalid type for core process count '{core_process_count}'. Must be an integer; using default value {default_value}.")
            return default_value

        if core_process_count < 0 or core_process_count > self.PhysicalCores * 2:
            self.Logger.warning(f"Core process count {core_process_count} is out of valid range (0 to {self.PhysicalCores * 2}); using default value {default_value}.")
            return default_value

        if core_process_count == 0:
            self.Logger.warning(f"Core process count set to 0, process pool will be unavailable")

        return core_process_count

    def _validateCoreThreadCount(self, core_thread_count: Optional[int]) -> int:
        """
        Validates the core thread count and returns a valid integer.

        :param core_thread_count: An optional integer representing the desired core thread count.

        :return: The validated core thread count, which may be the provided value or a default.

        steps:
            1. Calculate the default value for the core thread count:
                - If CoreProcessCount is not zero, set the default to twice the CoreProcessCount.
                - If CoreProcessCount is zero, set the default to one-fourth of the PhysicalCores.
            2. Check if the provided core_thread_count is None:
                - If it is, log a warning indicating that the default value will be used and return the default value.
            3. Check if the provided core_thread_count is not an integer:
                - If it is of an invalid type, log a warning message about the invalid type and return the default value.
            4. Compare the provided core_thread_count with the following conditions:
                - If it is less than the default value, log a warning and set adjusted_count to the default value.
                - If it exceeds twice the number of physical CPU cores, log a warning and set adjusted_count to the default value.
                - If it is an odd number, log a warning about adjusting it to the nearest lower even number and set adjusted_count accordingly.
                - If it meets all criteria, set adjusted_count to the provided core_thread_count as is.
            5. Return the adjusted_count.

        Notes:
            - This method is important for ensuring that the core thread count configuration is appropriate based on the system's capabilities and avoids misconfigurations.
            - Proper logging provides visibility into potential issues and ensures that defaults are clearly communicated.
        """

        default_value = self.PhysicalCores // 4
        if core_thread_count is None:
            self.Logger.warning(f"Core thread count not set, using default value {default_value}.")
            return default_value

        if not isinstance(core_thread_count, int):
            self.Logger.warning(f"Invalid type for core thread count '{core_thread_count}'. Must be an integer; using default value {default_value}.")
            return default_value

        if core_thread_count < 0:
            self.Logger.warning(f"Core thread count {core_thread_count} is less than 0. Using default value {default_value}.")
            adjusted_count = default_value
        elif core_thread_count > int(self.PhysicalCores * 2):
            self.Logger.warning(f"Core thread count {core_thread_count} exceeds the maximum allowed {int(self.PhysicalCores * 2)}. Using default value {default_value}.")
            adjusted_count = default_value
        elif core_thread_count % 2 != 0:
            self.Logger.warning(f"Core thread count {core_thread_count} is not an even number. Adjusting to {core_thread_count - 1}.")
            adjusted_count = core_thread_count - 1
        else:
            adjusted_count = core_thread_count
        return adjusted_count

    def _validateMaximumProcessCount(self, maximum_process_count: Optional[int]) -> int:
        """
        Validates the maximum process count and returns a valid integer.

        :param maximum_process_count: An optional integer representing the desired maximum process count.

        :return: The validated maximum process count, which may be the provided value or a default.

        steps:
            1. Set the default value for the maximum process count to the number of physical CPU cores (default_value).
            2. Check if the provided maximum_process_count is None:
                - If it is, log a warning indicating that the default value will be used and return the default value.
            3. Check if the provided maximum_process_count is not an integer:
                - If it is of an invalid type, log a warning message about the invalid type and return the default value.
            4. Check if the provided maximum_process_count exceeds the number of physical CPU cores:
                - If it does, log a warning and return the default value.
            5. Check if the provided maximum_process_count is less than the current CoreProcessCount:
                - If it is, log a warning and return the default value.
            6. If the provided maximum_process_count is valid, return it as is.

        Notes:
            - This method ensures that the maximum process count configuration is appropriate based on the system's physical capabilities.
            - Proper logging provides visibility into potential misconfigurations and ensures that defaults are clearly communicated.
        """

        default_value = self.PhysicalCores
        if maximum_process_count is None:
            self.Logger.warning(f"Maximum process count not set, using default value: {default_value}.")
            return default_value

        if not isinstance(maximum_process_count, int):
            self.Logger.warning(f"Invalid type for maximum process count '{maximum_process_count}'. Must be an integer; using default value: {default_value}.")
            return default_value

        if maximum_process_count > self.PhysicalCores:
            self.Logger.warning(f"Maximum process count {maximum_process_count} exceeds the number of physical CPU cores ({self.PhysicalCores}). Using default value: {default_value}.")
            return default_value

        if maximum_process_count < self.CoreProcessCount.value:
            self.Logger.warning(f"Maximum process count {maximum_process_count} is less than the core process count ({self.CoreProcessCount}). Using default value: {default_value}.")
            return default_value

        return maximum_process_count

    def _validateMaximumThreadCount(self, maximum_thread_count: Optional[int]) -> int:
        """
        Validates the maximum thread count and returns a valid integer.

        :param maximum_thread_count: An optional integer representing the desired maximum thread count.

        :return: The validated maximum thread count, which may be the provided value or a default.

        steps:
            1. Calculate the default value for the maximum thread count as twice the number of physical cores (default_value).
            2. Check if the provided maximum_thread_count is None:
                - If it is, log a warning indicating that the default value will be used and return the default value.
            3. Check if the provided maximum_thread_count is not an integer:
                - If it is of an invalid type, log a warning message about the invalid type and return the default value.
            4. Compare the provided maximum_thread_count with the following conditions:
                - If it is less than the current CoreThreadCount, log a warning and set adjusted_count to the default value.
                - If it exceeds the default value, log a warning and set adjusted_count to the default value.
                - If it is an odd number, adjust it to the nearest lower even number and log a warning about the adjustment.
                - If it meets all criteria, set adjusted_count to the maximum_thread_count as is.
            5. Return the adjusted_count.

        Notes:
            - This method is important for ensuring that the maximum thread count is appropriate for the system's capabilities and current configuration.
            - Proper logging provides visibility into potential misconfigurations and ensures that defaults are clearly communicated.
        """

        default_value = int(self.PhysicalCores * 2)
        if maximum_thread_count is None:
            self.Logger.warning(f"Maximum thread count not set, using default value: {default_value}.")
            return default_value

        if not isinstance(maximum_thread_count, int):
            self.Logger.warning(f"Invalid type for maximum thread count '{maximum_thread_count}'. Must be an integer; using default value: {default_value}.")
            return default_value

        if maximum_thread_count < self.CoreThreadCount.value:
            self.Logger.warning(f"Maximum thread count {maximum_thread_count} is less than the minimum default value: {default_value}. Using default value.")
            adjusted_count = default_value
        elif maximum_thread_count > default_value:
            self.Logger.warning(f"Maximum thread count {maximum_thread_count} exceeds the maximum default value: {default_value}. Using default value.")
            adjusted_count = default_value
        elif maximum_thread_count % 2 != 0:
            adjusted_count = maximum_thread_count - 1
            self.Logger.warning(f"Adjusted maximum thread count to even number: {adjusted_count}.")
        else:
            adjusted_count = maximum_thread_count

        return adjusted_count

    def _validateIdleCleanupThreshold(self, idle_cleanup_threshold: Optional[int]) -> int:
        """
        Validates the idle cleanup threshold value and returns a valid integer.

        :param idle_cleanup_threshold: An optional integer representing the desired idle cleanup threshold.

        :return: The validated idle cleanup threshold value, which may be the provided value or a default.

        steps:
            1. Define a default value for the idle cleanup threshold (default_value = 60).
            2. Check if the provided idle_cleanup_threshold is None:
                - If it is, log a warning indicating that the default value will be used and return the default value.
            3. Check if the provided idle_cleanup_threshold is not an integer:
                - If it is of an invalid type, log a warning message about the invalid type and return the default value.
            4. If the provided idle_cleanup_threshold is valid, return it as is.

        Notes:
            - This method is essential for ensuring that the configuration for idle cleanup thresholds is valid and usable.
            - Proper logging provides visibility into potential misconfigurations and ensures that defaults are clearly communicated.
        """

        default_value = 60
        if idle_cleanup_threshold is None:
            self.Logger.warning(f"Idle cleanup threshold not set, using default value: {default_value}.")
            return default_value

        if not isinstance(idle_cleanup_threshold, int):
            self.Logger.warning(f"Invalid type for idle cleanup threshold '{idle_cleanup_threshold}'. Must be an integer; using default value: {default_value}.")
            return default_value

        return idle_cleanup_threshold

    def _validateTaskThreshold(self, task_threshold: Optional[int]) -> int:
        """
        Validates the task threshold value and returns a valid integer.

        :param task_threshold: An optional integer representing the desired task threshold.

        :return: The validated task threshold value, which may be the provided value or a default.

        steps:
            1. Calculate the default value for the task threshold using the calculateTaskThreshold method.
            2. Check if the provided task_threshold is None:
                - If it is, log a warning indicating that the default value will be used and return the default value.
            3. Check if the provided task_threshold is not an integer:
                - If it is of an invalid type, log a warning message about the invalid type and return the default value.
            4. If the provided task_threshold is valid, return it as is.

        Notes:
            - This method is essential for ensuring that the configuration for task thresholds is valid and usable.
            - Logging warnings provides visibility into potential misconfigurations and ensures that defaults are clearly communicated.
        """

        default_value = self.calculateTaskThreshold()
        if task_threshold is None:
            self.Logger.warning(f"Task threshold not set, using default value: {default_value}.")
            return default_value

        if not isinstance(task_threshold, int):
            self.Logger.warning(f"Invalid type for task threshold '{task_threshold}'. Must be an integer; using default value: {default_value}.")
            return default_value

        return task_threshold

    def _validateGlobalTaskThreshold(self, global_task_threshold: Optional[int]) -> int:
        """
        Validates the global task threshold value and returns a valid integer.

        :param global_task_threshold: An optional integer representing the desired global task threshold.

        :return: The validated global task threshold value, which may be the provided value or a default.

        steps:
            1. Calculate the default value for the global task threshold based on the core process count, core thread count, and the configured task threshold.
            2. Check if the provided global_task_threshold is None:
                - If it is, log a warning indicating that the default value will be used and return the default value.
            3. Check if the provided global_task_threshold is not an integer:
                - If it is of an invalid type, log a warning message about the invalid type and return the default value.
            4. If the provided global_task_threshold is valid, return it as is.

        Notes:
            - This method is essential for ensuring that the configuration for global task thresholds is valid and usable.
            - Logging warnings helps provide visibility into potential misconfigurations and ensures that defaults are clearly communicated.
        """

        default_value = int((self.CoreProcessCount.value + self.CoreThreadCount.value) * self.TaskThreshold.value)
        if global_task_threshold is None:
            self.Logger.warning(f"Global task threshold not set, using default value: {default_value}.")
            return default_value

        if not isinstance(global_task_threshold, int):
            self.Logger.warning(f"Invalid type for global task threshold '{global_task_threshold}'. Must be an integer; using default value: {default_value}.")
            return default_value

        return global_task_threshold

    def _validateProcessPriority(self, process_priority: Literal[None, "IDLE", "BELOW_NORMAL", "NORMAL", "ABOVE_NORMAL", "HIGH", "REALTIME"]) -> str:
        """
        Validates the provided process priority and returns a valid priority string.

        :param process_priority: The process priority to validate, which can be None, "IDLE", "BELOW_NORMAL", "NORMAL", "ABOVE_NORMAL", "HIGH", or "REALTIME".

        :return: A validated process priority string, which may be the provided value or a default.

        steps:
            1. Check if the provided process_priority is in the list of valid priorities [None, "IDLE", "BELOW_NORMAL", "NORMAL", "ABOVE_NORMAL", "HIGH", "REALTIME"]:
                - If it is not valid, log a warning indicating the invalid priority and return the default value "NORMAL".
            2. If the provided process_priority is None:
                - Log a warning indicating that the process priority is not set and return the default value "NORMAL".
            3. If the CoreProcessCount is equal to the number of physical cores and the process_priority is either "HIGH" or "REALTIME":
                - Log a warning indicating that using this priority is not recommended and return the adjusted default value "ABOVE_NORMAL".
            4. If the provided process_priority is valid, return it as is.

        Notes:
            - This method is crucial for ensuring that the configuration for process priorities is valid and suitable for the system's resource capabilities.
            - Proper logging provides visibility into potential misconfigurations and recommendations for optimal performance.
        """

        if process_priority not in [None, "IDLE", "BELOW_NORMAL", "NORMAL", "ABOVE_NORMAL", "HIGH", "REALTIME"]:
            self.Logger.warning(f"Invalid process priority '{process_priority}'. Using default value: NORMAL.")
            return "NORMAL"

        if process_priority is None:
            self.Logger.warning("Process priority not set, using default value: NORMAL.")
            return "NORMAL"

        if self.CoreProcessCount.value == self.PhysicalCores and process_priority in ["HIGH", "REALTIME"]:
            self.Logger.warning(f"Process priority {process_priority} is not recommended for all physical cores; using default value: ABOVE_NORMAL.")
            return "ABOVE_NORMAL"
        return process_priority

    def _validateExpandPolicy(self, expand_policy: Literal[None, "NoExpand", "AutoExpand", "BeforehandExpand"]) -> str:
        """
        Validates the provided expand policy and returns a valid policy string.

        :param expand_policy: The expand policy to validate, which can be None, "NoExpand", "AutoExpand", or "BeforehandExpand".

        :return: A validated expand policy string, which may be the provided value or a default.

        steps:
            1. Check if the provided expand_policy is in the list of valid policies [None, "NoExpand", "AutoExpand", "BeforehandExpand"]:
                - If it is not valid, log a warning indicating the invalid policy and return the default value "NoExpand".
            2. If the provided expand_policy is None:
                - Log a warning indicating that the expand policy is not set and return the default value "NoExpand".
            3. If the provided expand_policy is valid, return it as is.

        Notes:
            - This method is essential for ensuring that the configuration for expand policies is valid and usable.
            - Logging warnings provides visibility into potential misconfigurations and ensures that defaults are clearly communicated.
        """

        if expand_policy not in [None, "NoExpand", "AutoExpand", "BeforehandExpand"]:
            self.Logger.warning(f"Invalid expand policy '{expand_policy}'. Using default value: NoExpand.")
            return "NoExpand"

        if expand_policy is None:
            self.Logger.warning("Expand policy not set, using default value: NoExpand.")
            return "NoExpand"

        return expand_policy

    def _validateShrinkagePolicy(self, shrinkage_policy: Literal[None, "NoShrink", "AutoShrink", "TimeoutShrink"]) -> str:
        """
        Validates the provided shrinkage policy and returns a valid policy string.

        :param shrinkage_policy: The shrinkage policy to validate, which can be None, "NoShrink", "AutoShrink", or "TimeoutShrink".

        :return: A validated shrinkage policy string, which may be the provided value or a default.

        steps:
            1. Check if the provided shrinkage_policy is in the list of valid policies [None, "NoShrink", "AutoShrink", "TimeoutShrink"]:
                - If it is not valid, log a warning indicating the invalid policy and return the default value "NoShrink".
            2. If the provided shrinkage_policy is None:
                - Log a warning indicating that the shrinkage policy is not set and return the default value "NoShrink".
            3. If the provided shrinkage_policy is valid, return it as is.

        Notes:
            - This method is essential for ensuring that the configuration for shrinkage policies is valid and usable.
            - Logging warnings provides visibility into potential misconfigurations and ensures that defaults are clearly communicated.
        """

        if shrinkage_policy not in [None, "NoShrink", "AutoShrink", "TimeoutShrink"]:
            self.Logger.warning(f"Invalid shrinkage policy '{shrinkage_policy}'. Using default value: NoShrink.")
            return "NoShrink"

        if shrinkage_policy is None:
            self.Logger.warning("Shrinkage policy not set, using default value: NoShrink.")
            return "NoShrink"

        return shrinkage_policy

    def _validateShrinkagePolicyTimeout(self, shrinkage_policy_timeout: Optional[int]) -> int:
        """
        Validates the shrinkage policy timeout value and returns a valid integer.

        :param shrinkage_policy_timeout: An optional integer representing the desired shrinkage policy timeout.

        :return: The validated shrinkage policy timeout value, which may be the provided value or a default.

        steps:
            1. Define a default value for the shrinkage policy timeout (default_value = 15).
            2. Check if the provided shrinkage_policy_timeout is None:
                - If it is, log a warning indicating that the default value will be used and return the default value.
            3. Check if the provided shrinkage_policy_timeout is not an integer:
                - If it is of an invalid type, log a warning message about the invalid type and return the default value.
            4. If the provided shrinkage_policy_timeout is valid, return it as is.

        Notes:
            - This method is essential for ensuring that the configuration for shrinkage policies is valid and usable.
            - Logging warnings helps provide visibility into potential misconfigurations and ensures that defaults are clearly communicated.
        """

        default_value = 15
        if shrinkage_policy_timeout is None:
            self.Logger.warning(f"Shrinkage policy timeout not set, using default value: {default_value}.")
            return default_value

        if not isinstance(shrinkage_policy_timeout, int):
            self.Logger.warning(f"Invalid type for shrinkage policy timeout '{shrinkage_policy_timeout}'. Must be an integer; using default value: {default_value}.")
            return default_value

        return shrinkage_policy_timeout

    def calculateTaskThreshold(self):
        """
        Calculates the task threshold based on the system's physical cores and total memory.

        :return: An integer representing the calculated task threshold.

        steps:
            1. Retrieve the number of physical CPU cores from the instance variable (PhysicalCores).
            2. Get the total system memory in gigabytes by calling _SystemTools.total_memory() and converting bytes to gigabytes.
            3. Calculate the balanced score using the formula:
               - balanced_score = ((physical_cores / 128) + (total_memory / 3072)) / 2
            4. Define thresholds for balanced scores and corresponding task thresholds:
               - balanced_score_thresholds = [0.2, 0.4, 0.6, 0.8]
               - task_thresholds = [40, 80, 120, 160, 200]
            5. Iterate through the balanced_score_thresholds and task_thresholds:
               - If the calculated balanced_score is less than or equal to a score_threshold, return the corresponding task threshold.
            6. If the balanced score exceeds all defined thresholds, return the highest task threshold.

        Notes:
            - This method is important for dynamically determining the optimal number of tasks that can be processed based on the system's resources.
            - By balancing CPU cores and memory, it helps optimize task execution and resource utilization.
        """

        physical_cores = self.PhysicalCores
        total_memory = _Monitor.totalPhysicalMemory() / (1024 ** 3)
        balanced_score = ((physical_cores / 128) + (total_memory / 3072)) / 2

        balanced_score_thresholds = [0.2, 0.4, 0.6, 0.8]
        task_thresholds = [40, 80, 120, 160, 200]
        for score_threshold, threshold in zip(balanced_score_thresholds, task_thresholds):
            if balanced_score <= score_threshold:
                return threshold
        return task_thresholds[-1]


class _SynchronizationManager:
    """
    Manages synchronization across processes and threads in the concurrent system.

    This class is responsible for maintaining status pools for processes and threads, as well as handling shared resources and synchronization locks.

    Attributes:
        SharedObjectManagerID: ID of the shared object manager process.
        CoreProcessStatusPool: Dictionary maintaining status information for core processes.
        ExpandProcessStatusPool: Dictionary maintaining status information for expanded processes.
        CoreThreadStatusPool: Dictionary maintaining status information for core threads.
        ExpandThreadStatusPool: Dictionary maintaining status information for expanded threads.
        ResultStorageQueue: Queue for storing results from completed tasks.
        TaskLock: Lock for synchronizing access to shared resources across processes.

    Methods:
        (No methods defined, but the class provides a structured way to manage synchronization-related data.)

    Notes:
        - The synchronization manager is crucial for ensuring that processes and threads can operate without conflicts, especially when accessing shared resources.
        - Proper management of status pools and locks helps maintain the integrity and performance of the concurrent system.
    """

    def __init__(self, SharedObjectManager: multiprocessing.Manager):
        # noinspection PyProtectedMember
        self.SharedObjectManagerID = SharedObjectManager._process.pid
        self.CoreProcessStatusPool: Dict[str, Tuple[int, int, int]] = SharedObjectManager.dict()
        self.ExpandProcessStatusPool: Dict[str, Tuple[int, int, int]] = SharedObjectManager.dict()
        self.CoreThreadStatusPool: Dict[str, Tuple[int, int]] = SharedObjectManager.dict()
        self.ExpandThreadStatusPool: Dict[str, Tuple[int, int]] = SharedObjectManager.dict()
        self.ResultStorageQueue: multiprocessing.Queue = multiprocessing.Queue()
        self.TaskLock: multiprocessing.Lock = multiprocessing.Lock()


class _TaskObject:
    """
    Represents a task to be executed in the concurrent system.

    This class encapsulates a callable task along with its configuration and state, managing parameters for execution, serialization, and GPU support.

    Attributes:
        Task: The callable function or coroutine to be executed.
        TaskID: Unique identifier for the task.
        TaskType: Type of the task, either "Sync" or "Async".
        IsCallback: Indicates if the task has a callback function.
        Lock: Indicates if the task requires a lock.
        LockTimeout: Timeout duration for acquiring the lock.
        TimeOut: Timeout duration for task execution.
        IsGpuBoost: Indicates if GPU acceleration is enabled for the task.
        GpuID: ID of the GPU to use for the task.
        IsRetry: Indicates if the task should be retried on failure.
        MaxRetries: Maximum number of retries allowed for the task.
        RetriesCount: Counter for the number of retries attempted.
        UnserializableInfo: Dictionary to hold non-serializable objects.
        Args: Serialized positional arguments for the task.
        Kwargs: Serialized keyword arguments for the task.
        RecoveredArgs: Deserialized positional arguments for the task.
        RecoveredKwargs: Deserialized keyword arguments for the task.

    Methods:
        reinitializeParams: Deserializes the parameters for task execution.
        setupGpuParams: Prepares parameters for GPU execution if applicable.
        paramsTransfer: Transfers parameters to the specified device (GPU).
        cleanupGpuResources: Cleans up GPU resources after execution.
        serialize: Serializes an object for safe storage and transmission.
        deserialize: Deserializes an object back to its original form.
        isSerializable: Checks if an object can be serialized.
        execute: Runs the task with timeout handling and retry logic.
        run: Executes the task, determining its type (synchronous or asynchronous).
        retry: Attempts to rerun the task in case of failure.
        result: Processes the result of the executed task, handling GPU results.

    Notes:
        - The task object is designed to facilitate asynchronous execution while providing robust handling for retries, GPU support, and parameter serialization.
        - Proper cleanup of GPU resources is essential to avoid memory leaks during execution.
    """

    def __init__(self, Task: callable, TaskID: str, Callback: bool = False, Lock: bool = False, LockTimeout: int = 3, TimeOut: int = None, GpuBoost: bool = False, GpuID: int = 0, Retry: bool = False, MaxRetries: int = 3, *args, **kwargs):
        self.Task = Task
        self.TaskID = TaskID
        self.TaskType: Literal["Sync", "Async"] = "Async" if asyncio.iscoroutinefunction(self.Task) else "Sync"
        self.IsCallback = Callback
        self.Lock = Lock
        self.LockTimeout = LockTimeout
        self.TimeOut = TimeOut
        self.IsGpuBoost = GpuBoost
        self.GpuID = GpuID
        self.IsRetry = Retry
        self.MaxRetries = MaxRetries
        self.RetriesCount = 0
        self.UnserializableInfo = {}
        self.Args = self.serialize(args)
        self.Kwargs = self.serialize(kwargs)
        self.RecoveredArgs = None
        self.RecoveredKwargs = None

    def reinitializeParams(self):
        """
        Reinitialized the parameters for the task by deserializing arguments and setting up GPU parameters if applicable.

        steps:
            1. Deserialize the arguments (Args) and store the result in RecoveredArgs.
            2. Deserialize the keyword arguments (Kwargs) and store the result in RecoveredKwargs.
            3. If GPU boost is enabled (IsGpuBoost) and PyTorch support is available:
                - Call the setupGpuParams method to transfer parameters to the appropriate GPU device.

        Notes:
            - This method is essential for resetting the parameters before task execution, ensuring that they are in the correct format and on the appropriate device.
            - Proper handling of deserialization allows for dynamic adjustment of parameters based on the current context of execution.
        """

        self.RecoveredArgs = self.deserialize(self.Args)
        self.RecoveredKwargs = self.deserialize(self.Kwargs)
        if self.IsGpuBoost and PyTorchSupport:
            self.setupGpuParams()

    def setupGpuParams(self):
        """
        Sets up GPU parameters for processing by transferring arguments to the specified GPU device.

        steps:
            1. Check if the specified GPU ID (GpuID) is available in the list of available CUDA devices:
                - If not, reset GpuID to 0 (defaulting to the first available device).
            2. Create a device object representing the target GPU using the specified GpuID.
            3. Transfer each argument in RecoveredArgs to the specified GPU device using the paramsTransfer method.
            4. Transfer each value in RecoveredKwargs to the specified GPU device, keeping the keys unchanged.

        Notes:
            - This method is crucial for ensuring that the correct GPU is used for computations and that all necessary parameters are moved to the GPU memory.
            - It helps prevent issues related to accessing unavailable GPU resources and optimizes performance by utilizing the specified device.
        """

        if self.GpuID not in AvailableCUDADevicesID:
            self.GpuID = 0
        device = torch.device(f"cuda:{self.GpuID}")
        self.RecoveredArgs = [self.paramsTransfer(arg, device) for arg in self.RecoveredArgs]
        self.RecoveredKwargs = {key: self.paramsTransfer(value, device) for key, value in self.RecoveredKwargs.items()}

    def paramsTransfer(self, obj, device):
        """
        Transfers parameters or tensors to the specified device (CPU or GPU).

        :param obj: The object to be transferred, which may be a tensor, module, list, tuple, or dictionary.
        :param device: The target device to which the object should be transferred (e.g., 'cuda' or 'cpu').

        :return: The object transferred to the specified device.

        steps:
            1. Check if the object is a PyTorch tensor or a neural network module:
                - If it is, transfer the object to the specified device using the .to() method.
            2. Check if the object is a list or tuple:
                - If it is, create a new list or tuple by recursively transferring each element to the specified device.
            3. Check if the object is a dictionary:
                - If it is, create a new dictionary by recursively transferring each value to the specified device while keeping the keys unchanged.
            4. If the object is of any other type, return it as is (indicating it doesn't require a device transfer).

        Notes:
            - This method is useful for preparing data for processing on different devices, ensuring compatibility with PyTorch's operations on CPU and GPU.
            - It allows for seamless movement of data structures containing tensors or models without needing to handle each case individually.
        """

        if isinstance(obj, (torch.Tensor, torch.nn.Module)):
            return obj.to(device)
        if isinstance(obj, (list, tuple)):
            return type(obj)(self.paramsTransfer(x, device) for x in obj)
        if isinstance(obj, dict):
            return {k: self.paramsTransfer(v, device) for k, v in obj.items()}
        return obj

    @staticmethod
    def cleanupGpuResources():
        """
        Cleans up GPU resources by synchronizing and clearing the CUDA cache.

        steps:
            1. Call torch.cuda.synchronize() to ensure that all pending CUDA operations are completed.
            2. Call torch.cuda.empty_cache() to free up unused memory currently held by the CUDA allocator.

        Notes:
            - This method is important for managing GPU memory usage and preventing memory leaks in applications utilizing PyTorch.
            - Regular cleanup of GPU resources can help improve performance and stability, especially in long-running applications or when dealing with large datasets.
        """

        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    def serialize(self, obj):
        """
        Serializes an object, handling tuples, dictionaries, and unserializable objects appropriately.

        :param obj: The object to be serialized, which may be a tuple, dictionary, or other types.

        :return: The serialized representation of the object.

        steps:
            1. Check if the object is a tuple:
                - If it is, return a new tuple created by recursively serializing each item in the original tuple.
            2. Check if the object is a dictionary:
                - Return a new dictionary created by recursively serializing each key-value pair.
            3. Check if the object is serializable using the isSerializable method:
                - If the object is not serializable:
                    - Store the object in UnserializableInfo with its unique ID (using the built-in id function).
                    - Return a dictionary indicating that the object is unserializable, including its ID and type name.
            4. If the object is serializable, return it as is.

        Notes:
            - This method is useful for preparing objects for serialization, ensuring that complex structures like tuples and dictionaries are handled correctly.
            - It provides a mechanism to keep track of objects that cannot be serialized, allowing for future reference or special handling.
        """

        if isinstance(obj, tuple):
            return tuple(self.serialize(item) for item in obj)
        elif isinstance(obj, dict):
            return {key: self.serialize(value) for key, value in obj.items()}
        elif not self.isSerializable(obj):
            obj_id = id(obj)
            self.UnserializableInfo[obj_id] = obj
            return {"__unserializable__": True, "id": obj_id, "type": type(obj).__name__}
        else:
            return obj

    def deserialize(self, obj):
        """
        Deserializes an object, handling tuples and dictionaries appropriately.

        :param obj: The object to be deserialized, which may be a tuple, dictionary, or other types.

        :return: The deserialized object.

        steps:
            1. Check if the object is a tuple:
                - If it is, return a new tuple created by recursively deserializing each item in the original tuple.
            2. Check if the object is a dictionary:
                - If the dictionary contains the key "__unserializable__", return the corresponding unserializable info using the stored id.
                - Otherwise, return a new dictionary created by recursively deserializing each key-value pair.
            3. If the object is neither a tuple nor a dictionary, return it as is (indicating it is already in its final form).

        Notes:
            - This method is useful for reconstructing objects from their serialized forms, ensuring that complex structures like tuples and dictionaries are handled correctly.
            - It provides special handling for unserializable objects based on predefined logic (using "__unserializable__").
        """

        if isinstance(obj, tuple):
            return tuple(self.deserialize(item) for item in obj)
        elif isinstance(obj, dict):
            if "__unserializable__" in obj:
                return self.UnserializableInfo[obj["id"]]
            return {key: self.deserialize(value) for key, value in obj.items()}
        else:
            return obj

    @staticmethod
    def isSerializable(obj):
        """
        Checks if an object is serializable using the pickle module.

        :param obj: The object to be tested for serialization.

        :return: A boolean indicating whether the object can be serialized (True) or not (False).

        steps:
            1. Attempt to serialize the object using pickle.dumps:
                - If serialization is successful, return True.
            2. If a PicklingError, AttributeError, or TypeError occurs during serialization:
                - Catch the exception and return False.

        Notes:
            - This method is useful for determining if an object can be safely serialized before attempting to do so.
            - It helps prevent runtime errors that may occur during serialization of non-serializable objects.
        """

        try:
            pickle.dumps(obj)
            return True
        except (pickle.PicklingError, AttributeError, TypeError):
            return False

    async def execute(self):
        """
        Executes the task, handling timeouts, retries, and GPU resource cleanup.

        steps:
            1. Reinitialize parameters necessary for the task execution by calling the reinitializeParams method.
            2. In a try block:
                - If TimeOut is specified, use asyncio.wait_for to execute the run method with the specified timeout:
                    - If the task completes within the timeout, store the result.
                - If TimeOut is not specified, simply await the execution of the run method.
            3. Handle exceptions that may occur during execution:
                - If a TimeoutError occurs, return without further action.
                - If a CancelledError occurs, return without further action.
                - For other exceptions:
                    - If IsRetry is enabled, attempt to retry the execution by calling the retry method.
                    - If retries are not allowed, raise an exception with a detailed error message including the traceback.
            4. In the finally block, if IsGpuBoost is enabled, clean up GPU resources by calling the cleanupGpuResources method.
            5. Return the result of the task execution.

        Notes:
            - This method provides robust handling of task execution scenarios, ensuring that timeouts and exceptions are properly managed.
            - The ability to retry tasks and manage GPU resources enhances the flexibility and reliability of task execution.
        """

        self.reinitializeParams()
        try:
            if self.TimeOut is not None:
                result = await asyncio.wait_for(self.run(), timeout=self.TimeOut)
            else:
                result = await self.run()

        except asyncio.TimeoutError:
            return
        except asyncio.CancelledError:
            return
        except Exception as e:
            if self.IsRetry:
                result = await self.retry()
            else:
                raise Exception(f"Failed to execute task {self.Task.__name__} due to {str(e)}\n\n{traceback.format_exc()}.")
        finally:
            if self.IsGpuBoost:
                self.cleanupGpuResources()
        return result

    async def run(self):
        """
        Executes the assigned task and processes the result based on the task type.

        steps:
            1. Check the type of the task:
                - If the TaskType is "Async":
                    - Await the execution of the task using the provided arguments (RecoveredArgs and RecoveredKwargs).
                    - Store the result in task_result.
                - If the TaskType is not "Async":
                    - Execute the task synchronously with the provided arguments and store the result in task_result.
            2. Process the task result by awaiting the result method and return its outcome.

        Notes:
            - This method allows for flexible task execution, supporting both asynchronous and synchronous tasks.
            - Proper handling of task results ensures that any necessary post-processing is applied based on the task type.
        """

        if self.TaskType == "Async":
            task_result = await self.Task(*self.RecoveredArgs, **self.RecoveredKwargs)
        else:
            task_result = self.Task(*self.RecoveredArgs, **self.RecoveredKwargs)
        return await self.result(task_result)

    async def retry(self):
        """
        Retries the execution of a task with exponential backoff in case of failure.

        steps:
            1. Initialize a backoff time starting at 0.1 seconds.
            2. Enter a loop that continues until the number of retries (RetriesCount) reaches the maximum allowed retries (MaxRetries):
                - Try to run the task by awaiting the run method.
                - If the task executes successfully, return the result.
                - If an exception occurs during execution:
                    - Increment the retry count (RetriesCount).
                    - Sleep for the current backoff time to implement a delay before the next retry.
                    - Double the backoff time for the next iteration to increase the delay progressively.

        Notes:
            - This method is useful for handling transient errors where retrying the operation may succeed after a delay.
            - Exponential backoff helps to reduce the load on the system and avoids overwhelming resources during repeated failures.
        """

        backoff_time = 0.1
        while self.RetriesCount < self.MaxRetries:
            # noinspection PyBroadException
            try:
                return await self.run()
            except Exception:
                self.RetriesCount += 1
                await asyncio.sleep(backoff_time)
                backoff_time *= 2

    async def result(self, task_result):
        """
        Processes the result of a task, managing GPU resources if applicable.

        :param task_result: The result obtained from the task, which may be a GPU tensor or module.

        steps:
            1. Check if GPU boost is enabled and if the task result is an instance of torch.Tensor or torch.nn.Module:
                - If both conditions are met:
                    - Clone the task result, detach it from the GPU, and move it to the CPU.
                    - Delete the original task result to free GPU memory.
                    - Call the cleanupGpuResources method to release any GPU resources.
                    - Return the CPU result.
            2. If the GPU boost is not enabled or the task result is not a tensor/module, return the original task result.

        Notes:
            - This method is important for efficiently handling GPU resources, ensuring that memory is managed properly.
            - It allows for seamless transitions between GPU and CPU operations while minimizing resource usage.
        """

        if self.IsGpuBoost and isinstance(task_result, (torch.Tensor, torch.nn.Module)):
            cpu_result = task_result.clone().detach().cpu()
            del task_result
            self.cleanupGpuResources()
            return cpu_result
        return task_result


class _ProcessObject(multiprocessing.Process):
    """
    Represents a process that handles task execution in a concurrent system.

    This class extends `multiprocessing.Process` and manages tasks with different priority levels, using an event loop to handle asynchronous execution.

    Attributes:
        ProcessName: Name of the process.
        ProcessType: Type of the process (Core or Expand).
        SynchronizationManager: Manages synchronization across tasks.
        ConfigManager: Manages system configuration settings.
        DebugMode: Flag indicating whether debug mode is active.
        Logger: Logger for the process object.
        PendingTasks: Dictionary of currently pending tasks.
        HighPriorityQueue: Queue for high-priority tasks.
        MediumPriorityQueue: Queue for medium-priority tasks.
        LowPriorityQueue: Queue for low-priority tasks.
        WorkingEvent: Event to indicate if the process is currently working on tasks.
        CloseEvent: Event to signal the process to stop.
        EventLoop: Event loop for managing asynchronous tasks.

    Methods:
        run: Starts the process, sets up logging and priority, and runs the task processor.
        stop: Signals the process to stop and waits for it to finish.
        addProcessTask: Adds a task to the appropriate priority queue.
        _setLogger: Configures logging for the process.
        _setProcessPriority: Sets the process priority based on configuration.
        _setStatusUpdateThread: Starts a thread to update the process status.
        _cleanupProcessMemory: Cleans up memory for the process.
        _updateProcessStatus: Updates the process status in the synchronization manager.
        _taskProcessor: Main loop for processing tasks from the queues.
        _taskExecutor: Executes a given task and handles its result.
        _taskResultProcessor: Processes the result of a completed task.
        _requeueTask: Requeues a task if it cannot be executed due to locking issues.
        _cleanup: Cleans up remaining tasks when stopping the process.

    Notes:
        - The process manages tasks based on their priority, ensuring that higher-priority tasks are executed first.
        - Proper management of task execution, memory cleanup, and resource monitoring is essential for system efficiency.
    """

    def __init__(self, ProcessName: str, ProcessType: Literal['Core', 'Expand'], SM: _SynchronizationManager, CM: _ConfigManager, DebugMode: bool):
        super().__init__(name=ProcessName, daemon=True)
        self.ProcessName = ProcessName
        self.ProcessType = ProcessType
        self.SynchronizationManager = SM
        self.ConfigManager = CM
        self.DebugMode = DebugMode
        self.Logger = None
        self.PendingTasks = {}
        self.HighPriorityQueue: multiprocessing.Queue = multiprocessing.Queue()
        self.MediumPriorityQueue: multiprocessing.Queue = multiprocessing.Queue()
        self.LowPriorityQueue: multiprocessing.Queue = multiprocessing.Queue()
        self.WorkingEvent = multiprocessing.Event()
        self.CloseEvent = multiprocessing.Event()
        self.EventLoop: Optional[asyncio.AbstractEventLoop] = None

    def run(self):
        """
        Runs the executor, initializing logging, process priority, and setting up the event loop for task processing.

        steps:
            1. Call the _setLogger method to configure logging for the process.
            2. Call the _setProcessPriority method to set the execution priority of the process.
            3. Perform initial memory cleanup for the process by calling _cleanupProcessMemory.
            4. Create a new asyncio event loop to manage asynchronous tasks.
            5. Set up a thread for updating process status by calling _setStatusUpdateThread.
            6. Set the newly created event loop as the current event loop for asyncio operations.
            7. In a try-finally block, run the _taskProcessor coroutine until it completes:
                - Ensure that the event loop is properly closed after task processing is finished.
            8. In the finally block, log a debug message indicating that the process has been closed.

        Notes:
            - This method is essential for the proper operation of the executor, ensuring that all necessary configurations and setups are made before processing tasks.
            - Using a try-finally block ensures that resources are cleaned up correctly, preventing potential memory leaks or resource locks.
        """

        self._setLogger()
        self._setProcessPriority()
        self._cleanupProcessMemory()
        self.EventLoop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.EventLoop)
        try:
            self.EventLoop.run_until_complete(self._taskProcessor())
        finally:
            self.EventLoop.close()
            self.Logger.debug(f"[{self.ProcessName} - {self.pid}] has been closed.")

    def stop(self):
        """
        Stops the current thread by signaling it to terminate and waits for it to finish.

        steps:
            1. Set the CloseEvent to signal the thread to stop its operations.
            2. Wait for the thread to finish execution, allowing up to 2 seconds for it to complete.
            3. If the thread is still alive after the wait, call the terminate method to forcefully stop it.
            4. Delete the thread object to free up resources after it has stopped.

        Notes:
            - This method ensures a graceful shutdown of the thread, allowing for proper cleanup of any resources.
            - The use of the CloseEvent provides a clear mechanism for the thread to terminate its ongoing tasks safely.
            - The `terminate` method is used as a fallback to ensure the thread does not remain active unintentionally.
        """

        self.CloseEvent.set()
        self.join(2)
        if self.is_alive():
            self.terminate()
        del self

    def addProcessTask(self, priority: int, task_object: _TaskObject):
        """
        Adds a task to the appropriate priority queue based on its priority level.

        :param priority: An integer representing the priority of the task (0 to 10).
        :param task_object: The task object to be added to the queue.

        steps:
            1. Check the priority level of the task:
                - If the priority is between 0 and 3 (inclusive), add the task to the HighPriorityQueue.
                - If the priority is between 4 and 7 (inclusive), add the task to the MediumPriorityQueue.
                - If the priority is between 8 and 10 (inclusive), add the task to the LowPriorityQueue.
                - If the priority is outside the range of 0 to 10, log an error indicating that the task has been rejected due to invalid priority.
            2. If the WorkingEvent is not already set, set it to indicate that there are tasks to process.

        Notes:
            - This method ensures that tasks are organized according to their priority, enabling efficient processing.
            - Proper logging is implemented to handle cases where invalid priorities are provided, aiding in debugging and monitoring.
        """

        if 0 <= priority <= 3:
            self.HighPriorityQueue.put_nowait(("HighPriority", task_object))
        elif 4 <= priority <= 7:
            self.MediumPriorityQueue.put_nowait(("MediumPriority", task_object))
        elif 8 <= priority <= 10:
            self.LowPriorityQueue.put_nowait(("LowPriority", task_object))
        else:
            self.Logger.error(f"[{self.ProcessName} - {self.pid}] task {task_object.TaskID} has been rejected due to invalid priority {priority}.")

    def _setLogger(self):
        """
        Sets up and configures a logger for the current process.

        steps:
            1. Create a logger instance with a name that includes the process name and process ID (self.pid).
            2. Set the logging level of the logger to DEBUG.
            3. Create a console handler for outputting log messages to the console.
            4. Determine the log level for the console handler:
                - If DebugMode is enabled, set the log level to DEBUG.
                - If DebugMode is disabled, set the log level to the maximum of DEBUG and WARNING.
            5. Create a formatter to define the format of the log messages, including timestamp, logger name, level, and message.
            6. Set the formatter for the console handler.
            7. Add the console handler to the logger.
            8. Assign the configured logger to the instance variable (self.Logger) for use in the class.

        Notes:
            - This method ensures that the process logs are well-structured and can be easily read.
            - Logging configuration is essential for debugging and monitoring the behavior of the process.
        """

        logger = logging.getLogger(f"[ConcurrentSystem]")
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        if self.DebugMode:
            console_handler.setLevel(logging.DEBUG)
        else:
            console_handler.setLevel(max(logging.DEBUG, logging.WARNING))

        formatter = _ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        self.Logger = logger

    def _setProcessPriority(self, priority: Literal["IDLE", "BELOW_NORMAL", "NORMAL", "ABOVE_NORMAL", "HIGH", "REALTIME"] = None):
        """
        Sets the priority of the current process.

        :param priority: A string indicating the desired priority level for the process.
                         Acceptable values are "IDLE", "BELOW_NORMAL", "NORMAL", "ABOVE_NORMAL", "HIGH", "REALTIME".
                         If None, the priority will default to the value from ConfigManager.

        steps:
            1. Define a mapping of priority levels to their corresponding numeric values.
            2. Initialize a handle variable to None.
            3. In a try block:
                - Open a handle to the current process using its process ID (self.pid).
                - If the handle is invalid (0), raise a ValueError.
                - Set the process priority using the SetPriorityClass function:
                    - Use the priority from the method argument if provided; otherwise, use the configured priority from ConfigManager.
                    - If the priority setting fails (result is 0), retrieve the last error code and raise an Exception with the error code.
            4. In the finally block, ensure the process handle is closed if it was successfully opened.

        Notes:
            - This method is crucial for managing the performance of the process by adjusting its priority as needed.
            - Proper error handling and logging provide visibility into any issues that occur during the priority setting process.
        """

        priority_mapping = {
            "IDLE": 0x00000040,
            "BELOW_NORMAL": 0x00004000,
            "NORMAL": 0x00000020,
            "ABOVE_NORMAL": 0x00008000,
            "HIGH": 0x00000080,
            "REALTIME": 0x00000100
        }
        handle = None
        try:
            handle = ctypes.windll.kernel32.OpenProcess(0x1F0FFF, False, self.pid)
            if handle == 0:
                raise ValueError("Failed to obtain a valid handle")
            result = ctypes.windll.kernel32.SetPriorityClass(handle, priority_mapping.get(self.ConfigManager.ProcessPriority if priority is None else priority, priority_mapping["NORMAL"]))
            if result == 0:
                error_code = ctypes.windll.kernel32.GetLastError()
                raise Exception(f"Set priority failed with error code {error_code}.")
        except Exception as e:
            self.Logger.error(f"[{self.ProcessName} - {self.pid}] set priority failed due to {str(e)}")
        finally:
            if handle:
                ctypes.windll.kernel32.CloseHandle(handle)

    def _cleanupProcessMemory(self):
        """
        Cleans up memory used by the current process on Windows.

        This method attempts to free memory allocated to the process by calling
        the `EmptyWorkingSet` function on the process handle associated with the current process.

        steps:
            1. Check if the operating system is Windows.
            2. If the operating system is Windows:
                - Attempt to obtain a handle to the current process using the process ID (`pid`).
                - If the handle is not valid, raise a ValueError indicating failure to obtain a valid handle.
                - If successful, call `EmptyWorkingSet` to free memory allocated to the process.
                - Log any errors that occur during handle retrieval or memory cleanup.
                - If `EmptyWorkingSet` fails, log the error code obtained from `GetLastError`.
            3. Close the handle to free the resource.

        Notes:
            - This method is essential for managing memory usage effectively, particularly in long-running processes where memory fragmentation can occur.
            - Proper error handling ensures that failures during the cleanup process are logged and do not cause crashes.
        """

        system = platform.system()
        if system == "Windows":
            try:
                handle = ctypes.windll.kernel32.OpenProcess(0x1F0FFF, False, self.pid)
                if handle == 0:
                    raise ValueError("Failed to obtain a valid handle")
            except Exception as e:
                self.Logger.error(f"[{self.ProcessName} - {self.pid}] memory cleanup failed due to {str(e)}\n\n{traceback.format_exc()}.")
                return
            if not handle:
                self.Logger.error(f"[{self.ProcessName} - {self.pid}] failed to obtain a valid process handle.")
                return
            result = ctypes.windll.psapi.EmptyWorkingSet(handle)
            if result == 0:
                error_code = ctypes.windll.kernel32.GetLastError()
                self.Logger.error(f"[{self.ProcessName} - {self.pid}] memory cleanup failed with error code {error_code}.")
            ctypes.windll.kernel32.CloseHandle(handle)
            return

    async def _taskProcessor(self):
        """
        Processes tasks from different priority queues and manages task execution.

        This asynchronous method continuously checks for tasks in high, medium, and low priority queues,
        executing them accordingly while handling idle states and cleanup.

        steps:
            1. Initialize a variable to track idle time (idle_times).
            2. Enter a loop that continues until the CloseEvent is set:
                - Attempt to retrieve task data from the HighPriorityQueue, MediumPriorityQueue, or LowPriorityQueue in that order.
                - If no tasks are available, sleep briefly (0.001 seconds) and:
                    - If idle_times reaches the configured threshold, perform process memory cleanup.
                    - Increment idle_times by 0.001 seconds.
                    - Clear the WorkingEvent if it is set.
                    - Continue to the next iteration of the loop.
            3. If task data is retrieved, unpack it into task_priority and task_object.
            4. Attempt to execute the task:
                - If the task requires a lock, acquire the lock with a specified timeout:
                    - If the lock cannot be acquired, requeue the task and continue.
                    - If acquired, create a coroutine task for executing the task object, store it in PendingTasks, and await its completion.
                    - Ensure to release the lock in a finally block.
                - If the task does not require a lock, create a coroutine task for execution without locking.
            5. If an exception occurs during task processing, log an error message with details about the failure.

        Notes:
            - This method is essential for managing the execution of tasks in a prioritized manner.
            - Proper error handling and cleanup ensure that the system remains efficient and responsive.
            - The asynchronous design allows for concurrent execution of tasks without blocking the event loop.
        """

        idle_times = 0
        while not self.CloseEvent.is_set():
            try:
                if not self.HighPriorityQueue.empty():
                    task_data: Tuple[str, _TaskObject] = self.HighPriorityQueue.get_nowait()
                elif not self.MediumPriorityQueue.empty():
                    task_data: Tuple[str, _TaskObject] = self.MediumPriorityQueue.get_nowait()
                elif not self.LowPriorityQueue.empty():
                    task_data: Tuple[str, _TaskObject] = self.LowPriorityQueue.get_nowait()
                else:
                    await asyncio.sleep(0.001)
                    if idle_times == self.ConfigManager.IdleCleanupThreshold.value:
                        self._cleanupProcessMemory()
                        idle_times = 0
                    idle_times += 0.001
                    if self.WorkingEvent.is_set():
                        self.WorkingEvent.clear()
                    continue
            except queue.Empty:
                await asyncio.sleep(0.001)
                if idle_times == self.ConfigManager.IdleCleanupThreshold.value:
                    self._cleanupProcessMemory()
                    idle_times = 0
                idle_times += 0.001
                if self.WorkingEvent.is_set():
                    self.WorkingEvent.clear()
                continue
            task_priority, task_object = task_data
            try:
                if task_object.Lock:
                    acquired = self.SynchronizationManager.TaskLock.acquire(timeout=task_object.LockTimeout)
                    if not acquired:
                        await self._requeueTask(task_priority, task_object)
                        continue
                    try:
                        coroutine_task = self.EventLoop.create_task(self._taskExecutor(task_object))
                        self.PendingTasks[task_object.TaskID] = coroutine_task
                        await coroutine_task
                    finally:
                        self.SynchronizationManager.TaskLock.release()
                else:
                    coroutine_task = self.EventLoop.create_task(self._taskExecutor(task_object))
                    self.PendingTasks[task_object.TaskID] = coroutine_task
            except Exception as e:
                self.Logger.error(f"[{self.ProcessName} - {self.pid}] task {task_object.TaskID} failed due to {str(e)}\n\n{traceback.format_exc()}.")
        await self._cleanup()

    async def _taskExecutor(self, task_object: _TaskObject):
        """
        Executes a given task asynchronously and processes the result.

        :param task_object: The task object to be executed.

        steps:
            1. Try to execute the task by awaiting its execute method:
                - Capture the result of the task execution.
            2. Process the task result using the _taskResultProcessor method, passing the result and task object.
            3. If the task execution raises a CancelledError:
                - Log a warning message indicating that the task has been cancelled.

        Notes:
            - This method is essential for handling the execution and result processing of tasks in an asynchronous environment.
            - Proper error handling ensures that cancellations are logged for monitoring and debugging purposes.
        """

        try:
            task_result = await task_object.execute()
            await self._taskResultProcessor(task_result, task_object)
        except asyncio.CancelledError:
            self.Logger.warning(f"[{self.ProcessName} - {self.pid}] task {task_object.TaskID} has been cancelled.")

    async def _taskResultProcessor(self, task_result: Any, task_object: _TaskObject):
        """
        Processes the result of a completed task and manages the corresponding task object.

        This asynchronous method stores the task result in the result storage queue and cleans up
        the pending tasks list by removing the completed task.

        :param task_result: The result obtained from executing the task.
        :param task_object: The object representing the task that has been completed.

        steps:
            1. Store the task result and its associated task ID in the ResultStorageQueue for further processing.
            2. Attempt to remove the task object from the PendingTasks dictionary using its TaskID:
                - If successful, the task is removed from the pending list.
                - If a KeyError occurs (indicating the task ID is not found), pass without taking action.
                - If any other exception occurs, log an error message indicating the failure to remove the task.

        Notes:
            - This method is critical for managing the lifecycle of tasks and their results within the system.
            - Proper error handling ensures that issues in removing tasks do not disrupt the overall task processing flow.
            - The use of asynchronous execution allows for efficient handling of task results without blocking other operations.
        """

        self.SynchronizationManager.ResultStorageQueue.put_nowait((task_result, task_object.TaskID))
        # noinspection PyBroadException
        try:
            del self.PendingTasks[task_object.TaskID]
        except KeyError:
            pass
        except Exception:
            self.Logger.error(f"[{self.ProcessName} - {self.pid}] failed to remove task {task_object.TaskID} from PendingTasks.")

    async def _requeueTask(self, task_priority: str, task_object: _TaskObject):
        """
        Requeues a task after a specified delay based on its priority and maximum retry attempts.

        :param task_priority: The priority of the task to be requeued (HighPriority, MediumPriority, LowPriority).
        :param task_object: The task object to be requeued.

        steps:
            1. Calculate the delay before requeuing the task:
                - Use an exponential backoff strategy to determine the delay, capping it at 10 seconds.
            2. Wait for the calculated delay using asyncio.sleep.
            3. Based on the task's priority:
                - If the priority is "HighPriority", add the task to the HighPriorityQueue.
                - If the priority is "MediumPriority", add the task to the MediumPriorityQueue.
                - If the priority is "LowPriority", add the task to the LowPriorityQueue.

        Notes:
            - This method is designed to handle task retries efficiently, allowing for gradual backoff to avoid overwhelming the system.
            - Using priority queues ensures that higher priority tasks are handled before lower priority ones.
        """

        await asyncio.sleep(min(0.1 * (2 ** task_object.MaxRetries), 10))
        if task_priority == "HighPriority":
            self.HighPriorityQueue.put_nowait((task_priority, task_object))
            return
        if task_priority == "MediumPriority":
            self.MediumPriorityQueue.put_nowait((task_priority, task_object))
            return
        if task_priority == "LowPriority":
            self.LowPriorityQueue.put_nowait((task_priority, task_object))
            return

    async def _cleanup(self):
        """
        Asynchronously cleans up remaining tasks from the priority queues and cancels pending tasks.

        steps:
            1. Initialize a counter (remaining_tasks) to track the number of discarded tasks.
            2. Check and discard tasks from the HighPriorityQueue:
                - While the queue is not empty, attempt to retrieve tasks without blocking.
                - Increment the remaining_tasks counter for each task retrieved.
            3. Repeat the above step for the MediumPriorityQueue and LowPriorityQueue.
            4. For each coroutine task in the PendingTasks:
                - Cancel the task to stop its execution.
            5. Log the number of discarded tasks for debugging purposes.
            6. Clear the PendingTasks dictionary to remove references to canceled tasks.

        Notes:
            - This method ensures that any tasks that are no longer needed are properly discarded and cleaned up.
            - Cancelling pending tasks helps to free up resources and avoid potential memory leaks.
            - Logging provides visibility into the cleanup operation and the number of tasks affected.
        """

        remaining_tasks = 0
        while not self.HighPriorityQueue.empty():
            try:
                _, task_object = self.HighPriorityQueue.get_nowait()
                remaining_tasks += 1
            except queue.Empty:
                break
        while not self.MediumPriorityQueue.empty():
            try:
                _, task_object = self.MediumPriorityQueue.get_nowait()
                remaining_tasks += 1
            except queue.Empty:
                break
        while not self.LowPriorityQueue.empty():
            try:
                _, task_object = self.LowPriorityQueue.get_nowait()
                remaining_tasks += 1
            except queue.Empty:
                break
        for i, coroutine_task in self.PendingTasks.items():
            coroutine_task.cancel()
        self.Logger.debug(f"[{self.ProcessName} - {self.pid}] discarded {remaining_tasks} tasks.")
        self.PendingTasks.clear()


class _ThreadObject(threading.Thread):
    """
    Represents a worker thread that processes tasks from different priority queues.

    This class extends `threading.Thread` and is responsible for executing tasks asynchronously using an event loop.

    Attributes:
        ThreadName: Name of the thread.
        ThreadType: Type of the thread (Core or Expand).
        SynchronizationManager: Manages synchronization across tasks.
        ConfigManager: Manages system configuration settings.
        Logger: Logger for the thread object.
        PendingTasks: Dictionary of currently pending tasks.
        HighPriorityQueue: Queue for high-priority tasks.
        MediumPriorityQueue: Queue for medium-priority tasks.
        LowPriorityQueue: Queue for low-priority tasks.
        WorkingEvent: Event to indicate if the thread is currently working on tasks.
        CloseEvent: Event to signal the thread to stop.
        EventLoop: Event loop for managing asynchronous tasks.

    Methods:
        run: Starts the event loop to process tasks.
        stop: Signals the thread to stop and waits for it to finish.
        addThreadTask: Adds a task to the appropriate priority queue.
        _taskProcessor: Main loop for processing tasks from the queues.
        _taskExecutor: Executes a given task and handles its result.
        _taskResultProcessor: Processes the result of a completed task.
        _requeueTask: Requeues a task if it cannot be executed due to locking issues.
        _cleanup: Cleans up remaining tasks when stopping the thread.

    Notes:
        - The thread processes tasks based on their priority, ensuring that higher-priority tasks are executed first.
        - Proper management of task execution and cleanup is essential for efficient resource usage.
    """

    def __init__(self, ThreadName: str, ThreadType: Literal['Core', 'Expand'], SM: _SynchronizationManager, CM: _ConfigManager, Logger: logging.Logger):
        super().__init__(name=ThreadName, daemon=True)
        self.ThreadName = ThreadName
        self.ThreadType = ThreadType
        self.SynchronizationManager = SM
        self.ConfigManager = CM
        self.Logger = Logger
        self.PendingTasks = {}
        self.HighPriorityQueue: queue.Queue = queue.Queue()
        self.MediumPriorityQueue: queue.Queue = queue.Queue()
        self.LowPriorityQueue: queue.Queue = queue.Queue()
        self.WorkingEvent = multiprocessing.Event()
        self.CloseEvent = multiprocessing.Event()
        self.EventLoop: Union[asyncio.AbstractEventLoop, None] = None

    def run(self):
        """
        Runs the thread by initializing and executing an asyncio event loop.

        steps:
            1. Create a new event loop for handling asynchronous tasks.
            2. Set the newly created event loop as the current event loop.
            3. Use a try-finally block to ensure proper cleanup:
                - Run the _taskProcessor coroutine until it completes.
            4. In the finally block, close the event loop to release resources.
            5. Log a debug message indicating that the thread has been stopped.

        Notes:
            - This method is essential for managing asynchronous task execution within the thread.
            - Proper handling of the event loop ensures that all resources are cleaned up correctly when the thread is stopped.
        """

        self.EventLoop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.EventLoop)
        try:
            self.EventLoop.run_until_complete(self._taskProcessor())
        finally:
            self.EventLoop.close()
            self.Logger.debug(f"[{self.ThreadName} - {self.ident}] has been stopped.")

    def stop(self):
        """
        Stops the current thread by signaling it to terminate and waiting for it to finish.

        steps:
            1. Set the CloseEvent to signal the thread to stop its operations.
            2. Wait for the thread to finish execution, allowing up to 2 seconds for it to complete.
            3. Delete the thread object to free up resources after it has stopped.

        Notes:
            - This method ensures a graceful shutdown of the thread, allowing for proper cleanup of any resources.
            - The use of the CloseEvent provides a clear mechanism for the thread to terminate its ongoing tasks safely.
        """

        self.CloseEvent.set()
        self.join(2)
        del self

    def addThreadTask(self, priority: int, task_object: _TaskObject):
        """
        Adds a task to the appropriate priority queue based on its priority level.

        :param priority: An integer representing the priority of the task (0 to 10).
        :param task_object: The task object to be added to the queue.

        steps:
            1. Check the priority level of the task:
                - If the priority is between 0 and 3 (inclusive), add the task to the HighPriorityQueue.
                - If the priority is between 4 and 7 (inclusive), add the task to the MediumPriorityQueue.
                - If the priority is between 8 and 10 (inclusive), add the task to the LowPriorityQueue.
                - If the priority is outside the range of 0 to 10, log an error indicating that the task has been rejected due to invalid priority.
            2. If the WorkingEvent is not already set, set it to indicate that there are tasks to process.

        Notes:
            - This method ensures that tasks are organized according to their priority, enabling efficient processing.
            - Proper logging is implemented to handle cases where invalid priorities are provided, aiding in debugging and monitoring.
        """

        if 0 <= priority <= 3:
            self.HighPriorityQueue.put_nowait(("HighPriority", task_object))
        elif 4 <= priority <= 7:
            self.MediumPriorityQueue.put_nowait(("MediumPriority", task_object))
        elif 8 <= priority <= 10:
            self.LowPriorityQueue.put_nowait(("LowPriority", task_object))
        else:
            self.Logger.error(f"[{self.ThreadName}] task {task_object.TaskID} has been rejected due to invalid priority {priority}.")
            return
        if not self.WorkingEvent.is_set():
            self.WorkingEvent.set()

    async def _taskProcessor(self):
        """
        Processes tasks from various priority queues until the thread is instructed to close.

        steps:
            1. Enter a loop that continues until the CloseEvent is set:
                - Try to retrieve a task from the HighPriorityQueue:
                    - If the high-priority queue is empty, check the MediumPriorityQueue next.
                    - If the medium-priority queue is also empty, check the LowPriorityQueue.
                    - If all queues are empty, sleep briefly (0.001 seconds) and continue to the next iteration.
            2. If a task is successfully retrieved, extract its priority and task object.
            3. Check if the task object requires a lock:
                - If it does, attempt to acquire the TaskLock:
                    - If the lock cannot be acquired within the specified timeout, requeue the task and continue.
                    - If the lock is acquired, create a coroutine task for executing the task object.
                    - Add the coroutine task to the PendingTasks dictionary.
                    - Await the completion of the coroutine task.
                - Ensure to release the lock in a finally block to prevent deadlocks.
            4. If the task does not require a lock, create a coroutine task for execution without locking.
            5. If any exception occurs during task processing, log the error with details.
            6. Once the loop is exited, call the cleanup method to perform any necessary cleanup actions.

        Notes:
            - This method ensures efficient processing of tasks based on their priority, while managing resource locking to avoid conflicts.
            - Error handling and logging provide visibility into any issues encountered during task execution.
            - The brief sleep during idle times helps to prevent busy-waiting and reduces CPU usage.
        """

        while not self.CloseEvent.is_set():
            try:
                if not self.HighPriorityQueue.empty():
                    task_data: Tuple[str, _TaskObject] = self.HighPriorityQueue.get_nowait()
                elif not self.MediumPriorityQueue.empty():
                    task_data: Tuple[str, _TaskObject] = self.MediumPriorityQueue.get_nowait()
                elif not self.LowPriorityQueue.empty():
                    task_data: Tuple[str, _TaskObject] = self.LowPriorityQueue.get_nowait()
                else:
                    await asyncio.sleep(0.001)
                    continue
            except queue.Empty:
                await asyncio.sleep(0.001)
                continue
            task_priority, task_object = task_data
            try:
                if task_object.Lock:
                    acquired = self.SynchronizationManager.TaskLock.acquire(timeout=task_object.LockTimeout)
                    if not acquired:
                        await self._requeueTask(task_priority, task_object)
                        continue
                    try:
                        coroutine_task = self.EventLoop.create_task(self._taskExecutor(task_object))
                        self.PendingTasks[task_object.TaskID] = coroutine_task
                        await coroutine_task
                    finally:
                        self.SynchronizationManager.TaskLock.release()
                else:
                    coroutine_task = self.EventLoop.create_task(self._taskExecutor(task_object))
                    self.PendingTasks[task_object.TaskID] = coroutine_task
            except Exception as e:
                self.Logger.error(f"[{self.ThreadName} - {self.ident}] task {task_object.TaskID} failed due to {str(e)}\n\n{traceback.format_exc()}.")
        await self._cleanup()

    async def _taskExecutor(self, task_object: Any):
        """
        Executes a given task asynchronously and processes the result.

        :param task_object: The task object to be executed.

        steps:
            1. Try to execute the task by awaiting its execute method:
                - Capture the result of the task execution.
            2. Process the task result using the _taskResultProcessor method, passing the result and task object.
            3. If the WorkingEvent is set, clear it to indicate that the thread is no longer busy.
            4. If the task execution raises a CancelledError:
                - Log an error message indicating that the task has been cancelled.
                - If the WorkingEvent is set, clear it.

        Notes:
            - This method is essential for handling the execution and result processing of tasks in an asynchronous environment.
            - Proper error handling ensures that cancellations are logged, aiding in debugging and resource management.
        """

        try:
            task_result = await task_object.execute()
            await self._taskResultProcessor(task_result, task_object)
            if self.WorkingEvent.is_set():
                self.WorkingEvent.clear()
        except asyncio.CancelledError:
            self.Logger.error(f"[{self.ThreadName} - {self.ident}] task {task_object.TaskID} has been cancelled.")
            if self.WorkingEvent.is_set():
                self.WorkingEvent.clear()

    async def _taskResultProcessor(self, task_result: Any, task_object: type(_TaskObject)):
        """
        Processes the result of a completed task and manages the corresponding task object.

        This asynchronous method stores the task result in the result storage queue and cleans up
        the pending tasks list by removing the completed task.

        :param task_result: The result obtained from executing the task.
        :param task_object: The object representing the task that has been completed.

        steps:
            1. Store the task result and its associated task ID in the ResultStorageQueue for further processing.
            2. Attempt to remove the task object from the PendingTasks dictionary using its TaskID:
                - If successful, the task is removed from the pending list.
                - If a KeyError occurs (indicating the task ID is not found), pass without taking action.
                - If any other exception occurs, log an error message indicating the failure to remove the task.

        Notes:
            - This method is critical for managing the lifecycle of tasks and their results within the system.
            - Proper error handling ensures that issues in removing tasks do not disrupt the overall task processing flow.
            - The use of asynchronous execution allows for efficient handling of task results without blocking other operations.
        """

        self.SynchronizationManager.ResultStorageQueue.put_nowait((task_result, task_object.TaskID))
        # noinspection PyBroadException
        try:
            del self.PendingTasks[task_object.TaskID]
        except KeyError:
            pass
        except Exception:
            self.Logger.error(f"[{self.ThreadName} - {self.ident}] failed to remove task {task_object.TaskID} from PendingTasks.")

    async def _requeueTask(self, task_priority: str, task_object: _TaskObject):
        """
        Requeues a task after a specified delay based on its priority and maximum retry attempts.

        :param task_priority: The priority of the task to be requeued (HighPriority, MediumPriority, LowPriority).
        :param task_object: The task object to be requeued.

        steps:
            1. Calculate the delay before requeuing the task:
                - Use an exponential backoff strategy to determine the delay, capping it at 10 seconds.
            2. Wait for the calculated delay using asyncio.sleep.
            3. Based on the task's priority:
                - If the priority is "HighPriority", add the task to the HighPriorityQueue.
                - If the priority is "MediumPriority", add the task to the MediumPriorityQueue.
                - If the priority is "LowPriority", add the task to the LowPriorityQueue.

        Notes:
            - This method is designed to handle task retries efficiently, allowing for gradual backoff to avoid overwhelming the system.
            - Using priority queues ensures that higher priority tasks are handled before lower priority ones.
        """

        await asyncio.sleep(min(0.1 * (2 ** task_object.MaxRetries), 10))
        if task_priority == "HighPriority":
            self.HighPriorityQueue.put_nowait((task_priority, task_object))
            return
        if task_priority == "MediumPriority":
            self.MediumPriorityQueue.put_nowait((task_priority, task_object))
            return
        if task_priority == "LowPriority":
            self.LowPriorityQueue.put_nowait((task_priority, task_object))
            return

    async def _cleanup(self):
        """
        Asynchronously cleans up remaining tasks from the priority queues and cancels pending tasks.

        steps:
            1. Initialize a counter (remaining_tasks) to track the number of discarded tasks.
            2. Check and discard tasks from the HighPriorityQueue:
                - While the queue is not empty, attempt to retrieve tasks without blocking.
                - Increment the remaining_tasks counter for each task retrieved.
            3. Repeat the above step for the MediumPriorityQueue and LowPriorityQueue.
            4. For each coroutine task in the PendingTasks:
                - Cancel the task to stop its execution.
            5. Log the number of discarded tasks for debugging purposes.
            6. Clear the PendingTasks dictionary to remove references to canceled tasks.

        Notes:
            - This method ensures that any tasks that are no longer needed are properly discarded and cleaned up.
            - Cancelling pending tasks helps to free up resources and avoid potential memory leaks.
            - Logging provides visibility into the cleanup operation and the number of tasks affected.
        """

        remaining_tasks = 0
        while not self.HighPriorityQueue.empty():
            try:
                _, task_object = self.HighPriorityQueue.get_nowait()
                remaining_tasks += 1
            except queue.Empty:
                break
        while not self.MediumPriorityQueue.empty():
            try:
                _, task_object = self.MediumPriorityQueue.get_nowait()
                remaining_tasks += 1
            except queue.Empty:
                break
        while not self.LowPriorityQueue.empty():
            try:
                _, task_object = self.LowPriorityQueue.get_nowait()
                remaining_tasks += 1
            except queue.Empty:
                break
        for i, coroutine_task in self.PendingTasks.items():
            coroutine_task.cancel()
        self.Logger.debug(f"[{self.ThreadName} - {self.ident}] discarded {remaining_tasks} tasks.")
        self.PendingTasks.clear()


class _LoadBalancer(threading.Thread):
    """
    Balances the load between processes and threads in the concurrent system.

    This class monitors the status of processes and threads, expanding or shrinking them based on the current workload and configured policies.

    Attributes:
        SynchronizationManager: Manages synchronization across processes and threads.
        ConfigManager: Manages system configuration settings.
        Logger: Logger for the load balancer.
        DebugMode: Flag indicating whether debug mode is active.
        CloseEvent: Event to signal the load balancer to stop.

    Methods:
        run: Continuously checks the load and manages process and thread expansion/shrinkage.
        stop: Signals the load balancer to stop and waits for it to finish.
        _cleanupMainProcessMemory: Cleans up memory for the main process.
        _cleanupServiceProcessMemory: Cleans up memory for service processes.
        _updateThreadStatus: Updates the status of threads in the synchronization manager.
        _updateProcessStatus: Updates the status of processes in the synchronization manager.
        _expandPolicyExecutor: Executes the appropriate expansion policy.
        _shrinkagePolicyExecutor: Executes the appropriate shrinkage policy.
        _isExpansionAllowed: Checks if expansion is allowed based on configured limits.
        _expandProcess: Expands the number of processes based on the current load.
        _expandThread: Expands the number of threads based on the current load.
        _generateExpandID: Generates a unique ID for a new process or thread.
        _autoExpand: Automatically expands processes and threads based on average load.
        _beforehandExpand: Expands processes and threads based on pending task counts.
        _noExpand: Placeholder for no expansion policy.
        _autoShrink: Automatically shrinks idle processes and threads after a timeout.
        _timeoutShrink: Shrinks processes and threads that exceed a specified timeout.

    Notes:
        - The load balancer plays a crucial role in maintaining system efficiency by adjusting resources according to demand.
        - Proper monitoring and management of processes and threads can prevent resource waste and improve performance.
    """

    def __init__(self, SM: _SynchronizationManager, CM: _ConfigManager, Logger: logging.Logger, DebugMode: bool):
        super().__init__(name='LoadBalancer', daemon=True)
        self.SynchronizationManager = SM
        self.ConfigManager = CM
        self.Logger = Logger
        self.DebugMode = DebugMode
        self.CloseEvent = multiprocessing.Event()

    def run(self):
        """
        Executes the main loop for the load balancer.

        This method continuously manages the state and resources of the system, performing regular updates
        and cleanups as needed while the system is running.

        steps:
            1. Initialize a variable to track elapsed time (rest_times).
            2. Perform an initial cleanup of memory for the main and service processes.
            3. Enter a loop that continues until the CloseEvent is set:
                - Update the status of threads by calling `_updateThreadStatus()`.
                - Update the status of processes by calling `_updateProcessStatus()`.
                - Execute any necessary expansion policies by calling `_expandPolicyExecutor()`.
                - Execute any necessary shrinkage policies by calling `_shrinkagePolicyExecutor()`.
                - Sleep for a short duration (0.001 seconds) to yield control.
                - Increment the rest_times tracker.
                - If the elapsed time exceeds 60 seconds (60000 milliseconds):
                    - Perform a cleanup of memory for the main and service processes.
                    - Reset the rest_times tracker.

        Notes:
            - This method is essential for maintaining the health and performance of the process management system.
            - Regular updates and cleanups help to mitigate resource leaks and ensure optimal operation.
            - The loop design allows for continuous monitoring and adjustments without blocking the system.
        """

        rest_times = 0
        self._cleanupMainProcessMemory()
        self._cleanupServiceProcessMemory()
        while not self.CloseEvent.is_set():
            self._updateThreadStatus()
            self._updateProcessStatus()
            self._expandPolicyExecutor()
            self._shrinkagePolicyExecutor()
            time.sleep(0.001)
            rest_times += 0.001
            if rest_times >= 60000:
                self._cleanupMainProcessMemory()
                self._cleanupServiceProcessMemory()
                rest_times = 0

    def stop(self):
        """
        Stops the current thread by signaling it to close and waiting for it to finish.

        steps:
            1. Set the CloseEvent to signal the thread to stop its operation.
            2. Wait for the thread to finish execution, allowing up to 2 seconds for completion.
            3. Delete the thread object to free resources after it has stopped.

        Notes:
            - This method ensures a graceful shutdown of the thread, allowing for proper cleanup.
            - The use of the CloseEvent allows the thread to terminate its operations safely.
        """

        self.CloseEvent.set()
        self.join(2)
        del self

    def _cleanupMainProcessMemory(self):
        """
        Cleans up memory used by the main process based on the operating system.

        This method performs memory cleanup operations tailored for Linux, Windows, and macOS.

        steps:
            1. Determine the operating system using `platform.system()`.
            2. If the operating system is Linux:
                - Try to access the `libc` library to call `malloc_trim` to release unused memory.
                - If `malloc_trim` is not available, run shell commands to sync the filesystem and drop caches.
                - Log any errors that occur during the cleanup process.
            3. If the operating system is Windows:
                - Attempt to obtain a handle to the current process.
                - If successful, call `EmptyWorkingSet` to free memory allocated to the process.
                - Log any errors that occur during handle retrieval or memory cleanup.
            4. If the operating system is macOS:
                - Use the `purge` command to clean up memory.
                - Log any errors that occur during the cleanup process.
            5. Handle exceptions gracefully and log errors using the provided logger.

        Notes:
            - This method is essential for managing memory usage effectively, particularly in long-running processes where memory fragmentation can occur.
            - Proper error handling ensures that failures during the cleanup process are logged and do not cause crashes.
        """

        system = platform.system()
        if system == "Linux":
            try:
                libc = ctypes.CDLL("libc.so.6")
                malloc_trim = getattr(libc, "malloc_trim", None)
                if malloc_trim:
                    malloc_trim(0)
                else:
                    subprocess.run(["sync"], check=True)
                    subprocess.run(["echo", "3", "|", "sudo", "tee", "/proc/sys/vm/drop_caches"], check=True)
                return
            except Exception as e:
                self.Logger.error(f"[{self.name} - {self.ident}] memory cleanup failed on Linux: {str(e)}\n\n{traceback.format_exc()}.")
                return
        if system == "Windows":
            try:
                handle = ctypes.windll.kernel32.OpenProcess(0x1F0FFF, False, os.getpid())
                if handle == 0:
                    raise ValueError("Failed to obtain a valid handle")
            except Exception as e:
                self.Logger.error(f"[{self.name} - {self.ident}] memory cleanup failed due to {str(e)}\n\n{traceback.format_exc()}.")
                return
            if not handle:
                self.Logger.error(f"[{self.name} - {self.ident}] failed to obtain a valid process handle.")
                return
            result = ctypes.windll.psapi.EmptyWorkingSet(handle)
            if result == 0:
                error_code = ctypes.windll.kernel32.GetLastError()
                self.Logger.error(f"[{self.name} - {self.ident}] memory cleanup failed with error code {error_code}.")
            ctypes.windll.kernel32.CloseHandle(handle)
            return
        if system == "Darwin":
            try:
                subprocess.run(["sudo", "purge"], check=True)
            except subprocess.SubprocessError as e:
                self.Logger.error(f"[{self.name} - {self.ident}] memory cleanup failed on macOS: {str(e)}")
            except Exception as e:
                self.Logger.error(f"[{self.name} - {self.ident}] unexpected error during memory cleanup on macOS: {str(e)}\n\n{traceback.format_exc()}.")
            finally:
                return

    def _cleanupServiceProcessMemory(self):
        """
        Cleans up memory used by the service process on Windows.

        This method attempts to free memory allocated to the service process by calling
        the `EmptyWorkingSet` function on the process handle associated with the service.

        steps:
            1. Check if the operating system is Windows.
            2. If the operating system is Windows:
                - Attempt to obtain a handle to the service process using `SynchronizationManager.SharedObjectManagerID`.
                - If the handle is not valid, raise a ValueError indicating failure to obtain a valid handle.
                - If successful, call `EmptyWorkingSet` to free memory allocated to the process.
                - Log any errors that occur during handle retrieval or memory cleanup.
                - If `EmptyWorkingSet` fails, log the error code obtained from `GetLastError`.
            3. Close the handle to free the resource.

        Notes:
            - This method is crucial for managing memory effectively in the service process, helping to mitigate memory leaks.
            - Proper error handling ensures that failures during the cleanup process are logged and do not cause crashes.
        """

        system = platform.system()
        if system == "Windows":
            try:
                handle = ctypes.windll.kernel32.OpenProcess(0x1F0FFF, False, self.SynchronizationManager.SharedObjectManagerID)
                if handle == 0:
                    raise ValueError("Failed to obtain a valid handle")
            except Exception as e:
                self.Logger.error(f"[{self.name} - {self.ident}] memory cleanup failed due to {str(e)}\n\n{traceback.format_exc()}.")
                return
            if not handle:
                self.Logger.error(f"[{self.name} - {self.ident}] failed to obtain a valid process handle.")
                return
            result = ctypes.windll.psapi.EmptyWorkingSet(handle)
            if result == 0:
                error_code = ctypes.windll.kernel32.GetLastError()
                self.Logger.error(f"[{self.name} - {self.ident}] memory cleanup failed with error code {error_code}.")
            ctypes.windll.kernel32.CloseHandle(handle)
            return

    def _updateThreadStatus(self):
        """
        Updates the status of all core and expanded threads in the synchronization manager.

        steps:
            1. Iterate over each thread in the core thread pool:
                - Update the core thread status pool with the thread's identifier and the count of its pending tasks.
            2. If the expanded thread pool is not empty, iterate over each thread in the expanded thread pool:
                - Update the expanded thread status pool with the thread's identifier and the count of its pending tasks.

        Notes:
            - This method is essential for maintaining accurate tracking of thread statuses and their workloads.
            - Keeping the status updated allows for better resource management and load balancing within the system.
        """

        global _CoreThreadPool, _ExpandThreadPool
        for thread_name, thread_obj in _CoreThreadPool.items():
            self.SynchronizationManager.CoreThreadStatusPool[thread_name] = (thread_obj.ident, len(thread_obj.PendingTasks))
        if _ExpandThreadPool:
            for thread_name, thread_obj in _ExpandThreadPool.items():
                self.SynchronizationManager.ExpandThreadStatusPool[thread_name] = (thread_obj.ident, len(thread_obj.PendingTasks))

    def _updateProcessStatus(self):
        """
        Updates the status of core and expand processes in the system.

        This method retrieves the current CPU and memory usage for each process in both the
        CoreProcessPool and ExpandProcessPool, and updates the corresponding status pools.

        steps:
            1. Iterate through each process in the CoreProcessPool:
                - Calculate the total task count by summing the sizes of high, medium, and low priority queues,
                  and the length of pending tasks.
                - Use the _Monitor to get the CPU and memory usage for the current process.
                - Calculate a weighted load based on the CPU and memory usage, giving equal weight to both.
                - Update the CoreProcessStatusPool with the process ID, task count, and calculated weighted load.
                - If an error occurs during this process, set the weighted load to 0 for that process in the status pool.
            2. If there are any processes in the ExpandProcessPool:
                - Repeat the same steps as above to update the ExpandProcessStatusPool.

        Notes:
            - This method is crucial for monitoring the performance and load of each process within the system.
            - Proper error handling ensures that any issues in retrieving process information do not affect the entire status update.
        """

        global _CoreProcessPool, _ExpandProcessPool
        for process_name, process_obj in _CoreProcessPool.items():
            task_count = process_obj.HighPriorityQueue.qsize() + process_obj.MediumPriorityQueue.qsize() + process_obj.LowPriorityQueue.qsize() + len(process_obj.PendingTasks)
            # noinspection PyBroadException
            try:
                process_cpu_usage = _Monitor.processCpuUsage(process_obj.pid, 0.001)
                process_memory_usage = _Monitor.processMemoryUsage(process_obj.pid)
                weighted_load = int((process_cpu_usage * 0.5) + (process_memory_usage * 0.5))
                self.SynchronizationManager.CoreProcessStatusPool[process_name] = (process_obj.pid, task_count, min(max(weighted_load, 0), 100))
            except Exception:
                self.SynchronizationManager.CoreProcessStatusPool[process_name] = (process_obj.pid, task_count, 0)
        if _ExpandProcessPool:
            for process_name, process_obj in _ExpandProcessPool.items():
                task_count = process_obj.HighPriorityQueue.qsize() + process_obj.MediumPriorityQueue.qsize() + process_obj.LowPriorityQueue.qsize() + len(process_obj.PendingTasks)
                # noinspection PyBroadException
                try:
                    process_cpu_usage = _Monitor.processCpuUsage(process_obj.pid, 0.001)
                    process_memory_usage = _Monitor.processMemoryUsage(process_obj.pid)
                    weighted_load = int((process_cpu_usage * 0.5) + (process_memory_usage * 0.5))
                    self.SynchronizationManager.ExpandProcessStatusPool[process_name] = (process_obj.pid, task_count, min(max(weighted_load, 0), 100))
                except Exception:
                    self.SynchronizationManager.ExpandProcessStatusPool[process_name] = (process_obj.pid, task_count, 0)

    def _expandPolicyExecutor(self):
        """
        Executes the configured expansion policy method to manage resource allocation.

        steps:
            1. Define a dictionary mapping expansion policy names to their corresponding methods.
            2. Retrieve the expansion policy method based on the current configuration.
            3. Call the selected expansion method to perform the necessary actions.

        Notes:
            - This method allows for flexible execution of different expansion strategies based on the system's configuration.
            - The available policies include "NoExpand", "AutoExpand", and "BeforehandExpand".
        """

        policy_method = {
            "NoExpand": self._noExpand,
            "AutoExpand": self._autoExpand,
            "BeforehandExpand": self._beforehandExpand,
        }
        expand_method = policy_method[self.ConfigManager.ExpandPolicy.value]
        expand_method()

    def _noExpand(self):
        """No expansion policy method."""
        pass

    def _autoExpand(self):
        """
        Automatically expands the process and thread pools based on their average load.

        steps:
            1. If the core process count is greater than zero:
                - Calculate the total current load of core processes by summing their load values from the status pool.
                - Compute the average load per core process. If there are no core processes, set the average to zero.
                - Determine the ideal load per process as 80% of the total allowable load divided by the total number of processes (core and expanded).
                - If the average process load exceeds the ideal load:
                    - Check if expansion of processes is allowed using the isExpansionAllowed method:
                        - If allowed, call the expandProcess method.
                        - If not allowed, log a debug message indicating the inability to expand processes despite high load.
            2. Calculate the total current load of core threads by summing their load values from the status pool.
            3. Compute the average load per core thread. If there are no core threads, set the average to zero.
            4. Define the ideal load per thread as 80% of the total allowable load.
            5. If the average thread load exceeds the ideal load:
                - Check if expansion of threads is allowed:
                    - If allowed, call the expandThread method.
                    - If not allowed, log a debug message indicating the inability to expand threads despite high load.

        Notes:
            - This method is crucial for dynamically adjusting system capacity based on current workloads to maintain performance.
            - It ensures that resources are efficiently utilized and prevents overload scenarios by expanding when needed.
        """

        if self.ConfigManager.CoreProcessCount.value != 0:
            current_core_process_total_load = sum([load for _, _, load in self.SynchronizationManager.CoreProcessStatusPool.values()])
            average_process_load = current_core_process_total_load / len(self.SynchronizationManager.CoreProcessStatusPool) if len(self.SynchronizationManager.CoreProcessStatusPool) > 0 else 0
            ideal_load_per_process = 100 * 0.8 / (len(self.SynchronizationManager.CoreProcessStatusPool) + len(self.SynchronizationManager.ExpandProcessStatusPool))
            if average_process_load > ideal_load_per_process:
                if self._isExpansionAllowed("Process"):
                    self._expandProcess()
                else:
                    self.Logger.debug(f"Load reaches {int(ideal_load_per_process)}%, but unable to expand more process")

        current_core_thread_total_load = sum([load for _, load in self.SynchronizationManager.CoreThreadStatusPool.values()])
        average_thread_load = current_core_thread_total_load / len(self.SynchronizationManager.CoreThreadStatusPool) if len(self.SynchronizationManager.CoreThreadStatusPool) > 0 else 0
        ideal_load_per_thread = 100 * 0.8
        if average_thread_load > ideal_load_per_thread:
            if self._isExpansionAllowed("Thread"):
                self._expandThread()
            else:
                self.Logger.debug(f"Load reaches {int(ideal_load_per_thread)}%, but unable to expand more thread")

    def _beforehandExpand(self):
        """
        Checks the pending task counts and expands the process and thread pools if necessary.

        steps:
            1. If the core process count is greater than zero:
                - Calculate the total number of pending tasks for core processes by summing the task counts from the status pool.
            2. If there are no core processes, set the pending task count for core processes to zero.
            3. Calculate the total number of pending tasks for core threads by summing the task counts from the status pool.
            4. Check if the combined pending task count of core processes and threads exceeds 80% of the global task threshold:
                - If it does, check if expansion of processes is allowed using the isExpansionAllowed method:
                    - If allowed, call the expandProcess method.
                    - If not allowed, log a debug message indicating the inability to expand processes due to the pending task count.
                - Similarly, check if expansion of threads is allowed:
                    - If allowed, call the expandThread method.
                    - If not allowed, log a debug message indicating the inability to expand threads due to the pending task count.

        Notes:
            - This method is important for proactive resource management to handle an increasing load by expanding the system's capacity.
            - It ensures that the system operates efficiently by monitoring task thresholds and taking appropriate actions.
        """

        if self.ConfigManager.CoreProcessCount.value != 0:
            core_process_pending_task_count = sum([task_count for _, task_count, _ in self.SynchronizationManager.CoreProcessStatusPool.values()])
        else:
            core_process_pending_task_count = 0
        core_thread_pending_task_count = sum([task_count for _, task_count in self.SynchronizationManager.CoreThreadStatusPool.values()])
        if (core_process_pending_task_count + core_thread_pending_task_count) >= self.ConfigManager.GlobalTaskThreshold.value * 0.8:
            if self._isExpansionAllowed("Process"):
                self._expandProcess()
            else:
                self.Logger.debug(f"Pending task count reaches {self.ConfigManager.GlobalTaskThreshold.value}, but unable to expand more process")
            if self._isExpansionAllowed("Thread"):
                self._expandThread()
            else:
                self.Logger.debug(f"Pending task count reaches {self.ConfigManager.GlobalTaskThreshold.value}, but unable to expand more thread")

    def _isExpansionAllowed(self, expand_type: Literal["Process", "Thread"]) -> bool:
        """
        Checks if expansion of the specified type (process or thread) is allowed based on the configured limits.

        :param expand_type: The type of expansion to check, either "Process" or "Thread".

        :return: A boolean indicating whether expansion is allowed.

        steps:
            1. Use a match-case statement to determine the type of expansion.
            2. For "Process":
                - Calculate the total number of core and expanded processes.
                - Compare against the configured maximum process count.
                - Return False if the total exceeds the limit; otherwise, return True.
            3. For "Thread":
                - Calculate the total number of core and expanded threads.
                - Compare against the configured maximum thread count.
                - Return False if the total exceeds the limit; otherwise, return True.

        Notes:
            - This method is crucial for managing resources and ensuring that the system does not exceed configured limits for processes and threads.
            - It helps maintain stability and performance by preventing over-allocation.
        """

        match expand_type:
            case "Process":
                if (len(self.SynchronizationManager.CoreProcessStatusPool) + len(self.SynchronizationManager.ExpandProcessStatusPool)) >= self.ConfigManager.MaximumProcessCount.value:
                    return False
                return True
            case "Thread":
                if (len(self.SynchronizationManager.CoreThreadStatusPool) + len(self.SynchronizationManager.ExpandThreadStatusPool)) >= self.ConfigManager.MaximumThreadCount.value:
                    return False
                return True

    def _expandProcess(self):
        """
        Expands the process pool by creating and starting a new process.

        steps:
            1. Retrieve the current time to track the process's survival.
            2. Generate a unique identifier for the new process using the expansion ID generator.
            3. Construct a process name based on the generated ID.
            4. Create a new process object, initializing it with the necessary parameters.
            5. Start the newly created process.
            6. Add the process object to the expand process pool using the generated ID.
            7. Update the synchronization manager's status pool with the new process's name and initial status.
            8. Record the survival time of the new process.

        Notes:
            - This method allows for dynamic expansion of the process pool to handle additional workloads.
            - The survival time tracking enables management and potential shrinking of idle processes later.
        """

        global _ExpandProcessPool, _ExpandProcessSurvivalTime
        current_time = time.time()
        process_id = self._generateExpandID("Process")
        process_name = f"Process-{process_id}"
        process_object = _ProcessObject(process_name, "Expand", self.SynchronizationManager, self.ConfigManager, self.DebugMode)
        process_object.start()
        _ExpandProcessPool[process_id] = process_object
        self.SynchronizationManager.ExpandProcessStatusPool[process_name] = (process_object.pid, 0, 0)
        _ExpandProcessSurvivalTime[process_name] = current_time

    def _expandThread(self):
        """
        Expands the thread pool by creating and starting a new thread.

        steps:
            1. Retrieve the current time to track the thread's survival.
            2. Generate a unique identifier for the new thread using the expansion ID generator.
            3. Construct a thread name based on the generated ID.
            4. Create a new thread object, initializing it with the necessary parameters.
            5. Start the newly created thread.
            6. Add the thread object to the expand thread pool using the generated ID.
            7. Update the synchronization manager's status pool with the new thread's name and initial status.
            8. Record the survival time of the new thread.

        Notes:
            - This method enables dynamic expansion of the thread pool to accommodate additional workloads.
            - The survival time tracking allows for management and potential shrinking of idle threads later.
        """

        global _ExpandThreadPool, _ExpandThreadSurvivalTime
        current_time = time.time()
        thread_id = self._generateExpandID("Thread")
        thread_name = f"Thread-{thread_id}"
        thread_object = _ThreadObject(thread_name, "Expand", self.SynchronizationManager, self.ConfigManager, self.Logger)
        thread_object.start()
        _ExpandThreadPool[thread_id] = thread_object
        self.SynchronizationManager.ExpandThreadStatusPool[thread_name] = (thread_object.ident, 0)
        _ExpandThreadSurvivalTime[thread_name] = current_time

    @staticmethod
    def _generateExpandID(expand_type: Literal["Process", "Thread"]):
        """
        Generates a unique identifier for expanded processes or threads.

        :param expand_type: The type of expansion for which to generate an ID, either "Process" or "Thread".

        :return: A unique identifier for the specified expansion type.

        steps:
            1. Define a mapping of expansion types to their corresponding pools (process or thread).
            2. Initialize a basic ID starting with the format "<expand_type>-0".
            3. Enter an infinite loop to find a unique ID:
                - Check if the current basic ID already exists in the corresponding pool.
                - If it does not exist, return the current basic ID.
                - If it does exist, increment the numeric suffix of the ID and repeat.

        Notes:
            - This method ensures that the generated ID is unique within the specified expansion type's pool.
            - The ID format helps in easily identifying the type and instance of the expanded resource.
        """

        global _ExpandProcessPool, _ExpandThreadPool
        expand_type_mapping = {
            "Process": _ExpandProcessPool,
            "Thread": _ExpandThreadPool,
        }
        basic_id = f"{expand_type}-{0}"
        while True:
            if basic_id not in expand_type_mapping[expand_type]:
                return basic_id
            basic_id = f"{expand_type}-{int(basic_id.split('-')[-1]) + 1}"

    def _shrinkagePolicyExecutor(self):
        """
        Executes the configured shrinkage policy method to manage resource allocation.

        steps:
            1. Define a dictionary mapping shrinkage policy names to their corresponding methods.
            2. Retrieve the shrinkage policy method based on the current configuration.
            3. Call the selected shrinkage method to perform the necessary actions.

        Notes:
            - This method allows for flexible execution of different shrinkage strategies based on the system's configuration.
            - The available policies include "NoShrink", "AutoShrink", and "TimeoutShrink".
        """

        policy_method = {
            "NoShrink": self._noShrink,
            "AutoShrink": self._autoShrink,
            "TimeoutShrink": self._timeoutShrink,
        }
        shrink_method = policy_method[self.ConfigManager.ShrinkagePolicy.value]
        shrink_method()

    def _noShrink(self):
        """No shrinkage policy method."""
        pass

    def _autoShrink(self):
        """
        Automatically shrinks the pool of expanded processes and threads by closing those that are idle.

        steps:
            1. If the core process count is greater than zero:
                - Retrieve a list of all expanded processes.
                - Identify idle processes that have not performed any tasks (status indicates zero workload).
                - Determine which idle processes can be closed based on their survival time exceeding the configured shrinkage timeout.
                - For each idle process that qualifies for closure:
                    - Stop the process and remove it from the expand process pool.
                    - Remove the corresponding entry from the expand process status pool and survival time tracking.
                    - Log the closure of the process due to idle status.
            2. Retrieve a list of all expanded threads.
            3. Identify idle threads that have not performed any tasks.
            4. Determine which idle threads can be closed based on their survival time exceeding the configured shrinkage timeout.
            5. For each idle thread that qualifies for closure:
                - Stop the thread and remove it from the expand thread pool.
                - Remove the corresponding entry from the expand thread status pool and survival time tracking.
                - Log the closure of the thread due to idle status.

        Notes:
            - This method helps manage resources effectively by closing idle processes and threads, thereby freeing up system resources.
            - Logging is utilized to track which processes and threads are closed due to being idle.
        """

        global _ExpandProcessPool, _ExpandThreadPool, _ExpandProcessSurvivalTime, _ExpandThreadSurvivalTime
        if self.ConfigManager.CoreProcessCount.value != 0:
            expand_process_obj = [obj for i, obj in _ExpandProcessPool.items()]
            idle_process = [obj for obj in expand_process_obj if self.SynchronizationManager.ExpandProcessStatusPool[obj.ProcessName][1] == 0]
            allow_close_process = [obj for obj in idle_process if (time.time() - _ExpandProcessSurvivalTime[obj.ProcessName]) >= self.ConfigManager.ShrinkagePolicyTimeout.value]
            if idle_process:
                for obj in idle_process:
                    if obj in allow_close_process:
                        obj.stop()
                        del _ExpandProcessPool[obj.ProcessName]
                        del self.SynchronizationManager.ExpandProcessStatusPool[obj.ProcessName]
                        del _ExpandProcessSurvivalTime[obj.ProcessName]
                        self.Logger.debug(f"Process {obj.ProcessName} has been closed due to idle status.")

        expand_thread_obj = [obj for i, obj in _ExpandThreadPool.items()]
        idle_thread = [obj for obj in expand_thread_obj if self.SynchronizationManager.ExpandThreadStatusPool[obj.ThreadName][1] == 0]
        allow_close_thread = [obj for obj in idle_thread if (time.time() - _ExpandThreadSurvivalTime[obj.ThreadName]) >= self.ConfigManager.ShrinkagePolicyTimeout.value]
        if idle_thread:
            for obj in idle_thread:
                if obj in allow_close_thread:
                    obj.stop()
                    del _ExpandThreadPool[obj.ThreadName]
                    del self.SynchronizationManager.ExpandThreadStatusPool[obj.ThreadName]
                    del _ExpandThreadSurvivalTime[obj.ThreadName]
                    self.Logger.debug(f"Thread {obj.ThreadName} has been closed due to idle status.")

    def _timeoutShrink(self):
        """
        Checks for and closes any expanded processes or threads that have exceeded their survival timeout.

        steps:
            1. If the core process count is greater than zero:
                - Identify expanded processes that have exceeded their survival time based on the configured timeout.
                - For each expanded process that has timed out:
                    - Stop the process and remove it from the expand process pool.
                    - Remove the corresponding entry from the expand process status pool and survival time tracking.
                    - Log the closure of the process due to timeout.
            2. Identify expanded threads that have exceeded their survival time based on the configured timeout.
            3. For each expanded thread that has timed out:
                - Stop the thread and remove it from the expand thread pool.
                - Remove the corresponding entry from the expand thread status pool and survival time tracking.
                - Log the closure of the thread due to timeout.

        Notes:
            - This method is used to ensure that resources are freed by closing processes and threads that are no longer needed.
            - Logging is used to keep track of which processes and threads are closed due to timeout.
        """

        global _ExpandProcessPool, _ExpandThreadPool, _ExpandProcessSurvivalTime, _ExpandThreadSurvivalTime
        if self.ConfigManager.CoreProcessCount.value != 0:
            expand_process_obj = [obj for obj, survival_time in _ExpandProcessSurvivalTime.items() if (time.time() - survival_time) >= self.ConfigManager.ShrinkagePolicyTimeout.value]
            if expand_process_obj:
                for obj in expand_process_obj:
                    _ExpandProcessPool[obj].stop()
                    del _ExpandProcessPool[obj]
                    del self.SynchronizationManager.ExpandProcessStatusPool[obj]
                    del _ExpandProcessSurvivalTime[obj]
                    self.Logger.debug(f"Process {obj} has been closed due to timeout.")

        expand_thread_obj = [obj for obj, survival_time in _ExpandThreadSurvivalTime.items() if (time.time() - survival_time) >= self.ConfigManager.ShrinkagePolicyTimeout.value]
        if expand_thread_obj:
            for obj in expand_thread_obj:
                _ExpandThreadPool[obj].stop()
                del _ExpandThreadPool[obj]
                del self.SynchronizationManager.ExpandThreadStatusPool[obj]
                del _ExpandThreadSurvivalTime[obj]
                self.Logger.debug(f"Thread {obj} has been closed due to timeout.")


class _ProcessTaskScheduler(threading.Thread):
    """
    Schedules and manages process tasks for concurrent execution.

    This class runs in a separate thread and pulls tasks from a multiprocessing queue for execution, distributing them to available processes based on load and priority.

    Attributes:
        SynchronizationManager: Manages synchronization across processes.
        ConfigManager: Manages system configuration settings.
        ProcessTaskStorageQueue: Queue for storing process tasks.
        Logger: Logger for the process task scheduler.
        LastSelectedProcess: The last process that was assigned a task.
        CloseEvent: Event to signal the scheduler to stop.
        NewTaskEvent: Event to indicate that new tasks are available.
        sleep_time: Time to sleep when no tasks are available.

    Methods:
        run: Continuously checks for new tasks and schedules them.
        stop: Signals the scheduler to stop and waits for it to finish.
        _scheduler: Distributes tasks to available processes based on priority and load.
        _checkNotFullProcess: Checks for processes that are not full and can take new tasks.
        _checkNotWorkingProcess: Checks for processes that are currently not working.
        _checkMinimumLoadProcess: Finds the process with the least load among the available processes.

    Notes:
        - The scheduler operates in a loop, processing tasks as they arrive.
        - Proper management of process workload is crucial for system efficiency.
    """

    def __init__(self, SM: _SynchronizationManager, CM: _ConfigManager, ProcessTaskStorageQueue: multiprocessing.Queue, Logger: logging.Logger):
        super().__init__(name='ProcessTaskScheduler', daemon=True)
        self.SynchronizationManager = SM
        self.ConfigManager = CM
        self.ProcessTaskStorageQueue = ProcessTaskStorageQueue
        self.Logger = Logger
        self.LastSelectedProcess = None
        self.CloseEvent = multiprocessing.Event()
        self.NewTaskEvent = threading.Event()
        self.sleep_time = 0.001

    def run(self):
        """
        Runs the process task scheduler thread, continuously processing tasks from the task storage queue until instructed to stop.

        steps:
            1. Enter a loop that continues until the CloseEvent is set.
            2. Initialize task_data to None for task retrieval.
            3. Attempt to retrieve a task from the ThreadTaskStorageQueue without blocking:
                - If the queue is empty:
                    - If there are no new tasks, sleep for a specified duration.
                    - Wait for the NewTaskEvent to be set, with a timeout of 0.1 seconds.
                    - Clear the NewTaskEvent flag after processing.
            4. If a task was successfully retrieved:
                - Extract the priority and task object from the task data.
                - Call the scheduler to process the task with the given priority.

        Notes:
            - This method ensures efficient processing of tasks while managing thread sleep and wake states based on task availability.
            - The use of events and queue handling allows for responsive task management.
        """

        while not self.CloseEvent.is_set():
            task_data = None
            try:
                task_data = self.ProcessTaskStorageQueue.get_nowait()
            except queue.Empty:
                if not self.NewTaskEvent.is_set():
                    time.sleep(self.sleep_time)
                self.NewTaskEvent.wait(timeout=0.1)
                self.NewTaskEvent.clear()
            if task_data:
                priority, task_object = task_data
                self._scheduler(priority, task_object)

    def stop(self):
        """
        Stops the current thread by signaling it to close and waiting for it to finish.

        steps:
            1. Set the CloseEvent to signal the thread to stop.
            2. Set the NewTaskEvent to ensure that any waiting tasks are released.
            3. Wait for the thread to finish execution, allowing up to 2 seconds for completion.
            4. Delete the thread object to free resources.

        Notes:
            - This method ensures a graceful shutdown of the thread.
            - Proper signaling is used to notify the thread to cease operations.
        """

        self.CloseEvent.set()
        self.NewTaskEvent.set()
        self.join(2)
        del self

    def _scheduler(self, priority: int, task_object: _TaskObject):
        """
        Schedules a process task by adding it to an appropriate process based on their availability and load.

        :param priority: The priority of the task to be scheduled.
        :param task_object: The task object representing the task to be executed.

        steps:
            1. Check for non-working core processes.
            2. Check for non-full core processes.
            3. If there are non-working core processes:
                - Select the core process with the minimum load and assign the task to it.
                - Set the WorkingEvent to indicate the process is now working.
                - Return from the method.
            4. If there are no non-working processes but there are non-full core processes:
                - Select the core process with the minimum load and assign the task to it.
                - Set the WorkingEvent to indicate the process is now working.
                - Return from the method.
            5. If the core process pool is not empty, check for non-working and non-full expand processes.
            6. If there are non-working expand processes:
                - Select the expand process with the minimum load and assign the task to it.
                - Set the WorkingEvent to indicate the process is now working.
                - Return from the method.
            7. If there are no non-working processes but there are non-full expand processes:
                - Select the expand process with the minimum load and assign the task to it.
                - Set the WorkingEvent to indicate the process is now working.
                - Return from the method.
            8. If no suitable processes are found, put the task in the process task storage queue.

        Notes:
            - This method prioritizes assigning tasks to processes that are either not working or not full to maintain optimal system performance.
            - The task will be queued if no processes are available to handle it immediately.
        """

        global _ExpandProcessPool
        not_working_core_processes = self._checkNotWorkingProcess("Core")
        not_full_core_processes = self._checkNotFullProcess("Core")

        if not_working_core_processes:
            minimum_load_core_process = self._checkMinimumLoadProcess(not_working_core_processes, "Core")
            minimum_load_core_process.addProcessTask(priority, task_object)
            minimum_load_core_process.WorkingEvent.set()
            return
        if not_full_core_processes:
            minimum_load_core_process = self._checkMinimumLoadProcess(not_full_core_processes, "Core")
            minimum_load_core_process.addProcessTask(priority, task_object)
            minimum_load_core_process.WorkingEvent.set()
            return
        if _ExpandProcessPool:
            not_working_expand_processes = self._checkNotWorkingProcess("Expand")
            not_full_expand_processes = self._checkNotFullProcess("Expand")

            if not_working_expand_processes:
                minimum_load_expand_process = self._checkMinimumLoadProcess(not_working_expand_processes, "Expand")
                minimum_load_expand_process.addProcessTask(priority, task_object)
                minimum_load_expand_process.WorkingEvent.set()
                return
            if not_full_expand_processes:
                minimum_load_expand_process = self._checkMinimumLoadProcess(not_full_expand_processes, "Expand")
                minimum_load_expand_process.addProcessTask(priority, task_object)
                minimum_load_expand_process.WorkingEvent.set()
                return
            self.ProcessTaskStorageQueue.put_nowait((priority, task_object))
            return
        self.ProcessTaskStorageQueue.put_nowait((priority, task_object))

    def _checkNotFullProcess(self, process_type: Literal["Core", "Expand"]) -> list:
        """
        Checks for processes that are not currently full based on their load.

        :param process_type: The type of processes to check, either "Core" or "Expand".

        :return: A list of process objects that are not full.

        steps:
            1. Determine the appropriate process pool to use based on the process type.
            2. Determine the corresponding status pool for monitoring process loads.
            3. Iterate through the selected process pool and collect processes where the current load is less than the task threshold.
            4. Return the list of processes that are not full.

        Notes:
            - This method is useful for load balancing by identifying processes that can accept more tasks.
            - The task threshold is used to determine when a process is considered full.
        """

        global _CoreProcessPool, _ExpandProcessPool
        obj_pool = _CoreProcessPool if process_type == "Core" else _ExpandProcessPool
        status_pool = self.SynchronizationManager.CoreProcessStatusPool if process_type == "Core" else self.SynchronizationManager.ExpandProcessStatusPool
        not_full_processes = [obj for index, obj in obj_pool.items() if status_pool[obj.ProcessName][1] < self.ConfigManager.TaskThreshold.value]
        return not_full_processes

    @staticmethod
    def _checkNotWorkingProcess(process_type: Literal["Core", "Expand"]) -> list:
        """
        Checks for processes that are not currently working.

        :param process_type: The type of processes to check, either "Core" or "Expand".

        :return: A list of process objects that are not currently working.

        steps:
            1. Determine the appropriate process pool to use based on the process type.
            2. Iterate through the selected process pool and collect processes where the WorkingEvent is not set.
            3. Return the list of non-working processes.

        Notes:
            - This method helps in identifying processes that may need attention or restarting.
            - The WorkingEvent is used as an indicator of whether a process is currently active.
        """

        global _CoreProcessPool, _ExpandProcessPool
        obj_pool = _CoreProcessPool if process_type == "Core" else _ExpandProcessPool
        not_working_processes = [obj for index, obj in obj_pool.items() if not obj.WorkingEvent.is_set()]
        return not_working_processes

    def _checkMinimumLoadProcess(self, processes: list, process_type: Literal["Core", "Expand"]) -> _ProcessObject:
        """
        Checks and selects a process with the minimum load from the provided list of processes.

        :param processes: A list of process objects to evaluate.
        :param process_type: The type of processes to check, either "Core" or "Expand".

        :return: The selected process object with the minimum load.

        steps:
            1. Determine the status pool to use based on the process type.
            2. Create a list of available processes, excluding the last selected process if it exists.
            3. If there are no available processes, select the first process from the original list.
            4. If there are available processes, select the one with the minimum load from the status pool.
            5. Update the last selected process reference.

        Notes:
            - This method is used to ensure that process selection is balanced based on load.
            - The minimum load is determined using the third value in the process's status, which represents the load.
        """

        status_pool = self.SynchronizationManager.CoreProcessStatusPool if process_type == "Core" else self.SynchronizationManager.ExpandProcessStatusPool
        available_processes = [p for p in processes if p != self.LastSelectedProcess] if self.LastSelectedProcess is not None else processes
        if not available_processes:
            selected_process = processes[0]
        else:
            selected_process = min(available_processes, key=lambda x: status_pool[x.ProcessName][2])
        self.LastSelectedProcess = selected_process
        return selected_process


class _ThreadTaskScheduler(threading.Thread):
    """
    Schedules and manages thread tasks for concurrent execution.

    This class runs in a separate thread and pulls tasks from a queue for execution, distributing them to available worker threads based on load and priority.

    Attributes:
        SynchronizationManager: Manages synchronization across threads.
        ConfigManager: Manages system configuration settings.
        ThreadTaskStorageQueue: Queue for storing thread tasks.
        Logger: Logger for the thread task scheduler.
        LastSelectedThread: The last thread that was assigned a task.
        CloseEvent: Event to signal the scheduler to stop.
        NewTaskEvent: Event to indicate that new tasks are available.
        sleep_time: Time to sleep when no tasks are available.

    Methods:
        run: Continuously checks for new tasks and schedules them.
        stop: Signals the scheduler to stop and waits for it to finish.
        _scheduler: Distributes tasks to available threads based on priority and load.
        _checkNonFullThread: Checks for threads that are not full and can take new tasks.
        _checkNonWorkingThread: Checks for threads that are currently not working.
        _checkMinimumLoadThread: Finds the thread with the least load among the available threads.

    Notes:
        - The scheduler operates in a loop, processing tasks as they arrive.
        - Proper management of thread workload is crucial for system efficiency.
    """

    def __init__(self, SM: _SynchronizationManager, CM: _ConfigManager, ThreadTaskStorageQueue: queue.Queue, Logger: logging.Logger):
        super().__init__(name='ThreadTaskScheduler', daemon=True)
        self.SynchronizationManager = SM
        self.ConfigManager = CM
        self.ThreadTaskStorageQueue = ThreadTaskStorageQueue
        self.Logger = Logger
        self.LastSelectedThread = None
        self.CloseEvent = multiprocessing.Event()
        self.NewTaskEvent = threading.Event()
        self.sleep_time = 0.001

    def run(self):
        """
        Runs the thread task scheduler thread, continuously processing tasks from the task storage queue until instructed to stop.

        steps:
            1. Enter a loop that continues until the CloseEvent is set.
            2. Initialize task_data to None for task retrieval.
            3. Attempt to retrieve a task from the ThreadTaskStorageQueue without blocking:
                - If the queue is empty:
                    - If there are no new tasks, sleep for a specified duration.
                    - Wait for the NewTaskEvent to be set, with a timeout of 0.1 seconds.
                    - Clear the NewTaskEvent flag after processing.
            4. If a task was successfully retrieved:
                - Extract the priority and task object from the task data.
                - Call the scheduler to process the task with the given priority.

        Notes:
            - This method ensures efficient processing of tasks while managing thread sleep and wake states based on task availability.
            - The use of events and queue handling allows for responsive task management.
        """

        while not self.CloseEvent.is_set():
            task_data = None
            try:
                task_data = self.ThreadTaskStorageQueue.get_nowait()
            except queue.Empty:
                if not self.NewTaskEvent.is_set():
                    time.sleep(self.sleep_time)
                self.NewTaskEvent.wait(timeout=0.1)
                self.NewTaskEvent.clear()
            if task_data:
                priority, task_object = task_data
                self._scheduler(priority, task_object)

    def stop(self):
        """
        Stops the current thread by signaling it to close and waiting for it to finish.

        steps:
            1. Set the CloseEvent to signal the thread to stop.
            2. Set the NewTaskEvent to ensure that any waiting tasks are released.
            3. Wait for the thread to finish execution, allowing up to 2 seconds for completion.
            4. Delete the thread object to free resources.

        Notes:
            - This method ensures a graceful shutdown of the thread.
            - Proper signaling is used to notify the thread to cease operations.
        """

        self.CloseEvent.set()
        self.NewTaskEvent.set()
        self.join(2)
        del self

    def _scheduler(self, priority: int, task_object: _TaskObject):
        """
        Schedules a thread task by adding it to an appropriate thread based on their availability and load.

        :param priority: The priority of the task to be scheduled.
        :param task_object: The task object representing the task to be executed.

        steps:
            1. Check for non-working core threads.
            2. Check for non-full core threads.
            3. If there are non-working core threads:
                - Select the core thread with the minimum load and assign the task to it.
                - Set the WorkingEvent to indicate the thread is now working.
                - Return from the method.
            4. If there are no non-working threads but there are non-full core threads:
                - Select the core thread with the minimum load and assign the task to it.
                - Set the WorkingEvent to indicate the thread is now working.
                - Return from the method.
            5. If the core thread pool is not empty, check for non-working and non-full expand threads.
            6. If there are non-working expand threads:
                - Select the expand thread with the minimum load and assign the task to it.
                - Set the WorkingEvent to indicate the thread is now working.
                - Return from the method.
            7. If there are no non-working threads but there are non-full expand threads:
                - Select the expand thread with the minimum load and assign the task to it.
                - Set the WorkingEvent to indicate the thread is now working.
                - Return from the method.
            8. If no suitable threads are found, put the task in the thread task storage queue.

        Notes:
            - This method prioritizes assigning tasks to threads that are either not working or not full to maintain optimal system performance.
            - The task will be queued if no threads are available to handle it immediately.
        """

        global _ExpandThreadPool
        not_working_core_threads = self._checkNonWorkingThread("Core")
        not_full_core_threads = self._checkNonFullThread("Core")

        if not_working_core_threads:
            minimum_load_core_thread = self._checkMinimumLoadThread(not_working_core_threads, "Core")
            minimum_load_core_thread.addThreadTask(priority, task_object)
            minimum_load_core_thread.WorkingEvent.set()
            return
        if not_full_core_threads:
            minimum_load_core_thread = self._checkMinimumLoadThread(not_full_core_threads, "Core")
            minimum_load_core_thread.addThreadTask(priority, task_object)
            minimum_load_core_thread.WorkingEvent.set()
            return
        if _ExpandThreadPool:
            not_working_expand_threads = self._checkNonWorkingThread("Expand")
            not_full_expand_threads = self._checkNonFullThread("Expand")

            if not_working_expand_threads:
                minimum_load_expand_thread = self._checkMinimumLoadThread(not_working_expand_threads, "Expand")
                minimum_load_expand_thread.addThreadTask(priority, task_object)
                minimum_load_expand_thread.WorkingEvent.set()
                return
            if not_full_expand_threads:
                minimum_load_expand_thread = self._checkMinimumLoadThread(not_full_expand_threads, "Expand")
                minimum_load_expand_thread.addThreadTask(priority, task_object)
                minimum_load_expand_thread.WorkingEvent.set()
                return
            self.ThreadTaskStorageQueue.put_nowait((priority, task_object))
            return
        self.ThreadTaskStorageQueue.put_nowait((priority, task_object))

    def _checkNonFullThread(self, thread_type: Literal["Core", "Expand"]) -> list:
        """
        Checks for threads that are not currently full based on their load.

        :param thread_type: The type of threads to check, either "Core" or "Expand".

        :return: A list of thread objects that are not full.

        steps:
            1. Determine the appropriate thread pool to use based on the thread type.
            2. Determine the corresponding status pool for monitoring thread loads.
            3. Iterate through the selected thread pool and collect threads where the current load is less than the task threshold.
            4. Return the list of threads that are not full.

        Notes:
            - This method is useful for load balancing by identifying threads that can accept more tasks.
            - The task threshold is used to determine when a thread is considered full.
        """

        global _CoreThreadPool, _ExpandThreadPool
        obj_pool = _CoreThreadPool if thread_type == "Core" else _ExpandThreadPool
        status_pool = self.SynchronizationManager.CoreThreadStatusPool if thread_type == "Core" else self.SynchronizationManager.ExpandThreadStatusPool
        not_full_threads = [obj for index, obj in obj_pool.items() if status_pool[obj.ThreadName][1] < self.ConfigManager.TaskThreshold.value]
        return not_full_threads

    @staticmethod
    def _checkNonWorkingThread(thread_type: Literal["Core", "Expand"]) -> list:
        """
        Checks for threads that are not currently working.

        :param thread_type: The type of threads to check, either "Core" or "Expand".

        :return: A list of thread objects that are not currently working.

        steps:
            1. Determine the appropriate thread pool to use based on the thread type.
            2. Iterate through the selected thread pool and collect threads where the WorkingEvent is not set.
            3. Return the list of non-working threads.

        Notes:
            - This method helps in identifying threads that may need attention or restarting.
            - The WorkingEvent is used as an indicator of whether a thread is currently active.
        """

        global _CoreThreadPool, _ExpandThreadPool
        obj_pool = _CoreThreadPool if thread_type == "Core" else _ExpandThreadPool
        not_working_threads = [obj for index, obj in obj_pool.items() if not obj.WorkingEvent.is_set()]
        return not_working_threads

    def _checkMinimumLoadThread(self, threads: list, thread_type: Literal["Core", "Expand"]) -> _ThreadObject:
        """
        Checks and selects a thread with the minimum load from the provided list of threads.

        :param threads: A list of thread objects to evaluate.
        :param thread_type: The type of threads to check, either "Core" or "Expand".

        :return: The selected thread object with the minimum load.

        steps:
            1. Determine the status pool to use based on the thread type.
            2. Create a list of available threads, excluding the last selected thread if it exists.
            3. If there are no available threads, select the first thread from the original list.
            4. If there are available threads, select the one with the minimum load from the status pool.
            5. Update the last selected thread reference.

        Notes:
            - This method is used to ensure that thread selection is balanced based on load.
            - The minimum load is determined using the second value in the thread's status, which represents the load.
        """

        status_pool = self.SynchronizationManager.CoreThreadStatusPool if thread_type == "Core" else self.SynchronizationManager.ExpandThreadStatusPool
        available_threads = [p for p in threads if p != self.LastSelectedThread] if self.LastSelectedThread is not None else threads
        if not available_threads:
            selected_thread = threads[0]
        else:
            selected_thread = min(available_threads, key=lambda x: status_pool[x.ThreadName][1])
        self.LastSelectedThread = selected_thread
        return selected_thread


class TaskFuture:
    """
    The TaskFuture class

    Represents a future result of a task execution, allowing for retrieval of the task's result once it has completed.
    This class provides a mechanism to handle task IDs and manage the retrieval of results in a structured manner.

    This class does not implement the execution logic but provides abstract methods to be implemented in subclasses,
    enabling flexible task handling and execution patterns.

    Attributes:
        _TaskID: An optional string that holds the unique identifier for the associated task.

    Methods:
        taskID: Property for getting and setting the task ID.
        result: Retrieves the result of the task execution, waiting until the result is available or a timeout occurs.
        execute: Abstract method for executing tasks, to be implemented by subclasses.
        asyncExecute: Abstract method for executing asynchronous tasks, to be implemented by subclasses.

    Notes:
        - This class is crucial for managing the results of asynchronous or concurrent task executions.
        - Proper error handling and encapsulation ensure that the task ID management is secure and efficient.
        - You can inherit and override the execute and asyncExecute methods to define specific task execution logic.
    """

    def __init__(self):
        self._TaskID: Optional[str] = None

    @property
    def taskID(self):
        """
        Getter for the taskID property.

        This getter retrieves the unique identifier for the task associated with the current instance.

        :return: A string representing the unique task identifier.

        Notes:
            - This method provides access to the internal storage of the task ID (_TaskID).
            - By using a property, it allows for seamless integration with other code that accesses the task ID,
            enabling read-only access while maintaining encapsulation.
        """

        return self._TaskID

    @taskID.setter
    def taskID(self, id: str):
        """
        Setter for the taskID property.

        This setter allows setting the unique identifier for the task associated with the current instance.

        :param id: A string representing the unique task identifier to be assigned.

        Notes:
            - This method ensures that the internal storage for the task ID (_TaskID) is updated whenever
            a new ID is assigned through the taskID property.
            - Proper encapsulation is maintained by using a setter, allowing for potential future validation
            or processing when setting the task ID.
        """

        self._TaskID = id

    def result(self, timeout: int = 0):
        """
        Retrieves the result of a task execution, waiting until the result is available or a timeout occurs.

        This method checks for the completion of a task associated with the current instance's TaskID.

        :param timeout: An optional integer specifying the maximum time to wait for the task result (in seconds). Default is 0, which means no timeout.

        :return: The result of the task execution.

        raises:
            TimeoutError: If the specified timeout period is exceeded before the result becomes available.

        steps:
            1. Initialize a variable to track the elapsed time (_timeout).
            2. Enter an infinite loop:
                - Check if the current instance's TaskID is present in the global _FutureResult dictionary:
                    - If found, retrieve the task result, delete the entry from _FutureResult, and return the result.
                - If timeout is set to 0, continue looping indefinitely until a result is available.
                - If a timeout is specified, sleep for a short duration (0.001 seconds).
                - Increment the elapsed timeout.
                - Check if the elapsed timeout exceeds the specified timeout:
                    - If it does, raise a TimeoutError indicating that task execution timed out.

        Notes:
            - This method is essential for synchronous retrieval of task results in a potentially asynchronous environment.
            - Proper error handling ensures that timeouts are managed effectively, preventing indefinite waiting for task results.
        """

        global _FutureResult
        _timeout = 0
        while True:
            if self._TaskID in _FutureResult:
                task_result = _FutureResult[self._TaskID]
                del _FutureResult[self._TaskID]
                return task_result
            if timeout == 0:
                continue
            time.sleep(0.001)
            timeout += 0.001
            if _timeout >= timeout:
                raise TimeoutError("Task execution timed out.")

    def execute(self):
        """
        Defines an abstract method for executing tasks.

        This method is intended to be overridden in subclasses to provide specific
        implementation for task execution.

        raises:
            NotImplementedError: Always raised when this method is called, indicating that
            the method must be implemented in a subclass.

        Notes:
            - Subclasses should implement this method to define how tasks are executed.
            - This design enforces that any subclass must provide a concrete implementation, ensuring that the expected behavior is defined and reducing the risk of runtime errors.
        """

        raise NotImplementedError("The execute method must be implemented in a subclass.")

    async def asyncExecute(self):
        """
        Defines an abstract method for executing asynchronous tasks.

        This method is intended to be overridden in subclasses to provide specific
        implementation for asynchronous task execution.

        raises:
            NotImplementedError: Always raised when this method is called, indicating that
            the method must be implemented in a subclass.

        Notes:
            - Subclasses should implement this method to define how asynchronous tasks are executed.
            - This design enforces that any subclass must provide a concrete implementation, promoting a clear contract for the functionality expected from derived classes.
        """

        raise NotImplementedError("The asyncExecute method must be implemented in a subclass.")


class ConcurrentSystem:
    """
    TheSeedCore concurrent system main class

    Handles the overall management and execution of concurrent tasks in the system, ensuring proper coordination
    between processes and threads while maintaining a flexible task submission mechanism.

    This class implements a singleton design pattern, ensuring that only one instance of ConcurrentSystem exists.
    It facilitates the initialization of the system, the submission of both process and thread tasks, and
    provides a structured method for closing down the entire system safely.

    Attributes:
        _INSTANCE: Singleton instance of the ConcurrentSystem class.
        _INITIALIZED: Indicates whether the system has been initialized.
        _SynchronizationManager: Manages synchronization across processes and threads.
        _ConfigManager: Handles configuration settings for the system.
        _DebugMode: A flag indicating whether debug mode is enabled.
        _ProcessTaskStorageQueue: Queue to store tasks meant for process execution.
        _ThreadTaskStorageQueue: Queue to store tasks meant for thread execution.
        _Logger: Logger instance for the concurrent system.
        _LoadBalancer: Manages the allocation of tasks to processes and threads.
        _ProcessTaskScheduler: Scheduler for managing process tasks.
        _ThreadTaskScheduler: Scheduler for managing thread tasks.
        _CallbackExecutor: Manages execution of callbacks after task completion.
        _SystemThreadPoolExecutor: ThreadPoolExecutor for managing system-wide tasks.
        _SystemProcessPoolExecutor: ProcessPoolExecutor for managing system-wide processes.

    Methods:
        __new__: Controls instantiation to enforce the singleton pattern.
        __init__: Initializes the concurrent system and its components.
        submitProcessTask: Submits a task for execution in a separate process.
        submitThreadTask: Submits a task for execution in a separate thread.
        submitSystemProcessTask: Submits a task to the system's process pool executor.
        submitSystemThreadTask: Submits a task to the system's thread pool executor.
        closeSystem: Shuts down the concurrent system and stops all processes and threads.
        _setLogger: Configures and returns a logger for the concurrent system.
        _setCallbackExecutor: Determines the appropriate callback executor based on application context.
        _initSystem: Initializes the system by starting core processes and threads.
        _startCoreProcess: Starts a core process and updates the process status pool.
        _startCoreThread: Starts a core thread and updates the thread status pool.

    Notes:
        - This class is central to the management of concurrent tasks and resources in the system.
        - Proper initialization and shutdown procedures are essential for maintaining system integrity and performance.
    """

    _INSTANCE: ConcurrentSystem = None
    _INITIALIZED: bool = False
    MainEventLoop: asyncio.AbstractEventLoop = None

    def __new__(cls, SM: '_SynchronizationManager' = None, CM: '_ConfigManager' = None, DebugMode: bool = False):
        if cls._INSTANCE is None:
            cls._INSTANCE = super(ConcurrentSystem, cls).__new__(cls)
        return cls._INSTANCE

    def __init__(self, SM: '_SynchronizationManager' = None, CM: '_ConfigManager' = None, DebugMode: bool = False):
        if ConcurrentSystem._INITIALIZED:
            return
        ConcurrentSystem.MainEventLoop = _MainEventLoop
        self._SynchronizationManager = SM
        self._ConfigManager = CM
        self._DebugMode = DebugMode
        self._ProcessTaskStorageQueue: multiprocessing.Queue = multiprocessing.Queue()
        self._ThreadTaskStorageQueue: queue.Queue = queue.Queue()
        self._Logger = self._setLogger()
        self._LoadBalancer = _LoadBalancer(self._SynchronizationManager, self._ConfigManager, self._Logger, self._DebugMode)
        self._ProcessTaskScheduler = _ProcessTaskScheduler(self._SynchronizationManager, self._ConfigManager, self._ProcessTaskStorageQueue, self._Logger)
        self._ThreadTaskScheduler = _ThreadTaskScheduler(self._SynchronizationManager, self._ConfigManager, self._ThreadTaskStorageQueue, self._Logger)
        self._CallbackExecutor: Union[_QtCallbackExecutor, _CoreCallbackExecutor] = self._setCallbackExecutor()
        self._SystemThreadPoolExecutor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=self._ConfigManager.CoreProcessCount.value + self._ConfigManager.CoreThreadCount.value)
        if self._ConfigManager.CoreProcessCount.value != 0:
            self._SystemProcessPoolExecutor: ProcessPoolExecutor = ProcessPoolExecutor(max_workers=self._ConfigManager.CoreProcessCount.value)
        ConcurrentSystem._INITIALIZED = True
        self._initSystem()

    @classmethod
    def submitProcessTask(cls, task: callable, priority: int = 0, callback: callable = None, future: type(TaskFuture) = None, lock: bool = False, lock_timeout: int = 3, timeout: int = None, gpu_boost: bool = False, gpu_id: int = 0, retry: bool = True, max_retries: int = 3, *args, **kwargs) -> TaskFuture:
        """
        Submits a process task to the ConcurrentSystem for execution with specified parameters.

        This class method allows for the asynchronous submission of tasks that will be processed in a separate process, including options for priority, callback handling, and resource locking.

        :param task: The callable task to be executed.
        :param priority: An integer representing the task's priority, where lower values indicate higher priority. Defaults to 0.
        :param callback: An optional callable to be executed upon completion of the task.
        :param future: An optional type for the task's future result. If None, defaults to TaskFuture.
        :param lock: A boolean indicating whether the task should acquire a lock during execution. Defaults to False.
        :param lock_timeout: An integer specifying the time (in seconds) to wait for the lock. Defaults to 3.
        :param timeout: An optional integer specifying the maximum time (in seconds) for the task to execute. Defaults to None.
        :param gpu_boost: A boolean indicating whether to utilize GPU resources for the task. Defaults to False.
        :param gpu_id: An integer specifying which GPU to use if gpu_boost is enabled. Defaults to 0.
        :param retry: A boolean indicating whether the task should be retried on failure. Defaults to True.
        :param max_retries: An integer specifying the maximum number of retries allowed. Defaults to 3.
        :param args: Additional positional arguments to pass to the task.
        :param kwargs: Additional keyword arguments to pass to the task.

        :return: An instance of the future type (TaskFuture) associated with the submitted task.

        raises:
            RuntimeError: If the current process is not the main process or if the ConcurrentSystem has not been initialized, or if the core process count is set to 0.
            ValueError: If the task is not callable.

        steps:
            1. Check if the current process is the main process:
                - If not, raise a RuntimeError indicating that process task submission must be done in the main process.
            2. Verify that the ConcurrentSystem instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the system has not been initialized.
            3. Check if the core process count is set to 0:
                - If it is, raise a RuntimeError indicating that process task submission is not allowed.
            4. Generate a unique task ID using UUID.
            5. Create an instance of the future type if provided, otherwise create a default TaskFuture instance.
            6. Attempt to serialize the task:
                - If serialization fails, log the error and return the future instance.
            7. Store the callback in the _CallbackObject dictionary using the task ID, if provided.
            8. Create a task object encapsulating the task and its parameters.
            9. Place the task object in the process task storage queue with its priority.
            10. Signal the process task scheduler that a new task is available.
            11. Return the future instance associated with the submitted task.

        Notes:
            - This method is essential for managing the execution of tasks within the ConcurrentSystem while maintaining responsiveness and resource management.
            - Proper handling of task priorities and callbacks ensures efficient execution and response to task completion.
        """

        global _CallbackObject, _CoreProcessPool, _ExpandProcessPool
        if multiprocessing.current_process().name != 'MainProcess':
            raise RuntimeError("Process task submission must be done in the main process.")
        if cls._INSTANCE is None:
            raise RuntimeError("TheSeedCore ConcurrentSystem has not been initialized.")
        if cls._INSTANCE._ConfigManager.CoreProcessCount.value == 0:
            raise RuntimeError("Core process count is set to 0. Process task submission is not allowed.")
        task_id = f"{uuid.uuid4()}"
        future_instance = future() if future is not None else TaskFuture()
        future_instance.taskID = task_id
        try:
            pickle.dumps(task)
        except (pickle.PicklingError, AttributeError, TypeError) as e:
            cls._INSTANCE._Logger.error(f"Task [{task.__name__} - {task_id}] serialization failed. Task submission has been rejected.\n{e}")
            return future_instance
        if callback is not None:
            _CallbackObject[task_id] = callback
        task_object = _TaskObject(task, task_id, False if callback is None else True, lock, lock_timeout, timeout, gpu_boost, gpu_id, retry, max_retries, *args, **kwargs)
        cls._INSTANCE._ProcessTaskStorageQueue.put_nowait((priority if not priority > 10 else 10, task_object))
        cls._INSTANCE._ProcessTaskScheduler.NewTaskEvent.set()
        return future_instance

    @classmethod
    def submitThreadTask(cls, task: callable, priority: int = 0, callback: callable = None, future: type(TaskFuture) = None, lock: bool = False, lock_timeout: int = 3, timeout: int = None, gpu_boost: bool = False, gpu_id: int = 0, retry: bool = True, max_retries: int = 3, *args, **kwargs):
        """
        Submits a thread task to the ConcurrentSystem for execution with specified parameters.

        This class method allows for the asynchronous submission of tasks that will be processed in a thread, including options for priority, callback handling, and resource locking.

        :param task: The callable task to be executed.
        :param priority: An integer representing the task's priority, where lower values indicate higher priority. Defaults to 0.
        :param callback: An optional callable to be executed upon completion of the task.
        :param future: An optional type for the task's future result. If None, defaults to TaskFuture.
        :param lock: A boolean indicating whether the task should acquire a lock during execution. Defaults to False.
        :param lock_timeout: An integer specifying the time (in seconds) to wait for the lock. Defaults to 3.
        :param timeout: An optional integer specifying the maximum time (in seconds) for the task to execute. Defaults to None.
        :param gpu_boost: A boolean indicating whether to utilize GPU resources for the task. Defaults to False.
        :param gpu_id: An integer specifying which GPU to use if gpu_boost is enabled. Defaults to 0.
        :param retry: A boolean indicating whether the task should be retried on failure. Defaults to True.
        :param max_retries: An integer specifying the maximum number of retries allowed. Defaults to 3.
        :param args: Additional positional arguments to pass to the task.
        :param kwargs: Additional keyword arguments to pass to the task.

        :return: An instance of the future type (TaskFuture) associated with the submitted task.

        raises:
            RuntimeError: If the current process is not the main process or if the ConcurrentSystem has not been initialized.
            ValueError: If the task is not callable.

        steps:
            1. Check if the current process is the main process:
                - If not, raise a RuntimeError indicating that thread task submission must be done in the main process.
            2. Verify that the ConcurrentSystem instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the system has not been initialized.
            3. Generate a unique task ID using UUID.
            4. Create an instance of the future type if provided, otherwise create a default TaskFuture instance.
            5. Store the callback in the _CallbackObject dictionary using the task ID.
            6. Create a task object encapsulating the task and its parameters.
            7. Place the task object in the thread task storage queue with its priority.
            8. Signal the task scheduler that a new task is available.
            9. Return the future instance associated with the submitted task.

        Notes:
            - This method is essential for managing the execution of tasks within the ConcurrentSystem while maintaining responsiveness and resource management.
            - Proper handling of task priorities and callbacks ensures efficient execution and response to task completion.
        """

        global _CallbackObject
        if multiprocessing.current_process().name != 'MainProcess':
            raise RuntimeError("Thread task submission must be done in the main process.")
        if cls._INSTANCE is None:
            raise RuntimeError("TheSeedCore ConcurrentSystem has not been initialized.")
        task_id = f"{uuid.uuid4()}"
        future_instance = future() if future is not None else TaskFuture()
        future_instance.taskID = task_id
        if callback is not None:
            _CallbackObject[task_id] = callback
        task_object = _TaskObject(task, task_id, False if callback is None else True, lock, lock_timeout, timeout, gpu_boost, gpu_id, retry, max_retries, *args, **kwargs)
        cls._INSTANCE._ThreadTaskStorageQueue.put_nowait((priority if not priority > 10 else 10, task_object))
        cls._INSTANCE._ThreadTaskScheduler.NewTaskEvent.set()
        return future_instance

    @classmethod
    def submitSystemProcessTask(cls, task: callable, count: int = 1, *args, **kwargs):
        """
        Submits one or more tasks to the system process pool for execution.

        This method allows tasks to be scheduled for execution in the ConcurrentSystem's process pool.

        :param task: A callable representing the task to be executed.
        :param count: An integer specifying the number of times to submit the task (default is 1).
        :param args: Positional arguments to pass to the task.
        :param kwargs: Keyword arguments to pass to the task.

        :return:
            - If count is greater than 1, returns a list of futures representing the submitted tasks.
            - If count is 1, returns a single future representing the submitted task.

        raises:
            RuntimeError:
                - If the ConcurrentSystem has not been initialized.
                - If the core process count is set to 0, indicating that process task submission is not allowed.
            ValueError: If the provided task is not callable.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the ConcurrentSystem has not been initialized.
            2. Validate that the provided task is callable:
                - If not, raise a ValueError indicating that the task must be callable.
            3. Check if the core process count is set to 0:
                - If it is, raise a RuntimeError indicating that process task submission is not allowed.
            4. Initialize an empty list to store futures.
            5. If count is greater than 1:
                - Loop count times and submit the task to the SystemProcessPoolExecutor for each iteration, appending each future to the list.
                - Return the list of futures.
            6. If count is 1:
                - Submit the task to the SystemProcessPoolExecutor and return the single future.

        Notes:
            - This method is useful for submitting multiple instances of a task to the process pool for concurrent execution.
            - Proper error handling ensures that only valid tasks are submitted, preventing runtime issues.
        """

        if cls._INSTANCE is None:
            raise RuntimeError("The ConcurrentSystem has not been initialized.")
        if not callable(task):
            raise ValueError("The task must be a callable.")
        if cls._INSTANCE._ConfigManager.CoreProcessCount.value == 0:
            raise RuntimeError("Core process count is set to 0. Process task submission is not allowed.")
        futures = []
        if count > 1:
            for i in range(count):
                future = cls._INSTANCE._SystemProcessPoolExecutor.submit(task, *args, **kwargs)
                futures.append(future)
            return futures
        future = cls._INSTANCE._SystemProcessPoolExecutor.submit(task, *args, **kwargs)
        return future

    @classmethod
    def submitSystemThreadTask(cls, task: callable, count: int = 1, *args, **kwargs):
        """
        Submits one or more tasks to the system thread pool for execution.

        This method allows tasks to be scheduled for execution in the ConcurrentSystem's thread pool.

        :param task: A callable representing the task to be executed.
        :param count: An integer specifying the number of times to submit the task (default is 1).
        :param args: Positional arguments to pass to the task.
        :param kwargs: Keyword arguments to pass to the task.

        :return:
            - If count is greater than 1, returns a list of futures representing the submitted tasks.
            - If count is 1, returns a single future representing the submitted task.

        raises:
            RuntimeError: If the ConcurrentSystem has not been initialized.
            ValueError: If the provided task is not callable.

        steps:
            1. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the ConcurrentSystem has not been initialized.
            2. Validate that the provided task is callable:
                - If not, raise a ValueError indicating that the task must be callable.
            3. Initialize an empty list to store futures.
            4. If count is greater than 1:
                - Loop count times and submit the task to the SystemThreadPoolExecutor for each iteration, appending each future to the list.
                - Return the list of futures.
            5. If count is 1:
                - Submit the task to the SystemThreadPoolExecutor and return the single future.

        Notes:
            - This method is useful for submitting multiple instances of a task to the thread pool for concurrent execution.
            - Proper error handling ensures that only valid tasks are submitted, preventing runtime issues.
        """

        if cls._INSTANCE is None:
            raise RuntimeError("The ConcurrentSystem has not been initialized.")
        if not callable(task):
            raise ValueError("The task must be a callable.")
        futures = []
        if count > 1:
            for i in range(count):
                future = cls._INSTANCE._SystemThreadPoolExecutor.submit(task, *args, **kwargs)
                futures.append(future)
            return futures
        future = cls._INSTANCE._SystemThreadPoolExecutor.submit(task, *args, **kwargs)
        return future

    @classmethod
    def closeSystem(cls):
        """
        Closes the system, shutting down all processes and threads in the ConcurrentSystem.

        raises:
            RuntimeError: If the function is not called from the main process.
            RuntimeError: If the ConcurrentSystem instance has not been initialized.

        steps:
            1. Check if the current process is the main process:
                - If not, raise a RuntimeError indicating that system closing must be done in the main process.
            2. Check if the class-level instance (_INSTANCE) is initialized:
                - If not, raise a RuntimeError indicating that the ConcurrentSystem has not been initialized.
            3. Initialize an empty list (futures) to store future objects for process and thread shutdowns.
            4. For each process in the CoreProcessPool:
                - Submit the stop method of each process object to the SystemThreadPoolExecutor and store the returned future.
            5. For each thread in the CoreThreadPool:
                - Submit the stop method of each thread object to the SystemThreadPoolExecutor and store the returned future.
            6. Repeat steps 4 and 5 for the ExpandProcessPool and ExpandThreadPool.
            7. Wait for all submitted futures to complete by calling future.result() for each future in the list.
            8. Stop the LoadBalancer associated with the ConcurrentSystem instance.
            9. If CoreProcessCount is not zero, stop the ProcessTaskScheduler.
            10. Stop the ThreadTaskScheduler associated with the ConcurrentSystem instance.
            11. Close the CallbackExecutor to clean up resources.
            12. Shutdown the SystemThreadPoolExecutor, waiting for tasks to complete and canceling any remaining futures.
            13. If CoreProcessCount is not zero, also shutdown the SystemProcessPoolExecutor, waiting and canceling as above.
            14. Delete the instance of the ConcurrentSystem to free resources.

        Notes:
            - This method ensures that all active processes and threads are properly stopped before the system is closed.
            - It provides a clean and graceful shutdown mechanism, preventing potential resource leaks and ensuring that all tasks are completed before exiting.
        """

        global _CoreProcessPool, _CoreThreadPool, _ExpandProcessPool, _ExpandThreadPool
        if multiprocessing.current_process().name != 'MainProcess':
            raise RuntimeError("System closing must be done in the main process.")
        if cls._INSTANCE is None:
            raise RuntimeError("TheSeedCore ConcurrentSystem has not been initialized.")
        futures = []
        for process_name, process_obj in _CoreProcessPool.items():
            future = cls._INSTANCE._SystemThreadPoolExecutor.submit(process_obj.stop)
            futures.append(future)
        for thread_name, thread_obj in _CoreThreadPool.items():
            future = cls._INSTANCE._SystemThreadPoolExecutor.submit(thread_obj.stop)
            futures.append(future)
        for process_name, process_obj in _ExpandProcessPool.items():
            future = cls._INSTANCE._SystemThreadPoolExecutor.submit(process_obj.stop)
            futures.append(future)
        for thread_name, thread_obj in _ExpandThreadPool.items():
            future = cls._INSTANCE._SystemThreadPoolExecutor.submit(thread_obj.stop)
            futures.append(future)
        for future in futures:
            future.result()
        cls._INSTANCE._LoadBalancer.stop()
        if cls._INSTANCE._ConfigManager.CoreProcessCount.value != 0:
            cls._INSTANCE._ProcessTaskScheduler.stop()
        cls._INSTANCE._ThreadTaskScheduler.stop()
        cls._INSTANCE._CallbackExecutor.closeExecutor()
        cls._INSTANCE._SystemThreadPoolExecutor.shutdown(wait=True, cancel_futures=True)
        if cls._INSTANCE._ConfigManager.CoreProcessCount.value != 0:
            cls._INSTANCE._SystemProcessPoolExecutor.shutdown(wait=True, cancel_futures=True)
        del cls._INSTANCE

    def _setLogger(self) -> logging.Logger:
        """
        Sets up and returns a logger for the concurrent system.

        :return: An instance of logging.Logger configured for the concurrent system.

        steps:
            1. Create a logger with the name 'ConcurrentSystem'.
            2. Set the logger's level to DEBUG.
            3. Create a console handler for the logger.
            4. Set the console handler's level based on the debug mode:
                - If in debug mode, set to DEBUG.
                - Otherwise, set to the maximum of DEBUG and WARNING.
            5. Create a formatter for log messages with a specific format.
            6. Set the formatter for the console handler.
            7. Add the console handler to the logger.

        Notes:
            - The logger is configured to display log messages to the console with color formatting.
            - Adjust the logging level based on the debug mode to control verbosity.
        """

        logger = logging.getLogger('[ConcurrentSystem]')
        logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        if self._DebugMode:
            console_handler.setLevel(logging.DEBUG)
        else:
            console_handler.setLevel(max(logging.DEBUG, logging.WARNING))

        formatter = _ColoredFormatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)

        logger.addHandler(console_handler)
        return logger

    def _setCallbackExecutor(self):
        """
        Sets the appropriate callback executor based on the application context.

        :return: An instance of either _QtCallbackExecutor or _CoreCallbackExecutor.

        steps:
            1. Check if QApplication is available and an instance exists.
            2. If a QApplication instance exists, return a _QtCallbackExecutor initialized with the synchronization manager.
            3. If no QApplication instance is available, return a _CoreCallbackExecutor initialized with the synchronization manager.

        Notes:
            - This method determines the executor to use for callbacks based on the presence of a Qt application context.
        """

        # noinspection PyUnresolvedReferences
        if QApplication and QApplication.instance():
            return _QtCallbackExecutor(self._SynchronizationManager)
        return _CoreCallbackExecutor(self._SynchronizationManager)

    def _initSystem(self):
        """
        Initializes the system by starting the specified number of core processes and threads.

        steps:
            1. Initialize empty futures list to track process and thread startup.
            2. Start core processes based on the configured count:
                - For each process, generate a unique process name and submit the start task to the executor.
            3. Start core threads based on the configured count:
                - For each thread, generate a unique thread name and submit the start task to the executor.
            4. Wait for all submitted tasks (processes and threads) to complete.
            5. Start the load balancer.
            6. If the core process count is greater than zero, start the process task scheduler.
            7. Start the thread task scheduler.
            8. Start the callback executor.

        Notes:
            - Ensure that the core process and thread counts are configured properly to avoid resource issues.
            - The system will only start the process task scheduler if there are processes to manage.
        """

        global _CoreProcessPool, _CoreThreadPool
        futures = []
        for i in range(self._ConfigManager.CoreProcessCount.value):
            process_name = f"Process-{i}"
            future = self._SystemThreadPoolExecutor.submit(self._startCoreProcess, process_name)
            futures.append(future)
        for i in range(self._ConfigManager.CoreThreadCount.value):
            thread_name = f"Thread-{i}"
            future = self._SystemThreadPoolExecutor.submit(self._startCoreThread, thread_name)
            futures.append(future)
        for future in futures:
            future.result()
        self._LoadBalancer.start()
        if self._ConfigManager.CoreProcessCount.value != 0:
            self._ProcessTaskScheduler.start()
        self._ThreadTaskScheduler.start()
        self._CallbackExecutor.startExecutor()

    def _startCoreProcess(self, process_name):
        """
        Starts a core process and manages its status in the process pool.

        :param process_name: The name of the process to be started.

        :return: An instance of the created process object.

        steps:
            1. Create a new process object with the specified process name and relevant managers.
            2. Start the process.
            3. Add the process object to the global process pool using its name.
            4. Update the core process status pool with the process's identifier and initial status.

        Notes:
            - Ensure that the process name is unique within the pool to avoid conflicts.
            - The process is initialized with synchronization and configuration managers, along with the debug mode setting.
        """

        global _CoreProcessPool
        process_object = _ProcessObject(process_name, "Core", self._SynchronizationManager, self._ConfigManager, self._DebugMode)
        process_object.start()
        _CoreProcessPool[process_name] = process_object
        self._SynchronizationManager.CoreProcessStatusPool[process_name] = (process_object.pid, 0, 0)
        return process_object

    def _startCoreThread(self, thread_name):
        """
        Starts a core thread and manages its status in the thread pool.

        :param thread_name: The name of the thread to be started.

        :return: An instance of the created thread object.

        steps:
            1. Create a new thread object with the specified thread name and relevant managers.
            2. Start the thread.
            3. Add the thread object to the global thread pool using its name.
            4. Update the core thread status pool with the thread's identifier and initial status.

        Notes:
            - Ensure that the thread name is unique within the pool to avoid conflicts.
            - The thread is initialized with synchronization and configuration managers for proper operation.
        """

        global _CoreThreadPool
        thread_object = _ThreadObject(thread_name, "Core", self._SynchronizationManager, self._ConfigManager, self._Logger)
        thread_object.start()
        _CoreThreadPool[thread_name] = thread_object
        self._SynchronizationManager.CoreThreadStatusPool[thread_name] = (thread_object.ident, 0)
        return thread_object


def ConnectConcurrentSystem(**kwargs) -> ConcurrentSystem:
    """
    Initializes and connects to a ConcurrentSystem.

    This function sets up the configuration for the ConcurrentSystem, initializes necessary managers,
    and creates an instance of the ConcurrentSystem.

    :param kwargs: Keyword arguments for configuring the ConcurrentSystem. Available options include:
        - DebugMode: A boolean indicating if the system should run in debug mode (default is determined by the environment).
        - CoreProcessCount: The number of core processes to be initialized.
        - CoreThreadCount: The number of core threads to be initialized.
        - MaximumProcessCount: The maximum number of processes allowed.
        - MaximumThreadCount: The maximum number of threads allowed.
        - IdleCleanupThreshold: The threshold for idle cleanup.
        - ProcessPriority: The priority of the process (default is "NORMAL").
        - TaskThreshold: The threshold for the number of tasks.
        - GlobalTaskThreshold: The global threshold for tasks.
        - ExpandPolicy: The policy for expanding resources.
        - ShrinkagePolicy: The policy for shrinking resources.
        - ShrinkagePolicyTimeout: The timeout for the shrinkage policy.

    :return: An instance of the ConcurrentSystem.

    raises:
        TypeError: If DebugMode is not a boolean.

    steps:
        1. Determine if the environment is a development environment based on the presence of `sys.frozen`.
        2. Retrieve the DebugMode from kwargs, defaulting to the development environment setting.
        3. Validate that DebugMode is a boolean; raise TypeError if not.
        4. Create a configuration object for the ConcurrentSystem using provided kwargs.
        5. Initialize a shared object manager using multiprocessing.Manager.
        6. Create a synchronization manager using the shared object manager.
        7. Create a configuration manager using the shared object manager and the configuration.
        8. Instantiate the ConcurrentSystem with the synchronization manager, configuration manager, and debug mode.
        9. Show the logo of ConcurrentSystem.
        10. Return the initialized ConcurrentSystem instance.

    Notes:
        - This function serves as the entry point for setting up the concurrent processing environment.
        - It includes necessary validation and configuration to ensure the system runs optimally.
    """

    global _MainEventLoop
    _development_env = not hasattr(sys, "frozen") and not globals().get("__compiled__", False)
    _debug_mode = kwargs.get('DebugMode', _development_env)
    if not isinstance(_debug_mode, bool):
        raise TypeError("DebugMode must be a boolean.")
    if QApplication is not None and QApplication.instance():
        _MainEventLoop = qasync.QEventLoop(QApplication.instance())
        # noinspection PyUnresolvedReferences
        QApplication.instance().aboutToQuit.connect(ConcurrentSystem.closeSystem)
    else:
        _MainEventLoop = asyncio.get_event_loop()
    # noinspection PyTypeChecker
    _config = _ConcurrentSystemConfig(
        CoreProcessCount=kwargs.get('CoreProcessCount', None),
        CoreThreadCount=kwargs.get('CoreThreadCount', None),
        MaximumProcessCount=kwargs.get('MaximumProcessCount', None),
        MaximumThreadCount=kwargs.get('MaximumThreadCount', None),
        IdleCleanupThreshold=kwargs.get('IdleCleanupThreshold', None),
        ProcessPriority=kwargs.get('ProcessPriority', "NORMAL"),
        TaskThreshold=kwargs.get('TaskThreshold', None),
        GlobalTaskThreshold=kwargs.get('GlobalTaskThreshold', None),
        ExpandPolicy=kwargs.get('ExpandPolicy', None),
        ShrinkagePolicy=kwargs.get('ShrinkagePolicy', None),
        ShrinkagePolicyTimeout=kwargs.get('ShrinkagePolicyTimeout', None),
    )
    _shared_object_manager = multiprocessing.Manager()
    _synchronization_manager = _SynchronizationManager(_shared_object_manager)
    _config_manager = _ConfigManager(_shared_object_manager, _config)
    _concurrent_system = ConcurrentSystem(_synchronization_manager, _config_manager, _debug_mode)
    CYAN_BOLD = "\033[1m\033[36m"
    RESET = "\033[0m"
    print(CYAN_BOLD + "   ______                                                               __    _____                    __                   ")
    print(CYAN_BOLD + "  / ____/  ____    ____   _____  __  __   _____   _____  ___    ____   / /_  / ___/   __  __   _____  / /_  ___    ____ ___ ")
    print(CYAN_BOLD + " / /      / __ \\  / __ \\ / ___/ / / / /  / ___/  / ___/ / _ \\  / __ \\ / __/  \\__ \\   / / / /  / ___/ / __/ / _ \\  / __ `__ \\")
    print(CYAN_BOLD + "/ /___   / /_/ / / / / // /__  / /_/ /  / /     / /    /  __/ / / / // /_   ___/ /  / /_/ /  (__  ) / /_  /  __/ / / / / / /")
    print(CYAN_BOLD + "\\____/   \\____/ /_/ /_/ \\___/  \\__,_/  /_/     /_/     \\___/ /_/ /_/ \\__/  /____/   \\__, /  /____/  \\__/  \\___/ /_/ /_/ /_/ ")
    print(CYAN_BOLD + "                                                                                   /____/                                  " + RESET)
    return _concurrent_system
