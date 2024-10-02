# -*- coding: utf-8 -*-
from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

from TheSeedCoreConcurrentSystem import ConcurrentSystem, ConnectConcurrentSystem

if TYPE_CHECKING:
    pass
count = 0


async def test_function(start_time: float):
    current_time = time.time()
    print(f"Test function arrived at {current_time - start_time:.3}\n")
    await asyncio.sleep(1)
    finish_time = time.time()
    return start_time, f"{finish_time - current_time:.3}"


async def test_callback(result):
    global count
    current_time = time.time()
    print(f"Test callback arrived at {current_time - result[0]:.3}")
    print(f"Task elapsed time is {result[1]}\n")
    count += 1
    if count == 10:
        for i in range(10):
            print("System shutdown in", 10 - i, "seconds")
            await asyncio.sleep(1)
        print("System shutdown")
        ConcurrentSystem.closeSystem()


async def example():
    print("Hello. This is TheSeedCoreConcurrentSystem")
    ConcurrentSystem.submitSystemThreadTask(
        ConcurrentSystem.submitThreadTask, 10, test_function, callback=test_callback, start_time=time.time()
    )


if __name__ == "__main__":
    ConnectConcurrentSystem(Priority="HIGH", CoreProcessCount=4, MaximumProcessCount=16, CoreThreadCount=16, MaximumThreadCount=16, ExpandPolicy="AutoExpand", ShrinkagePolicy="AutoShrink")
    ConcurrentSystem.MainEventLoop.create_task(example())
    ConcurrentSystem.MainEventLoop.run_forever()
