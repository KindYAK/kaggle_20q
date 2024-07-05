import asyncio

SEMAPHORES = {}


def set_semaphore(key, value):
    SEMAPHORES[key] = asyncio.Semaphore(value)


def get_semaphore(key):
    return SEMAPHORES.get(key, None)
