from typing import Union
from memory.memory_operator import MemoryOperator
from memory.no_memory_operator import NoMemoryOperator


def create_memory_operator(memory_length: int) -> Union[MemoryOperator, NoMemoryOperator]:
    use_memory = memory_length > 0
    if use_memory:
        print('creating memory operator with memory length ', memory_length)
        return MemoryOperator(memory_length)
    print('no memory')
    return NoMemoryOperator()
