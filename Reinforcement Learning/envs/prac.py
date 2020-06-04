import OptionsEnv
from OptionsEnv import FourRooms

a = FourRooms()
b = a.encode([1,[6,2]],in_hallway = None)
print('h',b)