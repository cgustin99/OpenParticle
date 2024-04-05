import numpy as np
from openparticle import ParticleOperator, FockState


op = ParticleOperator('a1^')
state = FockState([0, 3], [1], [(1, 1)])


print(op)
print(state)

print(op*state)
