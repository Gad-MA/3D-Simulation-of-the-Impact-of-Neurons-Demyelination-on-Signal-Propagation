from HH_Computational_Model import computional_model, dt
from helpers import toCSV, myelinToggles

myelinToggles = myelinToggles()

simulation_duration = 50
model = computional_model(
    stimulus_initial_time=0,
    stimulus_duration=1,
    simulation_duration=simulation_duration,
    isSensoryMylinated=myelinToggles['isSensoryMyelinated'],
    isExtensorMylinated=myelinToggles['isExtensorMyelinated'],
    isInhibitorMylinated=myelinToggles['isInhibitoryMyelinated'],
    isFlexorMylinated=myelinToggles['isFlexorMyelinated'],
)

activation_threshold = -53  # in mV


for i in range(int(simulation_duration / dt)):
    for neuron in model:
        model[neuron][i] = (
            int(abs(model[neuron][i] - activation_threshold))
            if (model[neuron][i] >= activation_threshold)
            else 0
        )

# print(model['sensory'][1])
# print(model['extensor'])
# print(model['inhibitory'])
# print(model['flexor'])

toCSV(
    model["sensory"],
    model["extensor"],
    model["inhibitory"],
    model["flexor"],
)
