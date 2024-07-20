from pymonntorch import NeuronGroup, SynapseGroup, Recorder, EventRecorder


from conex import Neocortex, InputLayer, Synapsis, prioritize_behaviors, CorticalColumn
from conex.behaviors.synapses import SynapseInit, WeightInitializer, LateralDendriticInput, SimpleDendriticInput, WeightNormalization
from conex.behaviors.synapses.learning import SimpleSTDP, SimpleRSTDP

from conex.behaviors.neurons.axon import NeuronAxon
from conex.behaviors.neurons.specs import SpikeTrace, Fire, KWTA
from conex.behaviors.neurons.dendrite import SimpleDendriteStructure, SimpleDendriteComputation
from conex.behaviors.neurons.neuron_types.lif_neurons import LIF, ELIF
from conex.behaviors.neurons import ActivityBaseHomeostasis, VoltageBaseHomeostasis

from conex.behaviors.network.neuromodulators import Dopamine
from conex.behaviors.network.payoff import Payoff


from neuralBehaviors import InputBehavior, LIFBehavior, FireBehavior
from inputs import DynamicInput


class ModelSTDP:
    def __init__(self, in_size, out_size, mid_size=10, mid_dends_sum:int=15, out_dends_sum:int=20, R=100) -> None:
        self.net = Neocortex(dt=1, behavior={
            # **prioritize_behaviors([
            #     Payoff(), #100
            #     Dopamine(tau_dopamine=2), #120
            # ])
        })
        self.dataset = DynamicInput(dim=in_size,)
        self.inLayer = NeuronGroup(size=in_size, net=self.net, behavior={
            170: InputBehavior(self.dataset, isForceSpike=True),
            360: SpikeTrace(tau_s=0.5),
            380: NeuronAxon(),
            500: EventRecorder(variables=['spikes']),
        })
        self.midLayer = NeuronGroup(size=mid_size, net=self.net, behavior={
            170: InputBehavior(),
            220: SimpleDendriteStructure(),
            240: SimpleDendriteComputation(),
            260: LIFBehavior(func='base', R=R),
            340: FireBehavior(),
            360: SpikeTrace(tau_s=0.5),
            380: NeuronAxon(),
            500: EventRecorder(variables=['spikes']),
            501: Recorder(variables=['v']),
        })
        self.outLayer = NeuronGroup(size=out_size, net=self.net, behavior={
            170: InputBehavior(),
            220: SimpleDendriteStructure(),
            240: SimpleDendriteComputation(),
            260: LIFBehavior(func='base', R=R),
            340: FireBehavior(),
            360: SpikeTrace(tau_s=0.5),
            380: NeuronAxon(),
            500: EventRecorder(variables=['spikes']),
            501: Recorder(variables=['v']),
        })

        self.syn_in_mid = SynapseGroup(net=self.net, src=self.inLayer, dst=self.midLayer, behavior={
            **prioritize_behaviors([
                SynapseInit(), # 2
                WeightInitializer(mode='normal(mean=5, std=3)', scale=1 + mid_dends_sum // mid_size), # 3
                SimpleDendriticInput(), # 180
                SimpleSTDP(a_plus=10, a_minus=10), # 400
                WeightNormalization(norm=mid_dends_sum) # 420
            ]),
            501: Recorder(variables=['weights']),
        }, tag="Proximal")

        self.syn_mid_out = SynapseGroup(net=self.net, src=self.midLayer, dst=self.outLayer, behavior={
            **prioritize_behaviors([
                SynapseInit(), # 2
                WeightInitializer(mode='normal(mean=5, std=3)', scale=1 + out_dends_sum // mid_size), # 3
                SimpleDendriticInput(), # 180
                SimpleSTDP(a_plus=10, a_minus=10,), # 400
                WeightNormalization(norm=out_dends_sum), # 420
            ]),
            501: Recorder(variables=['weights']),
        }, tag="Proximal")

        self.net.initialize(info=False)
    
    def fit(self, signal, periodic=False, iterations=0):
        """
        Params
        -----
        `period`
        """
        assert periodic == False or iterations > 0
        self.dataset.load(signal, period=1)
        if periodic:
            self.net.simulate_iterations(iterations * signal.shape[0], measure_block_time=False)
        else:
            self.net.simulate_iterations(signal.shape[0], measure_block_time=False)
        pass