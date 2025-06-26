import time
import sys
import os

sys.path.append(os.getcwd() + '/..')

from lib.dynapse2_util import set_parameter, clear_srams
from lib.dynapse2_spikegen import get_fpga_time, send_events
from lib.dynapse2_raster import *
from lib.dynapse2_obj import *

import random

from samna.dynapse2 import *


def perceptron_xor(board, number_of_chips):
    # your code starts here
    model = board.get_model()
    model.reset(ResetType.PowerCycle, 0b1)
    time.sleep(1)

    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(1)

    # for each core, set the neuron to monitor
    neuron = 3
    for c in range(1):
        myConfig.chips[0].cores[c].neuron_monitoring_on = True
        myConfig.chips[0].cores[c].monitored_neuron = neuron
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set neuron parameters
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N", 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", 3, 50)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", 1, 255)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 3, 255)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set synapse parameters  -- enabled AM and SC
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', 2, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_ETAU_P', 2, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_EGAIN_P', 3, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DESC_ITAU_P', 1, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DESC_IGAIN_P', 4, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', 0, 0)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', 4, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W2_P', 2, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W3_P', 4, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 4, 80)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set SRAM -- axons
    clear_srams(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips), all_to_all=True)
    model.apply_configuration(myConfig)
    time.sleep(1)

    # set CAM -- synapses
    for c in range(1):
        for i in range(256):
            cams = [Dynapse2Synapse() for _ in range(64)]
            for j in range(64):
                weights = [False, False, False, False]
                if i < 32:
                    weights[1] = True
                    cams[j].tag = 1024 + (j % 8)
                    cams[j].dendrite = Dendrite.ampa
                elif 32 <= i < 64:
                    weights[1] = True
                    cams[j].tag = 1024 + (j % 8)
                    cams[j].dendrite = Dendrite.nmda
                elif 96 <= i < 128 and j < 32:
                    weights[2] = True
                    cams[j].tag = j
                    cams[j].dendrite = Dendrite.ampa
                elif 96 <= i < 128 and j >= 32:
                    weights[3] = True
                    cams[j].tag = j
                    cams[j].dendrite = Dendrite.shunt
                else:
                    weights[0] = True
                    cams[j].tag = 0
                    cams[j].dendrite = Dendrite.none
                cams[j].weight = weights
            myConfig.chips[0].cores[c].neurons[i].synapses = cams
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    print("\nAll configurations done!\n")

    ts = get_fpga_time(board=board) + 100000

    input_events = []
    for i in range(100):
        input_events += VirtualSpikeConstructor(1024 + random.randint(0, 3), [True, False, False, False], ts + i * 10000).spikes
        input_events += VirtualSpikeConstructor(1024 + random.randint(4, 7), [True, False, False, False], ts + i * 10000).spikes

    ts += 1000000
    for i in range(100):
        input_events += VirtualSpikeConstructor(1024 + random.randint(0, 7), [True, False, False, False], ts + i * 10000).spikes

    send_events(board=board, events=input_events, min_delay=100000)
    output_events = [[], []]
    get_events(board=board, extra_time=100, output_events=output_events)
    spike_count(output_events=output_events)
    plot_raster(output_events=output_events)
