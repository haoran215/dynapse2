import itertools
import time
import sys
import os
import matplotlib.pyplot as plt
sys.path.append(os.getcwd() + '/..')

import random

from lib.dynapse2_util import set_parameter, clear_srams
from lib.dynapse2_obj import *
from lib.dynapse2_spikegen import get_fpga_time, send_events
from lib.dynapse2_raster import *

from samna.dynapse2 import *

def plot_raster(output_events):
    """
    Given output_events as a list-of-lists where output_events[i]
    is the list of spike times (in µs) for neuron i, draw a raster.
    """
    neuron_ids = []
    spike_times = []
    for neuron_id, times in enumerate(output_events):
        neuron_ids.extend([neuron_id] * len(times))
        spike_times.extend(times)

    plt.figure(figsize=(10, 6))
    plt.scatter(spike_times, neuron_ids, s=2)
    plt.xlabel("Time (µs)")
    plt.ylabel("Neuron ID")
    plt.title("Neural Activity Raster Plot")
    plt.tight_layout()
    plt.show()

def plot_neural_activity(output_events):
    """
    Plots a raster of neural spike activity.
    Assumes output_events is a list of [neuron_ids, spike_times].
    """
    neuron_ids = []
    spike_times = []
    for i, neuron_spikes in enumerate(output_events):
        for spike in neuron_spikes:
            neuron_ids.append(i)
            spike_times.append(spike)
    plt.figure(figsize=(10, 6))
    plt.scatter(spike_times, neuron_ids, s=2)
    plt.xlabel("Time (us)")
    plt.ylabel("Neuron ID")
    plt.title("Neural Activity Raster Plot")
    plt.show()


def config_latches(myConfig, adaptation):
    # for each core, set the neuron to monitor
    neuron = 3
    for c in range(1):
        myConfig.chips[0].cores[c].neuron_monitoring_on = True
        myConfig.chips[0].cores[c].monitored_neuron = neuron
        for n in range(160):
            myConfig.chips[0].cores[c].neurons[n].latch_so_adaptation = adaptation


def config_parameters(myConfig, delay, stp):
    # set neuron parameters
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N", 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", 0, 50)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", 1, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", 0, 1)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 3, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOAD_PWTAU_N", 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOAD_GAIN_P", 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOAD_TAU_P", 0, 10)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOAD_W_N", 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOAD_CASC_P", 5, 254)

    # set synapse parameters  -- enabled AM and SC
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', 2, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_ETAU_P', 1, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_EGAIN_P', 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DESC_ITAU_P', 1, 160)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DESC_IGAIN_P', 4, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', 4, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', 5, 30)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W2_P', 5, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W3_P', 4, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 3, 200)
        if delay:
            set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_DLY0_P', 0, 1)
            set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_DLY1_P', 5, 254)
            set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_DLY2_P', 0, 20)
        if stp:
            set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAW_STDSTR_N', 0, 1)
            set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_STDW_N', 3, 80)


def config_cams(myConfig, input_group_size, delay, stp):
    # set CAM -- synapses
    for c in range(1):
        for i in range(256):
            cams = [Dynapse2Synapse() for _ in range(64)]
            for j in range(64):
                weights = [False, False, False, False]
                if i < 64 and j < 8:
                    weights[0] = True
                    cams[j].tag = random.randint(1024, 1023 + input_group_size)
                    cams[j].dendrite = Dendrite.ampa
                    cams[j].mismatched_delay = delay
                    cams[j].stp = stp
                elif i < 64 and 8 <= j < 32:
                    weights[1] = True
                    cams[j].tag = random.randint(0, 63)
                    cams[j].dendrite = Dendrite.nmda
                    cams[j].mismatched_delay = delay
                elif i < 64 and 32 <= j < 64:
                    weights[2] = True
                    cams[j].tag = random.randint(160, 191)
                    cams[j].dendrite = Dendrite.shunt
                    cams[j].precise_delay = delay
                elif 64 <= i < 128 and j < 8:
                    weights[0] = True
                    cams[j].tag = random.randint(1024 + input_group_size, 1023 + input_group_size * 2)
                    cams[j].dendrite = Dendrite.ampa
                    cams[j].mismatched_delay = delay
                    cams[j].stp = stp
                elif 64 <= i < 128 and 8 <= j < 32:
                    weights[1] = True
                    cams[j].tag = random.randint(64, 127)
                    cams[j].dendrite = Dendrite.nmda
                    cams[j].mismatched_delay = delay
                elif 64 <= i < 128 and 32 <= j < 64:
                    weights[2] = True
                    cams[j].tag = random.randint(160, 191)
                    cams[j].dendrite = Dendrite.shunt
                    cams[j].precise_delay = delay
                elif 160 <= i < 192 and j < 64:
                    weights[3] = True
                    cams[j].tag = random.randint(0, 127)
                    cams[j].dendrite = Dendrite.nmda
                    cams[j].mismatched_delay = delay
                else:
                    weights[0] = True
                    cams[j].tag = 0
                    cams[j].dendrite = Dendrite.none
                cams[j].weight = weights
            myConfig.chips[0].cores[c].neurons[i].synapses = cams


def generate_events(board, group_size, input_events):
    ts = get_fpga_time(board=board) + 1000000
    for j in list(range(1024, 1024 + group_size + 1, group_size >> 2)) +\
             list(range(1024 + group_size, 1023, - (group_size >> 2))):
        for t in range(ts, ts + 1000000, 100):
            for k in range(j, j + group_size):
                if random.random() < 0.01:
                    input_events += VirtualSpikeConstructor(k, [True, False, False, False], t).spikes
        ts += 1000000


def wta_basic(board, number_of_chips, delay=True, adaptation=False, stp=False):

    # your code starts here
    input_group_size = 16

    model = board.get_model()
    model.reset(ResetType.PowerCycle, 0b1)
    time.sleep(1)

    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(1)

    print("Configuring latches")
    config_latches(myConfig, adaptation=adaptation)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    print("Configuring paramrters")
    config_parameters(myConfig=myConfig, delay=delay, stp=stp)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    print("Configuring cams")
    config_cams(myConfig=myConfig, input_group_size=input_group_size, delay=delay, stp=stp)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    print("configuring srams")
    clear_srams(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips), all_to_all=True)
    model.apply_configuration(myConfig)
    time.sleep(1)

    print("\nAll configurations done!\n")

    input_events = []
    generate_events(board, input_group_size, input_events)
    send_events(board=board, events=input_events, min_delay=100000)
    output_events = [[], []]
    get_events(board=board, extra_time=100, output_events=output_events)
    spike_count(output_events=output_events)
    plot_raster(output_events=output_events)
    plot_neural_activity(output_events)  # Add this line to plot neural activity

