import sys
import os

sys.path.append(os.getcwd() + '/..')

from lib.dynapse2_util import *
from lib.dynapse2_spikegen import get_fpga_time, send_events
from lib.dynapse2_raster import *
from lib.dynapse2_obj import *
from samna.dynapse2 import *


def test_weights(board, number_of_chips):

    # your code starts here
    model = board.get_model()
    model.reset(ResetType.PowerCycle, 0b1)
    time.sleep(1)

    # set initial configuration
    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(1)

    # for each core, set the neuron to monitor
    neuron = 0x00
    for c in range(1):
        myConfig.chips[0].cores[c].neuron_monitoring_on = True
        myConfig.chips[0].cores[c].monitored_neuron = neuron
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set dc latches
    set_dc_latches(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # # set neuron parameters
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N", 2, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", 0, 50)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", 3, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 3, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", 0, 1)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', 1, 20)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', 3, 160)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', 4, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', 4, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W2_P', 4, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W3_P', 4, 120)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 3, 100)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set CAM -- synapses
    for c in range(1):
        for i in range(256):
            cam_exc = [Dynapse2Synapse() for _ in range(64)]
            for j in range(64):
                weights = [False, False, False, False]
                weights[j % 4] = True
                cam_exc[j].weight = weights
                if j < 8:
                    cam_exc[j].tag = 1024 + i + (j % 4) * 256
                    cam_exc[j].dendrite = Dendrite.ampa
                else:
                    cam_exc[j].tag = 0
                    cam_exc[j].dendrite = Dendrite.none
            myConfig.chips[0].cores[c].neurons[i].synapses = cam_exc
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set SRAM -- axons
    clear_srams(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    ts = get_fpga_time(board=board) + 100000

    input_events = []
    for i in range(1024):
        input_events += [
            AerConstructor(DestinationConstructor(tag=1024+i,
                                                  core=[True, False, False, False]).destination,
                           ts + i * 10000).aer,
            AerConstructor(DestinationConstructor(tag=1024+i,
                                                  core=[True, False, False, False], x_hop=-7).destination,
                           ts + i * 10000).aer]

    print("\nAll configurations done!\n")
    send_events(board=board, events=input_events, min_delay=100000)
    output_events = [[], []]
    get_events(board=board, extra_time=100, output_events=output_events)
    spike_count(output_events=output_events)
    plot_raster(output_events=output_events)


def test_nmda(board, number_of_chips):
    # your code starts here
    model = board.get_model()
    model.reset(ResetType.PowerCycle, 0b1)
    time.sleep(1)

    # set initial configuration
    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(1)

    # for each core, set the neuron to monitor
    neuron = 0x00
    tag_ampa = 1024
    tag_nmda = 1025

    for c in range(1):
        myConfig.chips[0].cores[c].neuron_monitoring_on = True
        myConfig.chips[0].cores[c].monitored_neuron = neuron
        myConfig.chips[0].cores[c].neurons[neuron].latch_denm_nmda = True
        myConfig.chips[0].cores[c].neurons[neuron].latch_coho_ca_mem = True
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # # set dc latches
    # set_dc_latches(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    # model.apply_configuration(myConfig)
    # time.sleep(0.1)

    # # set neuron parameters
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N", 5, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", 1, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", 2, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 3, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", 0, 1)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', 2, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_ETAU_P', 2, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_EGAIN_P', 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_NMREV_N', 0, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', 3, 140)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', 4, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 3, 100)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set CAM -- synapses
    for c in range(1):
        cam_exc = [Dynapse2Synapse() for _ in range(64)]
        for _ in range(4):
            cam_exc[_].weight = [True, False, False, False]
            cam_exc[_].tag = tag_ampa
            cam_exc[_].dendrite = Dendrite.ampa
        for _ in range(4, 8):
            cam_exc[_].weight = [False, True, False, False]
            cam_exc[_].tag = tag_nmda
            cam_exc[_].dendrite = Dendrite.nmda
        myConfig.chips[0].cores[c].neurons[neuron].synapses = cam_exc
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # # set SRAM -- axons
    # clear_srams(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    # model.apply_configuration(myConfig)
    # time.sleep(0.1)

    print("\nAll configurations done!\n")

    ts = get_fpga_time(board=board) + 100000

    input_events = []

    for _ in range(1000):

        input_events += [
            AerConstructor(DestinationConstructor(tag=tag_ampa,
                                                  core=[True, False, False, False]).destination,
                           ts + 1000).aer,
            AerConstructor(DestinationConstructor(tag=tag_nmda,
                                                  core=[True, False, False, False]).destination,
                           ts + 5000).aer,
            AerConstructor(DestinationConstructor(tag=tag_nmda,
                                                  core=[True, False, False, False]).destination,
                           ts + 101000).aer,
            AerConstructor(DestinationConstructor(tag=tag_ampa,
                                                  core=[True, False, False, False]).destination,
                           ts + 105000).aer]
        ts += 300000

    send_events(board=board, events=input_events, min_delay=0)
    # output_events = [[], []]
    # get_events(board=board, extra_time=100, output_events=output_events)
    # spike_count(output_events=output_events)
    # plot_raster(output_events=output_events)


def test_conductance(board, number_of_chips):
    # your code starts here
    model = board.get_model()
    model.reset(ResetType.PowerCycle, 0b1)
    time.sleep(1)

    # set initial configuration
    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(1)

    # for each core, set the neuron to monitor
    neuron = 0x00
    tag_ampa = 1024
    tag_cond = 1025

    for c in range(1):
        myConfig.chips[0].cores[c].neuron_monitoring_on = True
        myConfig.chips[0].cores[c].monitored_neuron = neuron
        myConfig.chips[0].cores[c].neurons[neuron].latch_de_conductance = True
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # # set dc latches
    # set_dc_latches(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    # model.apply_configuration(myConfig)
    # time.sleep(0.1)

    # # set neuron parameters
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N", 5, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", 1, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", 2, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 3, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", 0, 1)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', 2, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_REV_N', 3, 70)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_ETAU_P', 2, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_EGAIN_P', 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_REV_N', 5, 70)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', 3, 140)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', 3, 140)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 3, 100)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set CAM -- synapses
    for c in range(1):
        cam_exc = [Dynapse2Synapse() for _ in range(64)]
        for _ in range(4):
            cam_exc[_].weight = [True, False, False, False]
            cam_exc[_].tag = tag_ampa
            cam_exc[_].dendrite = Dendrite.ampa
        for _ in range(4, 8):
            cam_exc[_].weight = [False, True, False, False]
            cam_exc[_].tag = tag_cond
            cam_exc[_].dendrite = Dendrite.nmda
        myConfig.chips[0].cores[c].neurons[neuron].synapses = cam_exc
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # # set SRAM -- axons
    # clear_srams(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    # model.apply_configuration(myConfig)
    # time.sleep(0.1)

    print("\nAll configurations done!\n")

    ts = get_fpga_time(board=board) + 100000

    input_events = []

    for _ in range(1000):

        input_events += [
            AerConstructor(DestinationConstructor(tag=tag_ampa,
                                                  core=[True, False, False, False]).destination,
                           ts + 1000).aer,
            AerConstructor(DestinationConstructor(tag=tag_cond,
                                                  core=[True, False, False, False]).destination,
                           ts + 5000).aer,
            AerConstructor(DestinationConstructor(tag=tag_cond,
                                                  core=[True, False, False, False]).destination,
                           ts + 101000).aer,
            AerConstructor(DestinationConstructor(tag=tag_ampa,
                                                  core=[True, False, False, False]).destination,
                           ts + 105000).aer]
        ts += 300000

    send_events(board=board, events=input_events, min_delay=0)
    # output_events = [[], []]
    # get_events(board=board, extra_time=100, output_events=output_events)
    # spike_count(output_events=output_events)
    # plot_raster(output_events=output_events)



def test_alpha(board, number_of_chips):
    # your code starts here
    model = board.get_model()
    model.reset(ResetType.PowerCycle, 0b1)
    time.sleep(1)

    # set initial configuration
    myConfig = model.get_configuration()
    model.apply_configuration(myConfig)
    time.sleep(1)

    # for each core, set the neuron to monitor
    neuron = 0x00
    tag_ampa = 1024
    tag_cond = 1025

    for c in range(1):
        myConfig.chips[0].cores[c].neuron_monitoring_on = True
        myConfig.chips[0].cores[c].monitored_neuron = neuron
        myConfig.chips[0].cores[c].neurons[neuron].latch_deam_alpha = True
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # # set dc latches
    # set_dc_latches(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    # model.apply_configuration(myConfig)
    # time.sleep(0.1)

    # # set neuron parameters
    for c in range(1):
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_GAIN_N", 5, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_LEAK_N", 1, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_REFR_N", 2, 60)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_SPKTHR_P", 3, 254)
        set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", 0, 1)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_ETAU_P', 2, 40)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DENM_EGAIN_P', 3, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ETAU_P', 0, 120)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_EGAIN_P', 1, 80)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_ITAU_P', 0, 140)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'DEAM_IGAIN_P', 1, 100)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W0_P', 3, 75)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYAM_W1_P', 3, 150)
        set_parameter(myConfig.chips[0].cores[c].parameters, 'SYPD_EXT_N', 3, 100)
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # set CAM -- synapses
    for c in range(1):
        cam_exc = [Dynapse2Synapse() for _ in range(64)]
        for _ in range(4):
            cam_exc[_].weight = [True, False, False, False]
            cam_exc[_].tag = tag_ampa
            cam_exc[_].dendrite = Dendrite.ampa
        for _ in range(4, 8):
            cam_exc[_].weight = [False, True, False, False]
            cam_exc[_].tag = tag_cond
            cam_exc[_].dendrite = Dendrite.nmda
        myConfig.chips[0].cores[c].neurons[neuron].synapses = cam_exc
    model.apply_configuration(myConfig)
    time.sleep(0.1)

    # # set SRAM -- axons
    # clear_srams(config=myConfig, neurons=range(256), cores=range(4), chips=range(number_of_chips))
    # model.apply_configuration(myConfig)
    # time.sleep(0.1)

    print("\nAll configurations done!\n")

    ts = get_fpga_time(board=board) + 100000

    input_events = []

    for _ in range(1000):

        input_events += [
            AerConstructor(DestinationConstructor(tag=tag_ampa,
                                                  core=[True, False, False, False]).destination,
                           ts).aer,
            AerConstructor(DestinationConstructor(tag=tag_cond,
                                                  core=[True, False, False, False]).destination,
                           ts + 20000).aer,
            AerConstructor(DestinationConstructor(tag=tag_cond,
                                                  core=[True, False, False, False]).destination,
                           ts + 300000).aer,
            AerConstructor(DestinationConstructor(tag=tag_ampa,
                                                  core=[True, False, False, False]).destination,
                           ts + 320000).aer]
        ts += 700000

    send_events(board=board, events=input_events, min_delay=0)
    # output_events = [[], []]
    # get_events(board=board, extra_time=100, output_events=output_events)
    # spike_count(output_events=output_events)
    # plot_raster(output_events=output_events)