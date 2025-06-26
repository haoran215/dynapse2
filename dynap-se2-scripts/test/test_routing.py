import time
import sys
import os

sys.path.append(os.getcwd() + '/..')

from lib.dynapse2_util import *
from lib.dynapse2_spikegen import *
from lib.dynapse2_raster import *
from samna.dynapse2 import *
import random


def test_dc(board, number_of_chips):

    # random.seed(2)
    neurons = range(10)
    synapses = range(10, 11)

    model = board.get_model()
    myConfig = model.get_configuration()

    for source_cores in [[0]]:
        for target_cores in [[2]]:
        # for neurons in [[i] for i in range(15, 241, 15)]:

            print(source_cores)
            print(target_cores)
            print(neurons)

            # model.reset(ResetType.ConfigReset, (1 << number_of_chips) - 1)
            # time.sleep(1)

            model.reset(ResetType.PowerCycle, (1 << number_of_chips) - 1)
            time.sleep(1)

            model.apply_configuration(myConfig)
            time.sleep(1)

            # set neuron parameters
            print("Setting parameters")
            set_parameter(myConfig.chips[0].global_parameters, "R2R_BUFFER_AMPB", 5, 255)
            set_parameter(myConfig.chips[0].global_parameters, "R2R_BUFFER_CCB", 5, 255)
            set_parameter(myConfig.chips[0].shared_parameters01, "LBWR_VB_P", 5, 255)
            set_parameter(myConfig.chips[0].shared_parameters23, "LBWR_VB_P", 5, 255)
            set_parameter(myConfig.chips[0].shared_parameters01, "PG_BUF_N", 1, 50)
            set_parameter(myConfig.chips[0].shared_parameters23, "PG_BUF_N", 1, 50)
            for h in range(number_of_chips):
                for c in range(4):
                #     # set_parameter(myConfig.chips[h].cores[c].parameters, "SYPD_DLY0_P", 1, 255)
                #     # set_parameter(myConfig.chips[h].cores[c].parameters, "SYPD_DLY1_P", 1, 255)
                #     # set_parameter(myConfig.chips[h].cores[c].parameters, "SYPD_DLY2_P", 1, 255)
                #     # set_parameter(myConfig.chips[h].cores[c].parameters, "DEAM_ITAU_P", 4, 255)
                #     # set_parameter(myConfig.chips[h].cores[c].parameters, "DENM_ETAU_P", 4, 255)
                #     # set_parameter(myConfig.chips[h].cores[c].parameters, "DENM_ITAU_P", 4, 255)
                #     # set_parameter(myConfig.chips[h].cores[c].parameters, "DEGA_ITAU_P", 4, 255)
                #     # set_parameter(myConfig.chips[h].cores[c].parameters, "DESC_ITAU_P", 4, 255)
                #     # set_parameter(myConfig.chips[h].cores[c].parameters, "SOHO_VB_P", 1, 255)
                #     # set_parameter(myConfig.chips[h].cores[c].parameters, "SOAD_PWTAU_N", 1, 255)
                #     # set_parameter(myConfig.chips[h].cores[c].parameters, "SOAD_TAU_P", 1, 255)
                #     # set_parameter(myConfig.chips[h].cores[c].parameters, "SOCA_TAU_P", 1, 255)
                    set_parameter(myConfig.chips[h].cores[c].parameters, "SYSA_VRES_N", 5, 254)
                    set_parameter(myConfig.chips[h].cores[c].parameters, "SYSA_VB_P", 5, 254)
                for c in source_cores + target_cores:
                    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_LEAK_N", 0, 255)
                    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_SPKTHR_P", 3, 255)
                    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_GAIN_N", 3, 255)
                    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_REFR_N", 1, 255)
                for c in source_cores:
                    set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_DC_P", 1, 160)
                for c in target_cores:
                    set_parameter(myConfig.chips[h].cores[c].parameters, "SYAM_W0_P", 5, 200)
                    set_parameter(myConfig.chips[h].cores[c].parameters, "SYAM_W1_P", 0, 0)
                    set_parameter(myConfig.chips[h].cores[c].parameters, "DEAM_ETAU_P", 3, 40)
                    set_parameter(myConfig.chips[h].cores[c].parameters, "DEAM_EGAIN_P", 4, 80)
                    set_parameter(myConfig.chips[h].cores[c].parameters, "SYPD_EXT_N", 3, 255)
                    # set_parameter(myConfig.chips[h].cores[c].parameters, "SYPD_DLY0_P", 3, 255)
                    # set_parameter(myConfig.chips[h].cores[c].parameters, "SYPD_DLY1_P", 3, 255)
                    # set_parameter(myConfig.chips[h].cores[c].parameters, "SYPD_DLY2_P", 3, 255)

            model.apply_configuration(myConfig)
            time.sleep(1)

            # set neurons to monitor
            print("Setting monitors")
            for h in range(number_of_chips):
                for c in source_cores + target_cores:
                    myConfig.chips[h].cores[c].neuron_monitoring_on = True
                    myConfig.chips[h].cores[c].monitored_neuron = neurons[-1]  # monitor neuron 3 on each core
                    myConfig.chips[0].cores[target_cores[-1]].enable_pulse_extender_monitor1 = True
            model.apply_configuration(myConfig)
            time.sleep(1)

            # for i in range(3):
            #     set_parameter(myConfig.chips[0].sadc_group_parameters01[i], "NCCF_PWLK_P", 5, 255)
            #     set_parameter(myConfig.chips[0].sadc_group_parameters23[i], "NCCF_PWLK_P", 5, 255)
            #     set_parameter(myConfig.chips[0].sadc_group_parameters01[i], "NCCF_HYS_P", 0, 0)
            #     set_parameter(myConfig.chips[0].sadc_group_parameters23[i], "NCCF_HYS_P", 0, 0)
            #     set_parameter(myConfig.chips[0].sadc_group_parameters01[i], "NCCF_BIAS_P", 0, 40)
            #     set_parameter(myConfig.chips[0].sadc_group_parameters23[i], "NCCF_BIAS_P", 0, 40)
            #     set_parameter(myConfig.chips[0].sadc_group_parameters01[i], "NCCF_REF_L_V", 0, 100)
            #     set_parameter(myConfig.chips[0].sadc_group_parameters23[i], "NCCF_REF_L_V", 0, 100)
            #     set_parameter(myConfig.chips[0].sadc_group_parameters01[i], "NCCF_REF_H_V", 5, 250)
            #     set_parameter(myConfig.chips[0].sadc_group_parameters23[i], "NCCF_REF_H_V", 5, 250)
            #
            #
            # model.apply_configuration(myConfig)
            # time.sleep(0.1)
            #
            # set_parameter(myConfig.chips[0].shared_parameters01, "NCCF_CAL_OFFBIAS_P", 0, 0)
            # set_parameter(myConfig.chips[0].shared_parameters23, "NCCF_CAL_OFFBIAS_P", 0, 0)
            # set_parameter(myConfig.chips[0].shared_parameters01, "NCCF_CAL_REFBIAS_V", 5, 255)
            # set_parameter(myConfig.chips[0].shared_parameters23, "NCCF_CAL_REFBIAS_V", 5, 255)
            # for core in myConfig.chips[0].cores:
            #     core.sadc_enables.soif_mem = True
            #     core.sadc_enables.soif_refractory = True
            #     core.sadc_enables.soad_dpi = True
            #     core.sadc_enables.soca_dpi = True
            #     core.sadc_enables.deam_edpi = True
            #     core.sadc_enables.deam_idpi = True
            #     core.sadc_enables.denm_edpi = True
            #     core.sadc_enables.denm_idpi = True
            #     core.sadc_enables.dega_idpi = True
            #     core.sadc_enables.desc_idpi = True
            #     core.sadc_enables.sy_w42 = True
            #     core.sadc_enables.sy_w21 = True
            #     core.sadc_enables.soho_sogain = True
            #     core.sadc_enables.soho_degain = True
            # myConfig.chips[0].sadc_enables.nccf_extin_vi_group0_pg1 = True
            # myConfig.chips[0].sadc_enables.nccf_cal_refbias_v_group1_pg1 = True
            # myConfig.chips[0].sadc_enables.nccf_cal_refbias_v_group2_pg1 = True
            # myConfig.chips[0].sadc_enables.nccf_extin_vi_group2_pg1 = True
            # myConfig.chips[0].sadc_enables.nccf_extin_vi_group0_pg0 = True
            # myConfig.chips[0].sadc_enables.nccf_cal_refbias_v_group2_pg0 = True
            # myConfig.chips[0].sadc_enables.nccf_extin_vi_group2_pg0 = True
            # model.apply_configuration(myConfig)
            # time.sleep(1)

            # set neuron latches to get DC input
            print("Setting DC")
            set_dc_latches(config=myConfig, neurons=neurons, cores=source_cores, chips=range(number_of_chips))
            # set_type_latches(config=myConfig, neurons=neurons, cores=range(4), chips=range(number_of_chips))
            # set_type_latches(config=myConfig, neurons=neurons, cores=range(1), chips=range(number_of_chips))
            model.apply_configuration(myConfig)
            time.sleep(1)


            # set CAM -- synapses
            for c in target_cores:
                for i in range(256):
                    cam_exc = [Dynapse2Synapse() for _ in range(64)]
                    for j in range(64):
                        if i in neurons and j in synapses:
                            # if j != 7:
                            weights = [True, False, False, False]
                            cam_exc[j].weight = weights
                            # cam_exc[j].tag = random.choice(source_cores) * 256 + random.choice(neurons)
                            cam_exc[j].tag = 1024 + source_cores[0] * 256 + i
                            cam_exc[j].dendrite = Dendrite.ampa
                        else:
                            # if j != 7:
                            weights = [False, True, False, False]
                            cam_exc[j].weight = weights
                            cam_exc[j].tag = 0
                            cam_exc[j].dendrite = Dendrite.none
                    myConfig.chips[0].cores[c].neurons[i].synapses = cam_exc
            model.apply_configuration(myConfig)
            time.sleep(1)

            # set SRAM -- axons
            print("Setting SRAMs")
            clear_srams(config=myConfig, neurons=neurons, source_cores=source_cores, cores=target_cores,
                        chips=range(number_of_chips), all_to_all=True)
            model.apply_configuration(myConfig)
            time.sleep(1)


            print("\nAll configurations done!\n")
            send_events(board=board, events=[], min_delay=10000000)
            output_events = [[], []]
            get_events(board=board, extra_time=100, output_events=output_events)

            for c in source_cores:
                set_parameter(myConfig.chips[0].cores[c].parameters, "SOIF_DC_P", 0, 0)
                for n in neurons:
                    myConfig.chips[0].cores[c].neurons[n].latch_so_dc = False
            for c in source_cores + target_cores:
                for n in neurons:
                    myConfig.chips[0].cores[c].neurons[n].destinations = [Dynapse2Destination()] * 4
            model.apply_configuration(myConfig)
            time.sleep(1)

            while len(board.read_events()) > 0:
                pass

            spike_count(output_events=output_events)
            plot_raster(output_events=output_events)
            # while True:
            #     pass

    # model.set_sadc_sample_period_ms(100)
    # board.enable_output(BusId.sADC, True)
    # board.enable_output(BusId.W, True)

    # print("\nAll spikes sent!\n")

    # for _ in range(100):
    #     # sadcValues = model.get_sadc_values(0)
    #     time.sleep(0.1)
    #     # for _ in range(300):
    #     #     time.sleep(0.00001)
    #     # for ev in board.read_events():
    #     #     if ev.event.tag < 1024:
    #     #         print(ev.event.tag)
    #     # pass
    #     # print(f"{get_sadc_description(42)}: {sadcValues[42]}; {get_sadc_description(57)}: {sadcValues[57]}")
    #     # print(f"{sadcValues[42]}, {sadcValues[57]}")
    #     sadc_values = model.get_sadc_values(0)
    #     for i, v in enumerate(sadc_values):
    #         print('%30s: %d' % (get_sadc_description(i), v))

    # set_type_latches(config=myConfig, neurons=neurons, cores=range(1), chips=range(number_of_chips))
    # model.apply_configuration(myConfig)
    # time.sleep(1)

    # for key in myConfig.chips[0].cores[0].parameters.keys():
    # for key in myConfig.chips[0].shared_parameters01.keys():
    #     print(key)
    #     set_parameter(myConfig.chips[0].shared_parameters01, key, 5, 255)
    #     set_parameter(myConfig.chips[0].shared_parameters23, key, 5, 255)
    #     model.apply_configuration(myConfig)
    #     time.sleep(5)

    # for h in range(number_of_chips):
    #     for c in range(4):
    #         set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_DC_P", 0, 0)
    #
    # model.apply_configuration(myConfig)
    # time.sleep(1)


def initialize(board, number_of_chips):
    model = board.get_model()
    myConfig = model.get_configuration()

    model.reset(ResetType.PowerCycle, (1 << number_of_chips) - 1)
    time.sleep(1)

    model.apply_configuration(myConfig)
    time.sleep(5)

    # set neuron parameters
    print("Setting parameters")
    set_parameter(myConfig.chips[0].global_parameters, "R2R_BUFFER_AMPB", 5, 255)
    set_parameter(myConfig.chips[0].global_parameters, "R2R_BUFFER_CCB", 5, 255)
    set_parameter(myConfig.chips[0].shared_parameters01, "LBWR_VB_P", 5, 255)
    set_parameter(myConfig.chips[0].shared_parameters23, "LBWR_VB_P", 5, 255)
    set_parameter(myConfig.chips[0].shared_parameters01, "PG_BUF_N", 1, 50)
    set_parameter(myConfig.chips[0].shared_parameters23, "PG_BUF_N", 1, 50)
    for h in range(number_of_chips):
        for c in range(4):
            set_parameter(myConfig.chips[h].cores[c].parameters, "SYSA_VRES_N", 5, 254)
            set_parameter(myConfig.chips[h].cores[c].parameters, "SYSA_VB_P", 5, 254)
            set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_LEAK_N", 0, 255)
            set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_SPKTHR_P", 4, 80)
            set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_GAIN_N", 3, 255)
            set_parameter(myConfig.chips[h].cores[c].parameters, "SOIF_REFR_N", 4, 255)
            set_parameter(myConfig.chips[h].cores[c].parameters, "SYAM_W0_P", 5, 200)
            set_parameter(myConfig.chips[h].cores[c].parameters, "SYAM_W1_P", 5, 200)
            set_parameter(myConfig.chips[h].cores[c].parameters, "SYAM_W2_P", 5, 200)
            set_parameter(myConfig.chips[h].cores[c].parameters, "SYAM_W3_P", 5, 200)
            set_parameter(myConfig.chips[h].cores[c].parameters, "DEAM_ETAU_P", 4, 40)
            set_parameter(myConfig.chips[h].cores[c].parameters, "DEAM_EGAIN_P", 5, 200)
            set_parameter(myConfig.chips[h].cores[c].parameters, "SYPD_EXT_N", 4, 40)
    model.apply_configuration(myConfig)
    time.sleep(1)

    return model, myConfig


def sweep_cams(board, model, myConfig, cores, neurons, synapse, weight, tag, cam_list, px_list, double=True, plot=False):

    # print(f"cores {cores}, neurons {neurons}, synapse {synapse}, weight {weight}, tag {tag}")
    # for c in cores:
    #     for i in range(256):
    #         myConfig.chips[0].cores[c].neurons[i].latch_soif_kill = False

    # set CAM -- synapses
    for c in range(4):
        for i in range(256):
            cam_exc = [Dynapse2Synapse() for _ in range(64)]
            if c in cores and i in neurons[c]:
                cam_exc[synapse[c][i]].weight = [_ == weight[c][i] for _ in range(4)]
                cam_exc[synapse[c][i]].tag = tag[c][i]
                cam_exc[synapse[c][i]].dendrite = Dendrite.ampa
            myConfig.chips[0].cores[c].neurons[i].synapses = cam_exc
    model.apply_configuration(myConfig)
    time.sleep(.1)

    # visited_tags = np.zeros(2048, dtype=bool)
    input_events = []
    ts = get_fpga_time(board=board) + 100000
    for c in cores:
        for i in neurons[c]:
            input_events += [AerConstructor(DestinationConstructor(tag=tag[c][i],
                                                                   core=[_ in cores for _ in range(4)]).destination,
                                            ts).aer]
            if double:
                input_events += [AerConstructor(DestinationConstructor(tag=tag[c][i],
                                                                       core=[_ in cores for _ in range(4)]).destination,
                                                ts + 800).aer]
            # visited_tags[tag[c][i]] = True
            ts += 1000
    # print(np.nonzero(1 - visited_tags)[0][0])
    # input_events += [AerConstructor(DestinationConstructor(tag=np.nonzero(1 - visited_tags)[0][0],
    #                                                        core=[_ in cores for _ in range(4)]).destination,
    #                                 ts).aer]

    send_events(board=board, events=input_events, min_delay=0)

    # time.sleep((len(cores) * 256 * 1000 + 100000) / 1000000)
    # print("killing all neurons")
    # for c in cores:
    #     for i in range(256):
    #         myConfig.chips[0].cores[c].neurons[i].latch_soif_kill = True
    # model.apply_configuration(myConfig)
    # time.sleep(.1)

    output_events = [[], []]
    get_events(board=board, extra_time=100, output_events=output_events)
    if plot:
        plot_raster(output_events=output_events)

    counts = spike_count(output_events=output_events, show=True)
    for c in cores:
        for i in range(256):
            s = synapse[c][i]
            if counts[c * 256 + i] == 0:
                cam_list += [[c, i, s, myConfig.chips[0].cores[c].neurons[i].synapses[s].tag]]
            elif counts[c * 256 + i] > 30:
                px_list += [[c, i, s, myConfig.chips[0].cores[c].neurons[i].synapses[s].tag]]


def compute_valid_neurons_and_tags(px_filename):

    data_px = np.loadtxt(px_filename, delimiter=",", dtype=int)

    bad_neuron = np.zeros((4, 256), dtype=bool)
    bad_tag = np.zeros((4, 2048), dtype=bool)
    bad_neuron_tag = np.zeros((4, 256, 2048), dtype=bool)

    for core_neuron_synapse_tag in data_px:
        core = core_neuron_synapse_tag[0]
        neuron = core_neuron_synapse_tag[1]
        tag = core_neuron_synapse_tag[3]
        bad_neuron[core, neuron] = True
        bad_tag[core, tag] = True
        bad_neuron_tag[core, neuron, tag] = True
    good_neuron = 1 - bad_neuron

    # valid_tags = [list(np.nonzero(1 - bad_tag[core, :])[0]) for core in range(4)]
    # return valid_tags

    for i in range(16):
        keep_core = [bool(i & (1 << n)) for n in range(4)]
        kept_cores = np.nonzero(keep_core)[0]
        usable_neurons = np.sum([good_neuron[core] + keep_core[core] * bad_neuron[core] for core in range(4)])
        usable_tags = np.sum(np.prod([np.logical_not(keep_core[core] * bad_tag[core]) for core in range(4)], axis=0))
        if usable_neurons <= usable_tags:
            print(f"keep bad neurons on cores {kept_cores}: {usable_neurons} neurons x {usable_tags} tags")

    valid_cores = lst1 = [int(_) for _ in input("Enter the cores you want to keep the bad neurons: ").split()]
    valid_neurons = [list(np.nonzero(1 - bad_neuron[core] * (core not in valid_cores))[0]) for core in range(4)]
    valid_tags = list(np.nonzero(np.prod(1 - bad_tag[valid_cores, :], axis=0))[0])

    return valid_cores, valid_neurons, valid_tags


def compute_valid_synapses(cam_filename, valid_tags):
    data_cams = np.loadtxt(cam_filename, delimiter=",", dtype=int)
    bad_cams = np.zeros((4, 256, 64), dtype=bool)
    for core_neuron_synapse_tag in data_cams:
        if core_neuron_synapse_tag[3] in valid_tags:
            bad_cams[core_neuron_synapse_tag[0], core_neuron_synapse_tag[1], core_neuron_synapse_tag[2]] = True
    valid_synapses = [[np.nonzero(1 - bad_cams[core, neuron, :])[0] for neuron in range(256)] for core in range(4)]
    return valid_synapses


# def compute_valid_synapses(cam_filename, valid_tags):
#     data_cams = np.loadtxt(cam_filename, delimiter=",", dtype=int)
#     bad_cams = np.zeros((4, 256, 64), dtype=bool)
#     for core_neuron_synapse_tag in data_cams:
#         if core_neuron_synapse_tag[3] in valid_tags:
#             bad_cams[core_neuron_synapse_tag[0], core_neuron_synapse_tag[1], core_neuron_synapse_tag[2]] = True
#     valid_synapses = [[np.nonzero(1 - bad_cams[core, neuron, :])[0] for neuron in valid_neurons[core]] for core in range(4)]
#     return valid_synapses


def test_cams_two_steps(board, number_of_chips, board_name="pink"):

    model, myConfig = initialize(board=board, number_of_chips=number_of_chips)

    px_filename = "bad_pxs_" + board_name + ".csv"
    cam_filename = "bad_cams_" + board_name + ".csv"

    if not os.path.exists(px_filename) or not os.path.exists(cam_filename):

        print("Test all cams first")

        px_list = []
        cam_list = []

        for i in range(4):
            print(f"round {i}")
            synapse_shuffle = np.array([[random.sample(range(64), k=64) for _ in range(256)] for __ in range(4)])
            # synapses_shuffle = np.array([[range(64) for _ in range(256)] for __ in range(4)])
            synapse_round = 0
            clear_srams(config=myConfig, neurons=range(256), source_cores=[], cores=range(4),
                        chips=range(number_of_chips), all_to_all=True, monitor_cam=i)
            model.apply_configuration(myConfig)
            time.sleep(1)
            for j in range(8):
                print(f"trial {j}")
                weight = np.array([[random.randint(0, 3) for _ in range(256)] for __ in range(4)])
                for tag_offset in range(0, 2048, 256):
                    synapse = synapse_shuffle[:, :, synapse_round]
                    tag = [[(__ * 256 + _ + tag_offset) % 2048 for _ in random.sample(range(256), k=256)] for __ in range(4)]
                    synapse_round = synapse_round + 1
                    sweep_cams(board=board, model=model, myConfig=myConfig, cores=random.sample(range(4), k=4),
                               neurons=[random.sample(range(256), k=256) for _ in range(4)],
                               synapse=synapse, weight=weight, tag=tag, cam_list=cam_list, px_list=px_list, plot=False)

        np.savetxt(px_filename, np.asarray(px_list).astype(int), fmt='%i', delimiter=",")
        np.savetxt(cam_filename, np.asarray(cam_list).astype(int), fmt='%i', delimiter=",")

    valid_cores, valid_neurons, valid_tags = compute_valid_neurons_and_tags(px_filename)
    # valid_tags = compute_valid_neurons_and_tags(px_filename)
    valid_synapses = compute_valid_synapses(cam_filename, valid_tags)

    px_list_verify = []
    cam_list_verify = []

    for n in range(16):
        print(f"Trial {n}:")
        valid_tags_test = random.sample(valid_tags, np.sum([len(valid_neurons[_]) for _ in range(4)]))
        tag = np.zeros((4, 256), dtype=int)
        tag_count = 0
        for core in range(4):
            for neuron in valid_neurons[core]:
                tag[core][neuron] = valid_tags_test[tag_count]
                tag_count += 1
        clear_srams(config=myConfig, neurons=range(256), source_cores=[], cores=range(4),
                    chips=range(number_of_chips), all_to_all=True, monitor_cam=random.randint(0, 3))
        model.apply_configuration(myConfig)
        time.sleep(1)
        sweep_cams(board=board, model=model, myConfig=myConfig, cores=random.sample(valid_cores, len(valid_cores)),
                   neurons=[random.sample(valid_neurons[c], len(valid_neurons[c])) for c in range(4)],
                   synapse=[[random.choice(valid_synapses[__][_]) if _ in valid_neurons[__] else 0
                             for _ in range(256)] for __ in range(4)],
                   weight=[random.choices(range(4), k=256) for _ in range(4)],
                   tag=tag, cam_list=cam_list_verify, px_list=px_list_verify, double=False, plot=False)

    np.savetxt("bad_pxs_verify_" + board_name + ".csv", np.asarray(px_list_verify).astype(int), fmt='%i', delimiter=",")
    np.savetxt("bad_cams_verify_" + board_name + ".csv", np.asarray(cam_list_verify).astype(int), fmt='%i', delimiter=",")


def test_cams_directly(board, number_of_chips, board_name="pink"):

    model, myConfig = initialize(board=board, number_of_chips=number_of_chips)

    all_neurons = np.ones((4, 256), dtype=bool)
    all_tags = np.ones((4, 2048), dtype=bool)
    all_synapses_visited = np.zeros((4, 256, 64), dtype=int)
    all_synapses_failed = np.zeros((4, 256, 64), dtype=int)
    cam_list = []
    px_list = []

    synapse_test = [[[random.sample(range(64), k=64) for _ in range(256)] for __ in range(4)] for ___ in range(4)]

    for r in range(4):

        clear_srams(config=myConfig, neurons=range(256), source_cores=[], cores=range(4),
                    chips=range(number_of_chips), all_to_all=True, monitor_cam=r)
        model.apply_configuration(myConfig)
        time.sleep(1)

        for n in range(64):

            for c in random.sample(range(4), k=4):

                if any(all_neurons[c]):

                    tag_test = random.sample(list(np.nonzero(all_tags[c])[0]), k=256)

                    # print(tag_test)
                    for core in range(4):
                        for i in range(256):
                            cam_exc = [Dynapse2Synapse() for _ in range(64)]
                            if c == core:
                                weight = random.randint(0, 3)
                                all_synapses_visited[c, i, synapse_test[r][c][i][n]] += 1
                                cam_exc[synapse_test[r][c][i][n]].weight = [_ == weight for _ in range(4)]
                                cam_exc[synapse_test[r][c][i][n]].tag = tag_test[i]
                                cam_exc[synapse_test[r][c][i][n]].dendrite = Dendrite.ampa
                            myConfig.chips[0].cores[core].neurons[i].synapses = cam_exc
                    model.apply_configuration(myConfig)
                    time.sleep(0.1)

                    input_events = []
                    ts = get_fpga_time(board=board) + 100000
                    for i in range(256):
                        input_events += [AerConstructor(DestinationConstructor(tag=tag_test[i],
                                                                               core=[_ == c for _ in range(4)]).destination,
                                                        ts).aer,
                                         AerConstructor(DestinationConstructor(tag=tag_test[i],
                                                                               core=[_ == c for _ in range(4)]).destination,
                                                        ts + 800).aer]
                        ts += 1000
                    send_events(board=board, events=input_events, min_delay=0)

                    output_events = [[], []]
                    get_events(board=board, extra_time=10000, output_events=output_events)
                    # if plot:
                    # plot_raster(output_events=output_events)

                    counts = spike_count(output_events=output_events, show=True)
                # for c in cores:
                    for i in range(256):
                        synapse = synapse_test[r][c][i][n]
                        tag = myConfig.chips[0].cores[c].neurons[i].synapses[synapse].tag
                        if counts[c * 256 + i] == 0:
                            all_synapses_failed[c, i, synapse] += 1
                            cam_list += [[c, i, synapse, tag]]
                        elif counts[c * 256 + i] > 30:
                            all_neurons[c][i] = False
                            all_tags[c][tag] = False
                            px_list += [[c, i, synapse, tag]]

                    print(f"Core {c}")
                    print(f"good neurons {np.sum(all_neurons[c])}")
                    print(f"good tags {np.sum(all_tags[c])}")
                    print(f"visited synapses {np.sum(all_synapses_visited[c])}")
                    print(f"bad synapses {np.sum(all_synapses_failed[c])}")

    np.savetxt("bad_pxs_direct_" + board_name + ".csv", np.asarray(px_list).astype(int), fmt='%i', delimiter=",")
    np.savetxt("bad_cams_direct_" + board_name + ".csv", np.asarray(cam_list).astype(int), fmt='%i', delimiter=",")




