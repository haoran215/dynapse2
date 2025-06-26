# stack board run with: python3 main.py ./bitfiles/Dynapse2Stack.bit 1
# stack board run with: python3 main.py -d ./bitfiles/Dynapse2DacTestboard.bit

# if self-built samna (no need if installed through pip)
# export PYTHONPATH=$PYTHONPATH:~/Documents/git/samna/build/src

import optparse

import samna

from lib.dynapse2_init import connect, dynapse2board

from test import test_neurons, test_routing, test_synapses, test_homeostasis, test_sadc
from measure import meas_syn_tau
from example import wta, perceptron, stp, stdp, stdp_mp


def main():
    parser = optparse.OptionParser()
    parser.set_usage("Usage: test_sadc.py [options] bitfile [number_of_chips]")
    parser.add_option("-d", "--devboard", action="store_const", const="devboard", dest="device",
                      help="use first XEM7360 found together with DYNAP-SE2 DevBoard")
    parser.add_option("-s", "--stack", action="store_const", const="stack", dest="device",
                      help="use first XEM7310 found together with DYNAP-SE2 Stack board(s)")
    parser.set_defaults(device="stack")
    opts, args = parser.parse_args()

    if len(args) == 2:
        number_of_chips = int(args[1])
    else:
        number_of_chips = 1

    receiver_endpoint = "tcp://0.0.0.0:33335"
    sender_endpoint = "tcp://0.0.0.0:33336"
    node_id = 1
    interpreter_id = 2
    samna_node = samna.SamnaNode(sender_endpoint, receiver_endpoint, node_id)
    remote = connect(opts.device, number_of_chips, samna_node, sender_endpoint, receiver_endpoint, node_id,
                     interpreter_id)
    board = dynapse2board(opts=opts, args=args, remote=remote)

    test_neurons.test_dc(board=board, number_of_chips=number_of_chips)
    # test_routing.test_cams_two_steps(board=board, number_of_chips=number_of_chips)
    # test_synapses.test_weights(board=board, number_of_chips=number_of_chips)
    # test_synapses.test_nmda(board=board, number_of_chips=number_of_chips)
    # test_synapses.test_conductance(board=board, number_of_chips=number_of_chips)
    # test_synapses.test_alpha(board=board, number_of_chips=number_of_chips)
    # test_homeostasis.homeostasis(board=board, number_of_chips=number_of_chips)
    # test_homeostasis.homeostasis_sadc(board=board, number_of_chips=number_of_chips)
    # test_sadc.test_calibration(board=board, number_of_chips=number_of_chips)
    # test_sadc.test_adaptation(board=board, number_of_chips=number_of_chips)
    # test_sadc.test_stp(board=board, number_of_chips=number_of_chips)
    # meas_syn_tau.syn_tau_ampa(board=board, number_of_chips=number_of_chips)
    # meas_syn_tau.syn_tau_gaba(board=board, number_of_chips=number_of_chips)
    # meas_syn_tau.syn_tau_shunt(board=board, number_of_chips=number_of_chips)
    # wta.wta_basic(board=board, number_of_chips=number_of_chips)
    # perceptron.perceptron_xor(board=board, number_of_chips=number_of_chips)
    # stp.short_term_potentiation(board=board, number_of_chips=number_of_chips)
    # stdp.learn_to_divide(board=board, number_of_chips=number_of_chips)
    # stdp_mp.learn_to_divide(board=board, number_of_chips=number_of_chips)


if __name__ == '__main__':
    main()
