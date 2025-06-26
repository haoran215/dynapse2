import time
from samna.dynapse2 import *
from lib.dynapse2_obj import *


def get_fpga_time(board):
    while True:
        board.input_interface_write_events(0,
                                           [AerConstructor(
                                               DestinationConstructor(tag=1024,
                                                                      core=[True]*4, x_hop=-1, y_hop=-1).destination,
                                               0).aer])
        for timeout in range(1000):
            evs = board.read_events()
            if len(evs) > 0:
                return evs[-1].timestamp


def send_events(board, events, min_delay=0):
    if len(events) > 0:
        ts = events[-1].timestamp
    else:
        ts = get_fpga_time(board=board)
    board.input_interface_write_events(0, events + [AerConstructor(DestinationConstructor(tag=2047,
                                                                       core=[True]*4, x_hop=-1, y_hop=-1).destination,
                                                ts + min_delay).aer]*32)
    # # can also send through the grid bus directly
    # board.grid_bus_write_events(events + [AerConstructor(DestinationConstructor(tag=2047,
    #                                                                    core=[True]*4, x_hop=-1, y_hop=-1).destination,
    #                                             ts + min_delay).aer]*32)
