#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live real-time activity GUI for DYNAPSE2 boards.
Usage:
    from dynapse2_activity_plotter import run_plotting_thread
    # after board is configured and monitoring neurons enabled:
    run_plotting_thread(board, refresh_rate=500, y_range=(0, 2048))
"""
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from threading import Thread
from time import time
from types import SimpleNamespace

class Dynapse2ActivityPlot:
    def __init__(self, board, refresh_rate=500, y_range=None):
        self.board = board
        self.refresh_rate = refresh_rate
        self.raster_y_range = y_range
        self.activity_dt = 0.01  # in seconds

        # Initialize GUI
        self.main = tk.Tk()
        self.main.geometry("800x600+100+50")
        self.main.title("DYNAPSE2 Real-time Activity Monitor")

        # Fetch initial events
        evts = self._get_events()
        evts_n = np.array([[evt.timestamp, evt.neuron_id] for evt in evts]) if evts else np.empty((0, 2))
        if evts_n.size:
            self.raster_x = evts_n[:,0] * 1e-6
            self.raster_y = evts_n[:,1]
            self.start_timestamp = self.raster_x[0]
        else:
            self.raster_x = np.array([0.0])
            self.raster_y = np.array([0])
            self.start_timestamp = 0.0

        # Set up figures
        self.fig, (self.raster_ax, self.activity_ax) = plt.subplots(2, 1, figsize=(8, 6), dpi=100)
        self.raster_plot, = self.raster_ax.plot(self.raster_x, self.raster_y, '|')
        self.raster_ax.set_xlabel("Time (s)")
        self.raster_ax.set_ylabel("Neuron ID")

        self.activity_x, self.activity_y = self._compute_activity(0, self.raster_x, self.start_timestamp,
                                                                 self.raster_x[-1], self.activity_dt)
        self.activity_plot, = self.activity_ax.plot(self.activity_x, self.activity_y)
        self.activity_ax.set_xlabel("Time (s)")
        self.activity_ax.set_ylabel("Core Activity")
        self.fig.tight_layout()

        # Embed in Tk
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=1)

        # Schedule updates
        self.main.after(self.refresh_rate, self._update)
        self.main.mainloop()

    def _get_events(self):
        raw = self.board.read_events()
        evts = []
        for ev in raw:
            if ev.event.y_hop == 0:
                neuron_id = ev.event.tag + (ev.event.x_hop + 6) * 2048
                evts.append(SimpleNamespace(timestamp=ev.timestamp, neuron_id=neuron_id))
        return evts

    def _compute_activity(self, last_val, spiketimes, t_start, t_end, dt):
        x, y = [], []
        t = t_start
        A_prev = last_val
        idx = 0
        n_spikes = len(spiketimes)
        for t in np.arange(t_start + dt, t_end, dt):
            A = A_prev * np.exp(-dt / 1.0)
            while idx < n_spikes and spiketimes[idx] < t:
                A += 1
                idx += 1
            x.append(t)
            y.append(A)
            A_prev = A
        return np.array(x), np.array(y)

    def _update(self):
        evts = self._get_events()
        if evts:
            evts_n = np.array([[evt.timestamp, evt.neuron_id] for evt in evts])
            times = evts_n[:,0] * 1e-6
            # Update activity
            ax, ay = self._compute_activity(self.activity_y[-1], times, self.raster_x[-1], times[-1], self.activity_dt)
            self.activity_x = np.concatenate((self.activity_x, ax))
            self.activity_y = np.concatenate((self.activity_y, ay))
            mask = (self.activity_x[-1] - self.activity_x) < 5
            self.activity_x = self.activity_x[mask]
            self.activity_y = self.activity_y[mask]
            self.activity_plot.set_data(self.activity_x, self.activity_y)
            self.activity_ax.set_xlim(self.activity_x.min(), self.activity_x.max())
            self.activity_ax.set_ylim(0, self.activity_y.max() + 1)

            # Update raster
            rx = times
            ry = evts_n[:,1]
            self.raster_x = np.concatenate((self.raster_x, rx))
            self.raster_y = np.concatenate((self.raster_y, ry))
            mask = (self.raster_x[-1] - self.raster_x) < 5
            self.raster_x = self.raster_x[mask]
            self.raster_y = self.raster_y[mask]
            self.raster_plot.set_data(self.raster_x, self.raster_y)
            self.raster_ax.set_xlim(self.raster_x.min(), self.raster_x.max())
            if self.raster_y_range:
                self.raster_ax.set_ylim(self.raster_y_range)
            else:
                self.raster_ax.set_ylim(self.raster_y.min(), self.raster_y.max())

            self.canvas.draw()

        self.main.after(self.refresh_rate, self._update)


def run_plotting_thread(board, refresh_rate=500, y_range=None):
    """
    Start the real-time plotting GUI in a background thread.
    """
    t = Thread(target=Dynapse2ActivityPlot, args=(board, refresh_rate, y_range), daemon=True)
    t.start()
