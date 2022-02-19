import time
import torch
import numpy as np


class TorchTimeMeter:
    '''
    Utility class for measuring iteration times
    '''

    def __init__(self, enabled=True, name: str="perf", mode="mean"):
        self.enabled = enabled
        self.name = name
        self.mode = mode

        self.past_times = {}
        self.current_begins = {}

    def keys(self):
        return self.past_times.keys()

    def start(self, key: str):
        '''
        Starts the current timing
        :return:
        '''
        if not self.enabled:
            return

        if not key in self.keys():
            self.past_times[key] = []

        # Wait for the gpu to finish
        torch.cuda.synchronize()
        self.current_begins[key] = time.perf_counter()

    def end(self, key: str):
        '''
        Ends the current timing
        :return:
        '''

        if not self.enabled:
            return

        # Do noting if the current key has not been initialized
        if not key in self.keys():
            return

        # Wait for the gpu to finish
        torch.cuda.synchronize()
        current_time = time.perf_counter() - self.current_begins[key]
        self.past_times[key].append(current_time)

    def get_time(self, key: str):
        '''
        Computes the average time in seconds of all the timings performed from the last call to this method
        :return:
        '''

        if not self.enabled:
            return 0.0

        if self.mode == "mean":
            time = float(np.asarray(self.past_times[key]).mean())
        elif self.mode == "sum":
            time = float(np.asarray(self.past_times[key]).sum())
        else:
            raise Exception(f"Unknown timer mode {self.mode}")

        del self.past_times[key]

        return time

    def print_summary(self):
        '''
        Prints a summary of the measured timings
        :return:
        '''

        if not self.enabled:
            return

        all_times = {}
        total_time = 0.0

        for timer_key in sorted(self.keys()):
            current_time = self.get_time(timer_key)
            all_times[timer_key] = current_time
            total_time += current_time

        for timer_key in sorted(all_times.keys()):
            current_time = all_times[timer_key]
            print(f'{self.name}/{timer_key}: {current_time:.4f} ({current_time/total_time * 100:.1f}%)')