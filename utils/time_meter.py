import time
import numpy as np


class TimeMeter:
    '''
    Utility class for measuring iteration times
    '''

    def __init__(self):
        self.past_times = []
        self.current_begin = 0

    def start(self):
        '''
        Starts the current timing
        :return:
        '''

        self.current_begin = time.time()

    def end(self):
        '''
        Ends the current timing
        :return:
        '''

        current_time = time.time() - self.current_begin
        self.past_times.append(current_time)

    def get_average_time(self):
        '''
        Computes the average time in seconds of all the timings performed from the last call to this method
        :return:
        '''

        average_time = float(np.asarray(self.past_times).mean())
        self.past_times = []

        return average_time
