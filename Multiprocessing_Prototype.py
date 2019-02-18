#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 17 22:10:41 2019

@author: tim
"""

from multiprocessing import Process, Queue, Event
import random
import time

class Preloader(Process):
    def __init__(self,event, queue):
        super(Preloader, self).__init__()

        self.e = event
        self.q = queue

    def run(self):
        while True:
            event_is_set = self.e.wait()
            if event_is_set:
                time.sleep(1)
                for i in range(3):
                    new_number = random.randint(0,9)
                    self.q.put(new_number)
                self.e.clear()

class Dataset():
    def __init__(self):
        self.bucket = [1,2,3]

        e = Event()
        self.q = Queue()

        for i in self.bucket:
            self.q.put(i)

        self.Preloader = Preloader(e, self.q)
        self.Preloader.start()

    def __getitem__(self, i):
        self.check_bucket()
        time.sleep(0.5)
        return self.q.get()

    def check_bucket(self):
        if self.q.qsize() < 2:
            self.Preloader.e.set()



d = Dataset()

s = time.time()
for i in range(20):
    print(d[i])
print(time.time() - s)
