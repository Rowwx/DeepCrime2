"""Mutate all the things - Main file of the mutation tool

Available functions:
mutate()
    mutates the un-mutated

"""

import os
import argparse
import mutate as m
from utils.logger_setup import setup_logger

def mutate(msc): # msc = 'DC' or 'DPP' or 'DM' 18.01 juan
    m.mutate_model(msc)


if __name__ == '__main__':
    logger = setup_logger(__name__)

    logger.info('DeepCrime started')
    mutate()
    logger.info('DeepCrime finished')
