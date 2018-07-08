#!/usr/bin/env python3

import argparse
import atexit
import json
from prometheus_client.core import GaugeMetricFamily, REGISTRY
from prometheus_client.exposition import start_http_server
from py3nvml import py3nvml as nvml
import socket
from sys import exit
import textwrap
import time
from typing import Dict, Generator
import urllib3


class NvidiaCollector(object):
    @staticmethod
    def call(nvml_getter_name, handle, arg=None):
        try:
            f = getattr(nvml, 'nvmlDeviceGet' + nvml_getter_name)
            return f(handle) if arg is None else f(handle, arg)
        except nvml.NVMLError:
            return 0.0

    @staticmethod
    def collect() -> Generator:
        gpu_utilization = GaugeMetricFamily('nvidia_gpu_utilization', 'GPU Utilization', labels=['gpu_id', 'type'])
        clock_speed = GaugeMetricFamily('nvidia_clock_speed', 'Clock Speed', labels=['gpu_id', 'type'])
        power_usage = GaugeMetricFamily('nvidia_power_usage', 'Power Usage', labels=['gpu_id', 'type'])
        memory_usage = GaugeMetricFamily('nvidia_memory_usage', 'Memory Usage', labels=['gpu_id', 'type'])
        bar1_memory_usage = GaugeMetricFamily('nvidia_bar1_memory_usage', 'BAR1 Memory Usage', labels=['gpu_id', 'type'])
        temperature = GaugeMetricFamily('nvidia_temperature', 'Temperature', labels=['gpu_id', 'type'])
        fan_speed = GaugeMetricFamily('nvidia_fan_speed', 'Fan Speed', labels=['gpu_id'])

        gpu_handles = [(i, nvml.nvmlDeviceGetHandleByIndex(i)) for i in range(nvml.nvmlDeviceGetCount())]
        for (i, handle) in gpu_handles:
            #gpu_id = nvml.nvmlDeviceGetUUID(handle)
            gpu_id = str(nvml.nvmlDeviceGetPciInfo(handle).bus)
            if gpu_id == "10":
                gpu_id = "A"
            if gpu_id == "11":
                gpu_id = "B"
            if gpu_id == "12":
                gpu_id = "C"
            if gpu_id == "13":
                gpu_id = "D"
            if gpu_id == "14":
                gpu_id = "E"
            if gpu_id == "15":
                gpu_id = "F"

            # GPU Utilization
            nvml_gpu_utilization = NvidiaCollector.call('UtilizationRates', handle)
            gpu_utilization.add_metric([gpu_id, 'gpu'], nvml_gpu_utilization.gpu)
            gpu_utilization.add_metric([gpu_id, 'memory'], nvml_gpu_utilization.memory)
            # Clock Speed
            clock_speed.add_metric([gpu_id, 'core'], NvidiaCollector.call('ClockInfo', handle, nvml.NVML_CLOCK_COUNT))
            clock_speed.add_metric([gpu_id, 'memory'], NvidiaCollector.call('ClockInfo', handle, nvml.NVML_CLOCK_MEM))
            clock_speed.add_metric([gpu_id, 'max_core'], NvidiaCollector.call('MaxClockInfo', handle, nvml.NVML_CLOCK_COUNT))
            clock_speed.add_metric([gpu_id, 'max_memory'], NvidiaCollector.call('MaxClockInfo', handle, nvml.NVML_CLOCK_MEM))
            # Power Usage
            power_usage.add_metric([gpu_id, 'usage'], NvidiaCollector.call('PowerUsage', handle))
            power_usage.add_metric([gpu_id, 'min_limit'], NvidiaCollector.call('PowerManagementLimitConstraints', handle)[0])
            power_usage.add_metric([gpu_id, 'max_limit'], NvidiaCollector.call('PowerManagementLimitConstraints', handle)[1])
            power_usage.add_metric([gpu_id, 'limit'], NvidiaCollector.call('PowerManagementLimit', handle))
            power_usage.add_metric([gpu_id, 'default_limit'], NvidiaCollector.call('PowerManagementDefaultLimit', handle))
            power_usage.add_metric([gpu_id, 'enforced_limit'], NvidiaCollector.call('EnforcedPowerLimit', handle))
            # Memory Usage
            nvml_memory_usage = NvidiaCollector.call('MemoryInfo', handle)
            memory_usage.add_metric([gpu_id, 'used'], nvml_memory_usage.used)
            memory_usage.add_metric([gpu_id, 'free'], nvml_memory_usage.free)
            memory_usage.add_metric([gpu_id, 'total'], nvml_memory_usage.total)
            # BAR1 Memory Usage
            nvml_bar1_memory_usage = NvidiaCollector.call('BAR1MemoryInfo', handle)
            bar1_memory_usage.add_metric([gpu_id, 'used'], nvml_bar1_memory_usage.bar1Used)
            bar1_memory_usage.add_metric([gpu_id, 'free'], nvml_bar1_memory_usage.bar1Free)
            bar1_memory_usage.add_metric([gpu_id, 'total'], nvml_bar1_memory_usage.bar1Total)
            # Temperature
            temperature.add_metric([gpu_id, 'current'], NvidiaCollector.call('Temperature', handle, nvml.NVML_TEMPERATURE_GPU))
            temperature.add_metric([gpu_id, 'slowdown_threshold'], NvidiaCollector.call('TemperatureThreshold', handle, nvml.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN))
            temperature.add_metric([gpu_id, 'shutdown_threshold'], NvidiaCollector.call('TemperatureThreshold', handle, nvml.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN))
            # Fan Speed
            fan_speed.add_metric([gpu_id], NvidiaCollector.call('FanSpeed', handle))

        yield gpu_utilization
        yield clock_speed
        yield power_usage
        yield memory_usage
        yield bar1_memory_usage
        yield temperature
        yield fan_speed

def parse_args() -> Dict:
    parser = argparse.ArgumentParser(
        description='Nvidia GPU, miner and pool statistics exporter for prometheus.io',
        allow_abbrev=False,
        formatter_class=argparse.RawTextHelpFormatter)

    pool_parser = parser.add_argument_group('Pool related arguments')
    miner_parser = parser.add_argument_group('Miner related arguments')

    parser.add_argument(
        '-p', '--port',
        metavar='<port>',
        type=int,
        required=False,
        default=9001,
        help=textwrap.dedent('''\
            The port the exporter listens on for Prometheus queries.
            Default: 9001'''))

    pool_parser.add_argument(
        '-o', '--pool',
        metavar='<name>',
        required=False,
        choices=['flypool'],
        help=textwrap.dedent('''\
            The pool name, in case pool stats are to be collected.
            Currently supported:
              - flypool'''))
    pool_parser.add_argument('-O', '--pool-api-host', metavar='<host>', required=False, help='Pool API host')
    pool_parser.add_argument('-u', '--pool-api-miner', metavar='<miner>', required=False, help='Pool API miner')

    miner_parser.add_argument(
        '-m', '--miner',
        metavar='<name>',
        required=False,
        choices=['dstm', 'bminer'],
        help=textwrap.dedent('''\
            The miner software, in case miner stats are to be collected.
            Currently supported:
              - dstm
              - bminer'''))
    miner_parser.add_argument('-H', '--miner-api-host', metavar='<host>', required=False, help='Miner API host')
    miner_parser.add_argument('-P', '--miner-api-port', metavar='<port>', type=int, required=False, help='Miner API port')

    args = parser.parse_args()

    if len(tuple(filter(None.__ne__, (args.pool, args.pool_api_host, args.pool_api_miner)))) not in (0, 3):
        parser.error('--pool requires --pool-api-host and --pool-api-miner.')
    if len(tuple(filter(None.__ne__, (args.miner, args.miner_api_host, args.miner_api_port)))) not in (0, 3):
        parser.error('--miner requires --miner-api-host and --miner-api-port.')

    return vars(args)

def main():
    args = parse_args()

    urllib3.disable_warnings()
    nvml.nvmlInit()
    atexit.register(nvml.nvmlShutdown)
    REGISTRY.register(NvidiaCollector())
 
    print('Starting exporter...')
    try:
        start_http_server(args['port'])
        while True:
            time.sleep(60)  # 1 query per minute so we don't reach API request limits
    except KeyboardInterrupt:
        print('Exiting...')
        exit(0)


if __name__ == '__main__':
    main()
