# -*- coding: utf-8 -*-
"""
Pyopencl - GPU devices
"""

import os
os.environ['PYOPENCL_COMPILER_OUTPUT'] = '1'

import pyopencl as cl
import numpy as np

def _create_context():
    """
    Search for a device with OpenCl support, and create device context
    :return: First found device or raise EnvironmentError if none is found
    """
    platforms = cl.get_platforms()  # Select the first platform [0]
    if not platforms:
        raise EnvironmentError('No openCL platform (or driver) available.')

    # Return first found device
    for platform in platforms:
        devices = platform.get_devices()
        if devices:
            return cl.Context([devices[0]])

    raise EnvironmentError('No openCL devices (or driver) available.') 


def create_context():
    platforms = cl.get_platforms()
#    logger.debug('OpenCL platforms: {}'.format(['{}: {}.'.format(platform.get_info(cl.platform_info.VENDOR),
#                                                                platform.get_info(cl.platform_info.NAME))
#                                                for platform in platforms]))
    device_to_use = None
    for device_type, type_str in [(cl.device_type.GPU, 'GPU'), (cl.device_type.CPU, 'CPU')]:
        for platform in platforms:
            devices = platform.get_devices(device_type)
            if len(devices) > 0:
#                logger.debug('OpenCL {} devices in {}: {}.'.format(type_str,
#                                                                   platform.get_info(cl.platform_info.NAME),
#                                                                   [device.get_info(cl.device_info.NAME) for device in devices]))
                if device_to_use is None:
                    device_to_use = devices[0]
#                    logger.info('OpenCL device to use: {} {}'.format(platform.get_info(cl.platform_info.NAME),
#                                                                     devices[0].get_info(cl.device_info.NAME)))
    if device_to_use is None:
        raise Exception('No OpenCL CPU or GPU device found!')
    return cl.Context([device_to_use]) 



def get_devices_by_name(name: str, case_sensitive: bool = False):
    """
    Searches through all devices looking for a partial match for 'name' among the available devices.
    :param name: The string to search for
    :param case_sensitive: If false, different case is ignored when searching
    :return: A list of all devices that is a partial match for the specified name
    """
    if not name:
        raise RuntimeError('Device name must be specified')

    platforms = cl.get_platforms()
    devices = [plat.get_devices(cl.device_type.ALL) for plat in platforms]
    devices = [dev for devices in devices for dev in devices]

    if case_sensitive:
        name_matches = [dev for dev in devices if name in dev.name]
    else:
        name_matches = [dev for dev in devices if name.lower() in dev.name.lower()]

    return name_matches 



def _get_device(self, gpuid):
        """Return GPU devices, context, and queue."""
        all_platforms = cl.get_platforms()
        platform = next((p for p in all_platforms if
                         p.get_devices(device_type=cl.device_type.GPU) != []),
                        None)
        if platform is None:
            raise RuntimeError('No OpenCL GPU device found.')
        my_gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
        context = cl.Context(devices=my_gpu_devices)
        if gpuid > len(my_gpu_devices)-1:
            raise RuntimeError(
                'No device with gpuid {0} (available device IDs: {1}).'.format(
                    gpuid, np.arange(len(my_gpu_devices))))
        queue = cl.CommandQueue(context, my_gpu_devices[gpuid])
        if self.settings['debug']:
            print("Selected Device: ", my_gpu_devices[gpuid].name)
        return my_gpu_devices, context, queue 


platforms = cl.get_platforms()
print(platforms)

devices = [plat.get_devices(cl.device_type.ALL) for plat in platforms]
print(devices)

devices = [dev for devices in devices for dev in devices]
print(devices)