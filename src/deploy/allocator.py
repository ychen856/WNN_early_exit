def choose_deployment(meta, device_profile, search_space):
    """
    explore (M_bits, keep_ratio, thresholds)，
    using validation proxy, optimize  accuracy and satisfy SRAM/Latency upper bound
    """
