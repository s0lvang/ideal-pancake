def init(experiment, _flags):
    global flags
    flags = _flags
    global comet_logger
    comet_logger = experiment
