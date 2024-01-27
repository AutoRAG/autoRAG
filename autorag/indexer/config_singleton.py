class ConfigSingleton:
    _instance = None

    @staticmethod
    def get_instance():
        if ConfigSingleton._instance is None:
            ConfigSingleton()
        return ConfigSingleton._instance

    def __init__(self):
        if ConfigSingleton._instance is not None:
            raise Exception("This class is a singleton!")
        else:
            ConfigSingleton._instance = self
            # Initialize the configuration attribute here
            self.cfg = None
