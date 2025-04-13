from prometheus_client import start_http_server

class TrainingMonitor:
    def __init__(self, port=8002):
        """
        Base class for monitoring training metrics.
        :param port: Port to expose Prometheus metrics.
        """
        self.port = port
        start_http_server(self.port)