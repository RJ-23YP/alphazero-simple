class MetricTracker(object): 
    """
    A utility class to track and calculate the average, total, and count
    of a metric during iterations (e.g., training/validation loss).
    """
    def __init__(self):
        self.current_value = 0  # The most recent value added
        self.average = 0        # The running average of all values
        self.total = 0          # The sum of all values
        self.num_samples = 0    # The number of samples added

    def __repr__(self):
        # Represent the average value in scientific notation
        return f'{self.average:.2e}'

    def record(self, value, num=1):
        """
        Update the tracker with a new value and optionally the sample count.

        Args:
            value (float): The new metric value.
            num (int): The number of samples associated with this value (default is 1).
        """
        self.current_value = value
        self.total += value * num
        self.num_samples += num
        self.average = self.total / self.num_samples


class dotdict(dict):
    """
    A dictionary subclass that allows attribute-style access to dictionary keys.
    """
    def __getattr__(self, name):
        # Access dictionary keys as attributes, raising an error if the key doesn't exist
        if name in self:
            return self[name]
        raise AttributeError(f"'dotdict' object has no attribute '{name}'") 
