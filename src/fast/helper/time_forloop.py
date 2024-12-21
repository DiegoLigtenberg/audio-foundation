import time

class TimeLoop:
    def __init__(self):
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self  # Allow optional chaining if needed

    def __exit__(self, exc_type, exc_value, traceback):
        self.start_time = None  # Clean up state

    def time_iteration(self):
        elapsed_time = time.time() - self.start_time
        print(f"Iteration took {elapsed_time:.4f} seconds",end="\r")
        self.start_time = time.time()  # Reset for the next iteration

class TimeBlock:
    def __init__(self):
        self.start_time = None

    def __enter__(self):
        self.start_time = time.time()
        return self  # Allow optional chaining if needed

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def time_block(self):
        elapsed_time = time.time() - self.start_time
        print(f"Iteration took {elapsed_time:.4f} seconds",end="\r")
        self.start_time = time.time()  # Reset for the next iteration


if __name__ == "__main__":
    # Usage
    a = 10
    for i in range(a):
        with TimeLoop():  # Time the iteration
            # Your iteration logic goes here
            print(f"Processing item {i}")
            # Simulate some work
            time.sleep(0.1)  # Simulated work (replace with actual code)
