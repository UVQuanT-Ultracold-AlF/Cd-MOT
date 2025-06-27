import time

# import resource
# import sys

# if __name__ != "__main__":
#     """Limit max memory usage to half."""
#     soft, hard = resource.getrlimit(resource.RLIMIT_AS)
#     # Convert KiB to bytes, and divide in two to half
#     resource.setrlimit(resource.RLIMIT_AS, (10e6, hard))

# def run(_):
# 	time.sleep(10)
# 	return None

# if __name__ == "__main__":
#     import multiprocessing as mp
#     a = lambda : [int(0)]*100*1024*1024
    
    
#     with mp.Pool(8) as pool:
#         d = pool.map(run,a())

time.sleep(10)
print("a")