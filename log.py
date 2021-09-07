import os
def create_logger(log_filename, rank=0, gpu=0, display=True):
    if rank == 0 and gpu==0:
        f = open(log_filename, 'a')
        counter = [0]
        # this function will still have access to f after create_logger terminates
        def logger(text):
            if display:
                print(text)
            f.write(text + '\n')
            counter[0] += 1
            if counter[0] % 10 == 0:
                f.flush()
                os.fsync(f.fileno())
            # Question: do we need to flush()
        return logger, f.close
    else:
        def logger(text):
            if display and gpu==0:
                print(text)
        def close():
            return
        return logger, close
