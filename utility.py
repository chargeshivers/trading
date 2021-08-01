from time import sleep

def retry(wait_duration):
    def _retry(f):
        def g(*args):
            while True:
                try:
                    out = f(*args)
                except:
                    sleep(wait_duration)
                    continue
                else:
                    break
            return out
        return g
    return _retry

def logg(f):
    def g(*args):
        out = f(*args)
        print(f.__name__,args, end=' ')
        return out
    return g

nearest_to = lambda xs, x: min(xs, key=lambda _:abs(_-x))
deltas = lambda xs: (a - b for a, b in zip(xs, xs[1:]))

floor = lambda b: lambda x: (x // b) * b
nearest = lambda b: lambda x: floor(b)(x) + (b if x - floor(b)(x) >= b / 2 else 0)