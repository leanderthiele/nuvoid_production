import pickle
import io
import torch

class Unpickler(pickle.Unpickler) :
    """ small utility to load cross-device """

    def __init__ (self, device, *args, **kwargs) :
        self.device = device
        super().__init__(*args, **kwargs)

    def find_class (self, module, name) :
        if module == 'torch.storage' and name == '_load_from_bytes' :
            return lambda b: torch.load(io.BytesIO(b), map_location=self.device)
        else :
            return super().find_class(module, name)

def load_posterior (fname, device, need_posterior=True) :
    """ returns (settings, posterior) """

    settings_str = b''
    with open(fname, 'rb') as f :
        while True :
            # scan until we reach newline, which indicates start of the pickled model
            c = f.read(1)
            if c == b'\n' :
                break
            settings_str += c
        if need_posterior :
            posterior = Unpickler(device, f).load()
        else :
            posterior = None

    if need_posterior :
        # fix some stuff
        posterior._device = device
        posterior.potential_fn.device = device

    settings = eval(settings_str)

    return settings, posterior
