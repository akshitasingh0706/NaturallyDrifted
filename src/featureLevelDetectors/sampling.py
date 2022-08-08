import numpy as np
from typing import Callable, Dict, Optional, Union

from base import detectorParent

class samplingData(detectorParent):
    def __init__(self, *args, **kwargs):
        detectorParent.__init__(self, *args, **kwargs)

    def random_sample(self, data: Union[np.ndarray, list]):
        return np.random.choice(data, self.sample_size)

    def sample_data_sudden(self):
        data_dict = {}
        data_dict[0]  = np.array(self.random_sample(self.data_ref)) # sample_ref
        data_dict[1]  = np.array(self.random_sample(self.data_h0)) # sample_h0
        data_dict[2]  = np.array(self.random_sample(self.data_h1)) # sample_h1
        return data_dict

    def sample_data_online(self): # need to revisit
        data_dict = {}
        data_dict[0]  = np.array(self.random_sample(self.data_ref)) # sample_ref
        data_dict[1]  = np.array(self.random_sample(self.data_h0)) # sample_h0
        data_dict[2]  = np.array(self.random_sample(self.data_h1)) # sample_h1
        # data_dict[2]  = np.array(self.data_h1) # sample_h1
        return data_dict

    def sample_data_gradual(self): 
        data_dict = {}
        start = 0
        if self.data_ref is not None:
            data_dict[0] = np.array(self.random_sample(self.data_ref))
            start += 1
        if self.data_h0 is not None:
            data_dict[1] = np.array(self.random_sample(self.data_h0))
            start += 1 

        # default breakdown          
        for i, ww in zip(range(self.windows), range(start, self.windows)):
            interval_size = len(self.data_h1)//self.windows
            data_window = self.data_h1[interval_size*i: interval_size*(i+1)] 
            data_dict[ww] = np.array(self.random_sample(data_window))

        # breakdown based on time stamps
        

        
        return data_dict

    def samples(self):
        if self.drift_type == "Sudden":
            return self.sample_data_sudden() 
        elif self.drift_type == "Online":
                return self.sample_data_online() 
        elif self.drift_type == "Gradual":
            if type(self.windows) != int:
                print("Please specify the number of data/time windows for Gradual drift")
            else:
                return self.sample_data_gradual()
        else:
            print("This is not one of the 2 drift types - Sudden or Gradual")  
