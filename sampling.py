import numpy as np
from typing import Callable, Dict, Optional, Union

class samplingData:
    def __init__(self, 
                data_ref: Optional[Union[np.ndarray, list, None]],
                data_h0: Optional[Union[np.ndarray, list, None]],
                data_h1: Union[np.ndarray, list],
                drift_type: Union["Sudden", "Gradual"],
                sample_size: int = 500,
                windows: int = 10):
        self.data_ref = data_ref
        self.data_h0 = data_h0
        self.data_h1 = data_h1
        self.sample_size = sample_size
        self.windows = windows
        self.drift_type = drift_type

    def random_sample(self, data: Union[np.ndarray, list]):
        return np.random.choice(data, self.sample_size)

    def sample_data_sudden(self):
        data_dict = {}
        data_dict[0]  = self.random_sample(self.data_ref) # sample_ref
        data_dict[1]  = self.random_sample(self.data_h0) # sample_h0
        data_dict[2]  = self.random_sample(self.data_h1) # sample_h1
        return data_dict

    def sample_data_gradual(self): 
        '''
        1) "data_ref" and "data_h0" are optional parameter in case we already have the 
        reference and h0 data separated from the data windows we wish to track a drift on 
        2) "data" is the gradual pieces of data we need to detect drift on (replacing data_h1)
        3) If "data_ref" and "data_h0" are None then the first 2 windows of "data"
        can be treat as "data_ref" and "data_h0"
        '''
        data_dict = {}
        start = 0
        if self.data_ref is not None:
            data_dict[0] = self.random_sample(self.data_ref)
            start += 1
        if self.data_h0 is not None:
            data_dict[1] = self.random_sample(self.data_h0)
            start += 1 

        # default breakdown          
        for i, ww in zip(range(self.windows), range(start, self.windows)):
            interval_size = len(self.data_h1)//self.windows
            data_window = self.data_h1[interval_size*i: interval_size*(i+1)] 
            data_dict[ww] = self.random_sample(data_window)

        # breakdown based on time stamps
        

        
        return data_dict

    def samples(self):
        if self.drift_type == "Sudden":
            return self.sample_data_sudden()  
        elif self.drift_type == "Gradual":
            if type(self.windows) != int:
                print("Please specify the number of data/time windows for Gradual drift")
            else:
                return self.sample_data_gradual()
        else:
            print("This is not one of the 2 drift types - Sudden or Gradual")  






# Archive

            # sample_dict = sample_data_sudden(data_ref, data_h0, data_h1, sample_size)        
            # sample_ref, sample_h0, sample_h1 = sample_dict[0], sample_dict[1], sample_dict[2]
            # return sample_ref, sample_h0, sample_h1
