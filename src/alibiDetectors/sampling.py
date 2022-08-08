import numpy as np
from typing import Callable, Dict, Optional, Union

class samplingData:
    def __init__(self, 
                data_ref: Optional[Union[np.ndarray, list, None]],
                data_h0: Optional[Union[np.ndarray, list, None]],
                data_h1: Union[np.ndarray, list],
                drift_type: Union["Sudden", "Gradual", "Online"],
                sample_size: int = 500,
                windows: int = 10):

        """
        This class attempts to make data sampling for drift detection more efficient. 
        
        Args
        ----------
        data_ref : np.ndarray, list
            This is the dataset on which is used as the reference/baseline when detecting drifts. 
            For instance, if our test of choice is KL Divergence, then we will declare a possible
            drift based on whether any other data is close in distribution to data_ref. 
            Generally, the goal is to have all future datasets be as close (in embeddings, distributions)
            to data_ref, which is how we conclude that there is no drift in the dataset.  
            
            data_ref is typically sampled from the "training data". During real world application, 
            this is the data on which the test will be modeled on because this would generally be 
            the only data the user would have access to at that point of time. 

        data_h0 :  np.ndarray, list (optional)
            This is generally the same dataset as data_ref (or a stream that comes soon after).
            We use the lack of drift in data_h0 (with data_ref as our reference) as the necessary 
            condition to decide the robustness of the drift detection method. If the method ends up 
            detecting a drift in data_h0 itself, we know it is most likely not doing a good job. 
            This is because both data_ref and data_h0 are expected to be coming from the same source 
            and hence should result in similar embeddings and distributions. 

            If the user is confident in the efficacy of their drift detection method, then it would be 
            worthwhile to consider change the size of data_ref and data_h0 and then re-evaluate detector
            performance, before proceeding to data_h1. 

        data_h1: np.ndarray, list
            This is the primary dataset on which we can expect to possibly detect a drift. In the real 
            world, this would usually be the dataset we get post model deployment. To test detectors, a
            convenient (but not necessarily the best) practice is to take the test data and use that as
            our proxy for the deployed dataset. 

            Multiple research papers and libraries tend to also use "perturbed" data for their choice of
            data_h1. Perturbations can include corruptions in images (vision data) or introduction of 
            unneccessary words and phrases (text data). This is generally the first step in testing the 
            efficacy of a drift detection method. Once again, if the detectors fails to detect a drift
            on manually perturbed data, then its quite likely it will not be able to detect drifts in 
            the real, deployed data as well. 

            Therefore, for our purposes, we have tried to minimize the use of artifically perturbed data
            and instead rely on test data/data from far away time periods as our data_h1 source. 

        sample_size: int
            This parameter decides the number of samples from each of the above 3 datasets that we would
            like to work with. For instance, if the entire training data is 100K sentences, we can use
            a sample_size = 500 to randomly sample 500 of those sentences. 

        drift_type: str
            As discussed in the README, there are many different types of drifts. Here we specify
            the drift type we are looking for, based on the time/frquency. drift_type asks us 
            to specify whether we want to detect sudden drifts or more gradual drifts. 

        windows: int (optional)
            This decided the number of segments we would like to break the data into. 
            This parameter is only required for gradual/incremental drift detection. 
            For instance, if data_h1 has 100K data points, and if we wish to detect drifts
            gradually over time, a proxy approach would be to break the data in sets of 5K points
            and then randomly sample from each set separately. 
        """
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
        data_dict[0]  = np.array(self.random_sample(self.data_ref)) # sample_ref
        data_dict[1]  = np.array(self.random_sample(self.data_h0)) # sample_h0
        data_dict[2]  = np.array(self.random_sample(self.data_h1)) # sample_h1
        return data_dict

    def sample_data_online(self):
        data_dict = {}
        data_dict[0]  = np.array(self.random_sample(self.data_ref)) # sample_ref
        data_dict[1]  = np.array(self.random_sample(self.data_h0)) # sample_h0
        data_dict[2]  = np.array(self.data_h1) # sample_h1
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
