from typing import Dict, Optional, Union
import numpy as np
from transformers import AutoTokenizer
import torch
import torch.nn as nn

class detectorParent:
    def __init__(self,
                # data specifications
                data_ref: Optional[Union[np.ndarray, list, None]] = None,
                data_h0: Optional[Union[np.ndarray, list, None]] = None,
                data_h1: Optional[Union[np.ndarray, list]] = None, # "data" in sample_data_gradual
                sample_dict: Optional[Dict] = None,

                # detector specifications
                test: Union["MMD", "LSDD"] = "MMD",
                sample_size: int = 500, 
                windows: Optional[int] = 10,
                drift_type: Optional[Union["Sudden", "Gradual"]] = "Sudden",
                SBERT_model: str = 'bert-base-uncased',   
                transformation: Union["UMAP", "UAE", None] = None,
                pval_thresh: int = .05,
                dist_thresh: int = .0009,

                # other functions
                plot: bool = True,

                # general Alibi Detector parameters
                emb_type: Optional[Union['pooler_output', 'last_hidden_state', 'hidden_state', 'hidden_state_cls']] = 'hidden_state',
                n_layers: Optional[int] = 9,
                max_len: Optional[int] = 200,
                enc_dim: Optional[int] = 32,
                batch_size: Optional[int] = 32,
                tokenizer_size: Optional[int] = 3, # keep it small else computer breaks down 

                # Alibi Detectors: learned kernel detector parameters 
                proj = nn.Sequential(),

                # Alibi Detectors: online detector parameters
                ert: Optional[int] = 50, 
                window_size: Optional[int] = 10, 
                n_runs: Optional[int] = 3, 
                n_bootstraps: Optional[int]= 250,

                # Alibi Detectors: context detector parameters
                context: Optional[Union['subpopulation', None]] = None
                 ):
        """
        In this class, we define the base arguments and parameters that are required by Alibi
        detectors. Not all of these parameters are used by each detector. 
        
        Args
        ----------
        data_ref : np.ndarray, list
            Dataset on which the original model is trained (ex: training dataset). We flag a drift 
            with a reference to the distribution of this dataset. 

        data_h0 :  np.ndarray, list 
            This is an optional dataset that we can use as a sanity check for the efficacy of a drift
            detector. Generally, we use the same dataset as data_ref (or a stream that comes soon after).
            The lack of drift in data_h0 (with data_ref as our reference) is the necessary 
            condition to decide the robustness of the drift detection method

        data_h1: np.ndarray, list
            This is the principal dataset on which we might see a drift (ex. deployment data). 
            It can be just one sample (for sudden drifts) or stream of samples (for gradual drifts).
            Often, for pipelines, datasets come in batches, and each new batch can then be updated
            to the new data_h1.
        
        sample_dict: dict
            Dictionary with samples for reference and comparison data (or streams of comparison data).
            The user can directly input the dictionary as our dataset source if they would prefer to 
            organize the data on their own. 

        sample_size: int
            This parameter decides the number of samples from each of the above 3 datasets that we would
            like to work with. For instance, if the entire training data is 100K sentences, we can use
            a sample_size = 500 to randomly sample 500 of those sentences. 
        
        test: str
            Here, we specify the kind of drift detection test we want (KS, KLD, JSD, MMD, LSDD).
            Each of them is described in greater detail in the README.md.  

        drift_type: str
            Drifts can vary depending on the time horizan and frequency at which we try to detect
            them. This parameter asks the user to specify the type of drift ("Sudden", "Gradual", 
            "Online"). The details of each are in README.md
        
        plot: bool
            This parameter asks the user if they wish to see some of the plots of the results
            from the drift detection. Not every detector will result in relevant plot.

        windows: int 
            This parameter is relevant for gradual drifts and helps break down the data into a 
            certain number of buckets. These buckets can act like "batches" or "data streams".
            The idea behind this approach is that we are trying to localize drifts to a certain 
            time frame and check for consistencies (or lack thereof) in detection. 
            If data_h1 has 100K data points, and if we wish to detect drifts
            gradually over time, a proxy approach would be to break the data in sets of 5K points
            and then randomly sample from each set separately. 
        
        SBERT_model: str 
            This parameter is specific to the SBERT embedding models. If we choose to work with SBERT,
            we can specify the type of SBERT embedding out here. Ex. 'bert-base-uncased'  

        ert: int (optional)
            Expected Run Time before a drift is detected. Alibi detect uses this approach for it's 
            online drift detectors. If the average ERT for the reference data is significantly higher
            than the average run time for the drifted data, that might indicate a possible drift. 
        
        window_size: int
            This parameter is used within Alibi's online detectors. 
            It specifies the number of datapoints to include in one window.

        n_run: int
            This parameter is used within Alibi's online detectors and specifies the number of runs
            the detector must perform before we can get an average ERT. 

        n_bootstraps: int
             This parameter is used within Alibi's online detectors

        context_type: str
            Context that we wish to ignore
            1) sub-population: if we wish to ignore the relative change in sub-population of certain 
            classes

        Returns
        ----------  
        Nothing
        """
        self.data_ref = data_ref
        self.data_h0  = data_h0
        self.data_h1  = data_h1
        self.sample_dict = sample_dict
        self.test = test
        self.sample_size = sample_size
        self.windows = windows

        self.drift_type = drift_type
        self.SBERT_model = SBERT_model
        self.transformation = transformation
        self.pval_thresh = pval_thresh
        self.dist_thresh = dist_thresh
        self.plot = plot

        self.emb_type = emb_type
        self.n_layers = n_layers
        self.max_len = max_len
        self.enc_dim = enc_dim
        self.tokenizer_size = tokenizer_size
        self.tokenizer = AutoTokenizer.from_pretrained(SBERT_model)
        self.batch_size = batch_size   

        self.proj = proj

        self.ert = ert
        self.window_size = window_size
        self.n_runs = n_runs 
        self.n_bootstraps = n_bootstraps
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.context = context



