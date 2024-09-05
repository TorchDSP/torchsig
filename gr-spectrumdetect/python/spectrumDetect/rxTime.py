import numpy as np
import pmt


class rxTime:

	#-----------------------------------------------------------------------
    def __init__(self,vlen=1):
        self.seconds                = np.uint64(0)      # integer seconds since EPOC as reported by uhd rx_time tag
        self.fracSeconds            = np.float64(0.0)   # fractional seconds since EPOC as reported by uhd rx_time tag
        self.timeTagOffset          = np.uint64(0)      # sample offset corresponding to rx_time tag
        self.timeTagInitialized	    = False             # boolean indicating readiness of time structure for use (uninitialized until first rx_time tag seen)
        self.tunedFreq              = np.float64(0.0)   # tuned center frequency of stream bearing the rx_time tag
        self.freqTagOffset          = np.uint64(0)      # sample offset corresponding to rx_freq tag
        self.freqTagInitialized	    = False             # boolean indicating readiness of freq structure for use (uninitialized until first rx_time tag seen)
        self.sampRate               = np.float64(1.0)   # sample rate of stream bearing the rx_time tag# sample rate of stream bearing the rx_time tag
        self.rateTagOffset          = np.uint64(0)      # sample offset corresponding to rx_rate tag
        self.rateTagInitialized	    = False             # boolean indicating readiness of rate structure for use (uninitialized until first rx_time tag seen)
        self.initialized            = False             # boolean indicating readiness of full structure for use (uninitialized until first rx_time tag seen)

        # offsets above are measured in "items" which can be either samples or vectors
        # therefore, we must know how many samples per item, i.e. vlen (initialized to one until we have more information)
        self.vlen                   = np.uint64(vlen)
    
	#-----------------------------------------------------------------------
    def getNanoSecondsSinceEPOC(self,reference):
        
        if self.initialized:
            samplesSinceRxTimeTag       = np.uint64((reference - self.timeTagOffset)*self.vlen)
            secondsSinceRxTimeTag       = np.float64(samplesSinceRxTimeTag / self.sampRate)
            nanoSecondsSinceRxTimeTag   = np.uint64(secondsSinceRxTimeTag * (1e9))
            nanoSecondsSinceEPOC        = np.uint64(nanoSecondsSinceRxTimeTag + (self.seconds * (1e9)) + (self.fracSeconds * (1e9)))
        else:
            nanoSecondsSinceEPOC        = np.uint64(0)
            print("!! Warning: rxTime.getNanoSecondsSinceEPOC called while rxTime uninitialized !!")

        return nanoSecondsSinceEPOC

	#-----------------------------------------------------------------------
    def isInitialized(self):
        return self.initialized

	#-----------------------------------------------------------------------
    def getTunedFreq(self):
        return self.tunedFreq

	#-----------------------------------------------------------------------
    def getSampRate(self):
        return self.sampRate

	#-----------------------------------------------------------------------
    def processTags(self,tags):
        
        for tag in tags:

            if pmt.symbol_to_string(tag.key) == "rx_time":

                print("\n\trxTime.processTags - New rx_time tag: ", tag.value ," at offset: ", tag.offset)         

                # set (initialize) Rx time reference based on recieved rx_time tag and sample rate
                self.seconds            = np.uint64(pmt.to_uint64(pmt.tuple_ref(pmt.to_tuple(tag.value),0)))
                self.fracSeconds        = np.float64(pmt.to_double(pmt.tuple_ref(pmt.to_tuple(tag.value),1)))
                self.timeTagOffset      = np.uint64(tag.offset)
                self.timeTagInitialized	= True;

                print("\t\tseconds            : " , self.seconds)
                print("\t\tfracSeconds        : " , self.fracSeconds)
                print("\t\ttimeTagOffset      : " , self.timeTagOffset)
                print("\t\tvlen               : " , self.vlen)
                print("\t\ttimeTagInitialized : " , self.timeTagInitialized)

            elif pmt.symbol_to_string(tag.key) == "rx_freq":

                print("\n\trxTime.processTags - New rx_freq tag: " , tag.value , " at offset: ", tag.offset)  

                self.tunedFreq              = np.float64(pmt.to_double(tag.value))
                self.freqTagOffset          = np.uint64(tag.offset)
                self.freqTagInitialized	    = True;

                print("\t\ttunedFreq          : " , self.tunedFreq)
                print("\t\tfreqTagOffset      : " , self.freqTagOffset)
                print("\t\tvlen               : " , self.vlen)
                print("\t\tfreqTagInitialized : " , self.freqTagInitialized)

            elif pmt.symbol_to_string(tag.key) == "rx_rate":

                print("\n\trxTime.processTags - New rx_rate tag: " , tag.value , " at offset: ", tag.offset)  

                self.sampRate               = np.float64(pmt.to_double(tag.value))
                self.rateTagOffset          = np.uint64(tag.offset)
                self.rateTagInitialized	    = True;

                print("\t\tsampRate           : " , self.sampRate)
                print("\t\trateTagOffset      : " , self.rateTagOffset)
                print("\t\tvlen               : " , self.vlen)
                print("\t\trateTagInitialized : " , self.rateTagInitialized)

        if(self.timeTagInitialized and self.freqTagInitialized and self.rateTagInitialized):
            if(self.initialized == False):
                self.initialized = True
                print("\n\trxTime.Time/Freq/Rate structure initialized : " , self.initialized , "\n")
                print("\n\trxTime.Timestamp at initialization: ",self.getNanoSecondsSinceEPOC(0))

        

