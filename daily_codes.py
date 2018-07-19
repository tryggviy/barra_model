# This is a document to record all useful codes, so that I can use them later.

# Class used to define customized error
class InitilizationError(Exception):
    def __init__(self,errmsg):
        super(Exception,self).__init__(errmsg)
# Class used to define a dataset feed to ml algorithms        
class Dataset(object):
    
    def __init__(self,setx,sety,name):
        self.setx = setx
        self.sety = sety
        self.name = name
        
        if self.setx.shape[0] != self.sety.shape[0]:
            raise InitilizationError("Instance failed to initialize !")
            
        elif not (isinstance(setx, np.ndarray) and isinstance(sety, np.ndarray)):
            raise InitilizationError("Instance failed to initialize !")
            
        else:
            pass
    
    def __del__(self):
        pass

# Little fucntion used to convert between dates
def date_cvt(astr):
    if len(astr) == 10:
        return astr[0:4]+astr[5:7]+astr[8:10]
    
    elif len(astr) == 8:
        return astr[0:4]+'-'+astr[4:6]+'-'+astr[6:8]
    
    else:
        raise PassingError('The passed argument is not as expected')
		
# Calculate the Exponential Weigted Moving Average for a iterable object
def expma_vec(amatrix, ratio):
    if type(amatrix) == np.ndarray:
        if len(amatrix.shape) == 2:
            if amatrix.shape[1] > 1:
                return amatrix[:,-1] + sum([ratio*amatrix[:,-x-1]*((1-ratio)**(x)) for x in range(1, amatrix.shape[1])])
            
            else:
                return amatrix
            
        elif len(amatrix.shape) == 1:
            return amatrix[-1] + sum([ratio*amatrix[-x-1]*((1-ratio)**(x)) for x in range(1, len(amatrix))])
        
        else:
            pass
        
    elif type(amatrix) == list:
        bmatrix = np.array(amatrix)
        return bmatrix[-1] + sum([ratio*bmatrix[-x-1]*((1-ratio)**(x)) for x in range(1, len(bmatrix))])
    
    else:
        pass