#order of function defs
class BaseSGD:
	def __init__(self, loss, penalty='l2', alpha=0.0001, C=1.0,
                 l1_ratio=0.15, fit_intercept=True, n_iter=5, shuffle=False,
                 verbose=0, epsilon=0.1, random_state=None,
                 learning_rate="optimal", eta0=0.0, power_t=0.5,
                 warm_start=False, rho=None):

def fit_binary(est, i, X, y, alpha, C, learning_rate, n_iter,
               pos_weight, neg_weight, sample_weight)

def _partial_fit(self, X, y, alpha, C,
                     loss, learning_rate, n_iter,
                     classes, sample_weight,
		     coef_init, intercept_init):
	 if self.t_ is None: 
		 self._init_t(self.loss_function)
	 if n_classes > 2: 
		 self._fit_multiclass(...)
	 elif n_classes==2:
		 self._fit_binary(...)
	self.t_ += n_iter*n_samples




def _fit(self, X, y, alpha, C, loss, learning_rate,
             coef_init=None, intercept_init=None, class_weight=None,
	     sample_weight=None):
	classes = np.unique(y)
	# Clear iteration count for multiple call to fit.
        self.t_ = None

        self._partial_fit(X, y, alpha, C, loss, learning_rate, self.n_iter,
                          classes, sample_weight, coef_init, intercept_init)

def _fit_binary():
	coef, intercept = fit_binary()
def _fit_multiclass():
	for i in range(len(classes)):
		coef, intercept = fit_binary()






def partial_fit(self, X, y, classes=None, sample_weight=None):
	return self._partial_fit(...)

def fit(self, X, y,...):
	return self._fit(...)

