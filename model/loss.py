from model.criterion import WeightedSDR


'''
Not implemented - used torch.functional.mse_loss instead
'''


def mse_loss():
    def loss_function(est, x, y, z):
	

        print("not implemented")
		return WeightedSDR(est, x, y, z)
		
        


    print("not implemented")
        
    return loss_function