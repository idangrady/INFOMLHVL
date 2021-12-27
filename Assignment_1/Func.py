import numpy as np

# performing normalisation ==> on each loc x-m \std
class activation_func():
    
  def sigmoid_activation_func(e):
    return np.maximum([0,(1/1+np.exp(-e))])

  def soft_max_activation(e):
    return ((np.exp(e)/ np.sum(np.exp(e))))

  def relu_activation_fun(mat):
    mat[mat<0] =0
    
    return (mat)


def normilied_(mat):
    std = np.std(mat)
    mean_ = np.mean(mat)
    return ((mat-mean_)/std)


def flattern(mat,weight):
    flattern = mat.flatten()
    reshaped_ =flattern.reshape((1,mat.shape[1]* mat.shape[0]))
    return (np.dot( weight,reshaped_.T))


def convo(mat,feature_maps):
  num_dim = mat.ndim
  return (two_d(mat, feature_maps) if num_dim == 2 else three_d_normal(
      mat, feature_maps))
    
    
def two_d(mat,filters):
    current_feature = filters.pop(0)
    d_mat,n_mat= mat.shape
    d_feat_map,n_feat_map =current_feature.shape
    
    output_arr= np.zeros((d_mat-d_feat_map+1,n_mat-n_feat_map+1))
    for row in range(output_arr.shape[0]):
        for col in range(output_arr.shape[1]):
            current_block = mat[row:row+d_feat_map,col:col+n_feat_map]
            output_arr[row][col] = np.sum(current_block * current_feature)
    if len(filters)==0:
        return output_arr
    else: return two_d(output_arr,filters)



def three_d_normal(mat,feature_maps):
    current_feature = feature_maps.pop(0)
    d_mat,n_mat,z_mat= mat.shape
    d_feat_map,n_feat_map,z_map =current_feature.shape
    
    output_arr= np.zeros((d_mat-d_feat_map+1,n_mat-n_feat_map+1,z_mat-z_map+1))
    for row in range(output_arr.shape[0]):
        for col in range(output_arr.shape[1]):
            for z_dim in range(output_arr.shape[2]):
                current_block = mat[row:row+d_feat_map,col:col+n_feat_map,z_dim:z_dim+z_map]
                output_arr[row][col][z_dim] = np.sum(current_block * current_feature)
    if len(feature_maps)==0:
        return output_arr
    else: return two_d(output_arr,feature_maps)



def split_data_fetu(mat,y_val):
    check_ = (np.vsplit(mat,y_val))
    return (np.vsplit(mat,y_val))

def split_on_x(mat,nun):
  check = np.split(mat,nun,axis =1)
  return [np.max(x) for x in check]
  
def map_1(mat,x_val,y_val,output_size):
  splitted_fetures  =split_data_fetu(mat,y_val)
  pri = list(splitted_fetures)
  plitted_output = [split_on_x(mat,x_val) for mat in pri]
  return np.array(plitted_output).reshape(output_size)
  
    
  
if __name__ == "__main__": 
    
  #innitiate values
  mat = np.random.randint(0,10,[4,4])
  featu_map =np.random.randint(-1,1,[2,2])
  weight = np.random.randint(-1,1,[3,9])
  print(' Original:')
  print(f" {mat}")
  print("weight:")
  print(weight)

  print('filter:')
  print(f"{featu_map}")
  fetu = convo(mat,[featu_map])
  print('convo: ')
  print(fetu)
  print("flattern:")
  print(flattern(fetu,weight))
  print("pooling:")
  print(map_1(mat,2,2,(2,2)))
    
    
    #to perform mat_pool:
# =============================================================================
#     print(map_1(mat,2,2,(2,2)))
# 
# =============================================================================

    # Perform nomalisation
# =============================================================================
#     print(normalisation_mat(mat,featu_map))
# =============================================================================
    