import cPickle as pickle
import numpy as N
import numpy as np
import random 

def to_categorical(y, num_classes=None):
    y = np.array(y, dtype='int')
    #print(y)
    input_shape = y.shape
    #print(input_shape)
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    #print(y)
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes))
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def subtract (a_list,b_list):
    ret_list = []
    for item in a_list:
        if item not in b_list:
            ret_list.append(item)
    for item in b_list:
        if item not in a_list:
            ret_list.append(item)
    return ret_list

def load_data(data):

    pdb_ids =list( data.keys())

    m = {p:i for i,p in enumerate(pdb_ids)}

    print m

    all_data=np.asarray(data[str(pdb_ids[0])])
    num = len(data[str(pdb_ids[0])])
    print num
    all_label=[]

    all_num = 0
#print(m)
    for i in pdb_ids:
        if i != pdb_ids[0]:

            all_data = np.concatenate( (all_data, np.asarray(data[i])), axis = 0)
            num = len(data[i])
            all_num += num
            print num

        all_label.extend( [m[i]] *num)


    all_num = all_num+ len(data[str(pdb_ids[0])])
    all_label = np.asarray(all_label)
    all_label = to_categorical(all_label)
    print all_label
    

    all_data = np.asarray(all_data).reshape(all_num,28,28,28)

    all_data = all_data - np.mean(all_data, axis = 0)
    all_data = all_data / np.std(all_data, axis = 0) 

    all_label = np.asarray(all_label).reshape(all_num,13)
    print all_label[14692]
    print all_label.shape
    '''
    {'none': 0, b'1s3x': 1, b'1qvr': 2, b'3qm1': 3, b'1bxn': 4, b'3d2f': 5, b'1u6g': 
    6, b'3h84': 7, b'2cg9': 8, b'3gl1': 9, b'4b4t': 10, b'3cf3': 11, b'4d8q': 12}
    '''
    numbers =[14692,428,1699,809,1706,1440,1630,1125,1625,1345,366,1747,1617]
    train_data=[]
    train_label=[]
    test_data = []
    test_label=[]
    mark = 0
    train_shape = 0
    #perm = np.random.permutation(all_data.shape[0])
    #all_data = X_train[perm]
    #Y_train = Y_train[perm]

    temp = list( range(0,numbers[0]))
    #perm = np.random.permutation(temp)
    train_index = random.sample(temp, int(0.98*len(temp))) 
    train_data = all_data[train_index]
    train_label = all_label[train_index]
    test_index = subtract(temp,train_index)
    test_data = all_data[test_index]
    test_label = all_label[test_index]

    train_num =  int (numbers[0]*0.98)
    mark += numbers[0]
    train_shape += train_num


    for i in range(13):
        
        if i != 0:
            train_num = int (numbers[i]*0.98)

            temp = list( range(mark,mark+numbers[i]))
            #perm = np.random.permutation(temp)
            train_index = random.sample(temp, int(0.98*len(temp))) 
            train_data_temp = all_data[train_index]
            train_label_temp = all_label[train_index]
            test_index = subtract(temp,train_index)
            test_data_temp = all_data[test_index]
            test_label_temp = all_label[test_index]


            train_data = np.concatenate((train_data, train_data_temp), axis = 0)
        #print(len(all_data[mark+train_num: mark+number[i],:,:,:]))
            test_data = np.concatenate((test_data, test_data_temp), axis =0)
            train_label = np.concatenate(( train_label, train_label_temp),axis=0)

            test_label = np.concatenate((test_label,test_label_temp),axis=0)
            mark += numbers[i]
            train_shape += train_num

    train_data = np.asarray(train_data).reshape(train_shape,28,28,28)
    train_label = np.asarray(train_label).reshape(-1,13)
    test_data = np.asarray(test_data).reshape(all_num-train_shape,28,28,28)
    test_label = np.asarray(test_label).reshape(-1, 13)
    #print train_label

    return train_data, train_label, test_data, test_label

    #N.savez('data_denoised',x_train = train_data, y_train = train_label, x_test = test_data, y_test = test_label) 

if __name__ == '__main__':

    #a=open('/shared/xiangruz/classification/SHREC/train_data/train_data.pickle','rb')
    #data=pickle.load(a,encoding ='latin1')

    b=open('/shared/xiangruz/classification/SHREC/train_data/train_data.pickle','rb')
    data_b=pickle.load(b)
    
    #xtrain_a, ytrain_a, xtest_a, ytest_a = load_data(data)
    xtrain_b, ytrain_b, xtest_b, ytest_b = load_data(data_b)
    #train_data = np.concatenate( (xtrain_a, xtrain_b), axis = 0)
    #test_data = np.concatenate( (xtest_a,xtest_b), axis = 0)
    #train_label = np.concatenate( (ytrain_a, ytrain_b), axis = 0)
    #test_label = np.concatenate( (ytest_a, ytest_b), axis = 0)

    train_data = xtrain_b
    train_label = ytrain_b
    test_data = xtest_b
    test_label = ytest_b
    
    print train_data[0]
    
    N.savez('data',x_train = train_data, y_train = train_label, x_test = test_data, y_test = test_label) 