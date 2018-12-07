import tensorflow as tf 
import numpy as np
import pickle
import random
import matplotlib.pyplot as plt
from PIL import Image


with open("dataset_grayscale32.pickle", 'rb') as f:
    train_data = [ pickle.load(f) ]    
    
with open("digit_labels.pickle", 'rb') as f:
    train_labels = pickle.load(f)
    
train_labels = train_labels[:,0:5]
train_d = train_data
    
#==================NORMALISATION AND PREPROCESSING=============================================

train_data = np.asarray(train_data).astype('float32') / 128.0 - 1

train_data = np.transpose(train_data, (1, 2, 3, 0))
print(train_data.shape)
train_tuple = list(zip(train_data, train_labels))

"""
Various Hyperparameters required for training the CNN.
"""
image_size = 32
width = 32
height = 32
channels = 1

n_labels = 11
patch = 5
depth = 16
hidden = 128
dropout = 0.9375

batch = 50
learning_rate = 0.001

"""
Constructing the placeholders and variables in the TensorFlow Graph
"""
#Training Dataset
tf_train_dataset = tf.placeholder(tf.float32, shape=(None, width, height, channels))
#Training Labels
tf_train_labels = tf.placeholder(tf.float32, shape=(None, 5, n_labels))

#   Layer 1: (5, 5, 3, 16)
layer1_weights = tf.Variable(tf.truncated_normal([patch, patch, channels, depth], stddev=0.1))
layer1_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

#   Layer 2: (5, 5, 16, 16)
layer2_weights = tf.Variable(tf.truncated_normal([patch, patch, depth, depth], stddev=0.1))
layer2_biases = tf.Variable(tf.constant(1.0, shape=[depth]))

#   Layer 3: (1024, 128)
layer3_weights = tf.Variable(tf.truncated_normal([image_size // 4 * image_size // 4 * depth, hidden], stddev=0.1))
layer3_biases = tf.Variable(tf.constant(1.0, shape=[hidden]))

#   Layer 4: (128, 10)
layer4_weights = tf.Variable(tf.truncated_normal([hidden, n_labels], stddev=0.1))
layer4_biases = tf.Variable(tf.constant(1.0, shape=[n_labels]))

#   Layer 4: (128, 10)
layer6_weights = tf.Variable(tf.truncated_normal([hidden, n_labels], stddev=0.1))
layer6_biases = tf.Variable(tf.constant(1.0, shape=[n_labels]))

#   Layer 4: (128, 10)
layer7_weights = tf.Variable(tf.truncated_normal([hidden, n_labels], stddev=0.1))
layer7_biases = tf.Variable(tf.constant(1.0, shape=[n_labels]))

#   Layer 4: (128, 10)
layer8_weights = tf.Variable(tf.truncated_normal([hidden, n_labels], stddev=0.1))
layer8_biases = tf.Variable(tf.constant(1.0, shape=[n_labels]))

#   Layer 4: (128, 10)
layer9_weights = tf.Variable(tf.truncated_normal([hidden, n_labels], stddev=0.1))
layer9_biases = tf.Variable(tf.constant(1.0, shape=[n_labels]))


dropout = tf.placeholder(tf.float32)

def model(data):
    #   Convolution 1 and RELU
    conv1 = tf.nn.conv2d(data, layer1_weights, [1, 1, 1, 1], padding='SAME')
    hidden1 = tf.nn.relu(conv1 + layer1_biases)
    #   Max Pool
    hidden2 = tf.nn.max_pool(hidden1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    #   Convolution 2 and RELU
    conv2 = tf.nn.conv2d(hidden2, layer2_weights, [1, 1, 1, 1], padding='SAME')
    hidden3 = tf.nn.relu(conv2 + layer2_biases)
    #   Max Pool
    hidden4 = tf.nn.max_pool(hidden3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    shape = hidden4.get_shape().as_list()

    reshape = tf.reshape(hidden4, [-1, shape[1] * shape[2] * shape[3]])
    hidden5 = tf.nn.relu(tf.matmul(reshape, layer3_weights) + layer3_biases)
    #   Dropout
    dropout_layer1 = tf.nn.dropout(hidden5, 0.93)
    
    logit1 = tf.matmul(dropout_layer1, layer4_weights) + layer4_biases
    logit2 = tf.matmul(dropout_layer1, layer6_weights) + layer6_biases
    logit3 = tf.matmul(dropout_layer1, layer7_weights) + layer7_biases
    logit4 = tf.matmul(dropout_layer1, layer8_weights) + layer8_biases
    logit5 = tf.matmul(dropout_layer1, layer9_weights) + layer9_biases
        
    return [logit1, logit2, logit3, logit4, logit5]

logits = model(tf_train_dataset)
predict = tf.stack([tf.nn.softmax(logits[0]), tf.nn.softmax(logits[1]),tf.nn.softmax(logits[2]), tf.nn.softmax(logits[3]), tf.nn.softmax(logits[4])])

best_prediction = tf.transpose(tf.argmax(predict, 2))

loss1 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits[0], labels=tf_train_labels[:, 0]))
loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits[1], labels=tf_train_labels[:, 1]))
loss3 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits[2], labels=tf_train_labels[:, 2]))
loss4 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits[3], labels=tf_train_labels[:, 3]))
loss5 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits[4], labels=tf_train_labels[:, 4]))
loss = loss1 + loss2 + loss3 + loss4 + loss5

optimizer = tf.train.AdamOptimizer(0.001).minimize(loss)


def accuracy(predictions, labels):
    #print('Inside Predictions')
    one = (100.0 * (np.sum(predictions[:, 0] == np.argmax(labels, 2)[:, 0])) / predictions.shape[0])
    two = (100.0 * (np.sum(predictions[:, 1] == np.argmax(labels, 2)[:, 1])) / predictions.shape[0])
    three = (100.0 * (np.sum(predictions[:, 2] == np.argmax(labels, 2)[:, 2])) / predictions.shape[0])
    four = (100.0 * (np.sum(predictions[:, 3] == np.argmax(labels, 2)[:, 3])) / predictions.shape[0])
    five = (100.0 * (np.sum(predictions[:, 4] == np.argmax(labels, 2)[:, 4])) / predictions.shape[0])
    return one, two, three, four, five
    
def accuracy_sequence(predictions, labels):
    count = 0
    for i in range(predictions.shape[0]):
        pred_ = []
        labels_ = []
        for j in range(5):
            if predictions[i][j] != 10:
                pred_.append(predictions[i][j])
            else:
                break
        for j in range(5):
            if labels[i][j] != 10:
                labels_.append(labels[i][j])
            else:
                break
        a = pred_ == labels_
        if a == True:
            count += 1
    return (100 * count/predictions.shape[0])

#   Number of iterations
num_steps = 5

init = tf.global_variables_initializer()

session = tf.Session()

saver = tf.train.Saver()

class SVHN(object):

    path = ""

    def __init__(self):
        """
            data_directory : path like /home/rajat/mlproj/dataset/
                            includes the dataset folder with '/'
            Initialize all your variables here
        """

    def train(self):
        """
            Trains the model on data given in path/train.csv

            No return expected
        """
        
        session.run(init)
        print('Initialized')
        seq_acc_list = []
        losses = []
        merged = tf.summary.merge_all()        
        model = tf.summary.FileWriter('model/', session.graph)        
        
        #average = 0
        for step in range(num_steps):
            batch = random.sample(train_tuple, 50)
            batch_data = [zz[0] for zz in batch]
            batch_labels = np.asarray( [zz[1] for zz in batch] )

            feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels, dropout: 0.93}
            summary, _, l, best_pred = session.run([merged, optimizer, loss, best_prediction], feed_dict=feed_dict)
            #   Calculating the Accuracy of the predictions
            acc1, acc2, acc3, acc4, acc5 = accuracy(best_pred, batch_labels)
            acc_seq = accuracy_sequence(best_pred, np.argmax(batch_labels, 2))
            seq_acc_list.append(acc_seq)
            losses.append(l)

            if (step % 50 == 0):

                print(np.argmax(batch_labels, 2)[:, 0])
                print(best_pred[:,0])
                print('and')
                print(np.argmax(batch_labels, 2)[:, 1])
                print(best_pred[:,1])

                print('Loss at step %d: %f' % (step, l))
                print('Accuracy for digit 1: %.1f%%' % acc1)
                print('Accuracy for digit 2: %.1f%%' % acc2)
                print('Accuracy for digit 3: %.1f%%' % acc3)
                print('Accuracy for digit 4: %.1f%%' % acc4)
                print('Accuracy for digit 5: %.1f%%' % acc5)
                print('Accuracy: %.1f%%' % acc_seq)
                
        #save_path = saver.save(session, "savedmodel/saved.ckpt")
        #print("Model saved in file: %s" % save_path)
                
#        plt.plot(range(20000), losses, color='red', label='Training')
#        plt.title('Training Losses')
#        plt.ylabel('Overall Cross Entropy Losses')
#        plt.xlabel('Number of iterations')
#        plt.legend()
#        plt.savefig('Losses.png', pad_inches=0.1, bbox_inches='tight')
#                
#        plt.plot(range(20000), seq_acc_list, color='blue', label='Training')
#        plt.title('Training Accuracies')
#        plt.ylabel('Percentage Accuracy')
#        plt.xlabel('Number of iterations')
#        plt.legend()
#        plt.savefig('Accuracies.png', pad_inches=0.1, bbox_inches='tight')
#
#        print("END OF TRAINING")
        
        
        model.add_summary(summary)

    def get_sequence(self, image):
        """
            image : a variable resolution RGB image in the form of a numpy array

            return: list of integers with the sequence of digits. Example: [5,0,3] for an image having 503 as the sequence.

        """
        image = Image.fromarray(image)
        image = image.convert('L')  
        img_resized=image.resize((32,32), Image.ANTIALIAS)
        img_resized=np.asarray(img_resized)
        img_re = [[img_resized]]
        img_re = np.transpose(img_re, (1, 2, 3, 0))
        
        fd = {tf_train_dataset:img_re}
        preds = session.run(best_prediction, feed_dict=fd)
        ans = []
        for j in range(5):
            if preds[0][j] != 10:
                ans.append(preds[0][j])
            else:
                break
        print(ans)
        return ans

    def save_model(self, **params):

        # file_name = params['name']
        # pickle.dump(self, gzip.open(file_name, 'wb'))

        """
            saves model on the disk

            no return expected
        """
        

    @staticmethod
    def load_model(**params):

        # file_name = params['name']
        # return pickle.load(gzip.open(file_name, 'rb'))

        """
            returns a pre-trained instance of SVHN class
        """
        obj1 = SVHN()
        
        saver.restore(session, "savedmodel/saved.ckpt")
        return obj1

if __name__ == "__main__":
        #obj = SVHN()
        #obj.train()
        #obj.save_model(name="svhn.gz")
        
        obj = SVHN.load_model()
        
        
        #with open("dataset.pickle", 'rb') as f:
            #batch = pickle.load(f)
        #train_d = np.transpose(train_data, (1, 2, 3, 0)  
        #print(train_data.shape)
        #print(type(train_data[0])

            
        
        
        
        
        
        
        
        
        
        