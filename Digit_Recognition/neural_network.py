from data import get_training_data
import numpy as np
import matplotlib.pyplot as plt

'''
--- understanding weight dimension :
    supose we have 2 input x1,x2 and we then need 2 weight for tweaking. w1,w2 and 1 bias,making a single output:
                                                    y = w1x1 + w2x2 + b
    now as hidden layer have multiple neurons so instead of getting single output suppose we need 3 outputs for the hidden layers 
    x_input=[x1,x2] but for 3 outputs we need total 6 weights as each neurons will have its own pair of (w1,w2) for 2 input :
                    y1 = w11x1 + w12x2 + b1
                    y2 = w12x1 + w22x2 + b2
                    y3 = w31x1 + w32x2 + b3
            these 3 outputs are 3 values for 3 hidden layer neurons now with matrix we can achieve this :
                x_input=[x1,x2]
                w_input_to_hidden=
                                [
                                [w11,w12],
                                [w21,w22],
                                [w31,w32]
                                ]
                bias_input_to_hidden=[b1,b2,b3]

                w matrix multiplication with transpose of x gives us a 3*1 mattrix , adding transpose of b with it gives us 3 data for our 3 neurons 

                so weight dimensions are defined like :
                 weight_input_to_hidden= (quantity of hidden layer neurons , quantity of inputs )

--- learning rate should be as less possible for better accuracy

--- epochs defines how many times we will feed the same data for refining our model example: we give 5 full cycle of 60k images instead of 1 full cycle 
    with this repeatation the model will have more time to adjust its weights also greater chance to find the local minima or absolute minima  

--- zip() ties image data with coresponding label

--- our images should be column vector so changing shape from (784,) to (784,1) same for labels This turns both arrays from 1D vectors to column vectors.
    its like adding another dimension thats it.

--- @ for mattrix multiplication and all the dimensions are in place 

--- here we will use 'sigmoid function' works as an activation function which will results us in a range of [0,1]

--- nr_correct : we compare how many times we are getting correct ans each epochs so tracking correct guesses with 
    comparing max number(probablity) of o with l

                        BACKPROPAGATION

--- the tweaking/refining the weight thing came straight from the differentiation of the cost function equation is :
        new_weight = previous_weight - learning_rate * dif(cost function) w.r.t w
        new_bias   = previous_bias   - learning_rate * dif(cost function) w.r.t b

    This method is known as "Gradient Descent" 

--- Skipping the part of hidden layer to input propagation. 
    I wasn't able to fully understand the equation (line:94), so leaving it for fellow contributors to review and update.
    derivative of sigmoid : https://global.discourse-cdn.com/dlai/original/2X/a/a474004d69be98ad28fa528a185dcf72b5d1c840.jpeg


'''

images,labels=get_training_data()
w_i_h=np.random.uniform(-0.5,0.5,(20,784))
w_h_o=np.random.uniform(-0.5,0.5,(10,20))
b_i_h=np.zeros((20,1))
b_h_o=np.zeros((10,1))

learning_rate=0.01
nr_correct=0
epochs=3

for epoch in range(epochs):
    for img,l in zip(images,labels):
        img.shape+=(1,)
        l.shape+=(1,)

        # assigning value from input layers to hidden layers
        h_pre=w_i_h @ img + b_i_h
        h=1/(1+np.exp(-h_pre))

        o_pre= w_h_o @ h + b_h_o
        o=1/(1+np.exp(-o_pre))

        # doing "cost error function" calculation but not used as we will use tweaks of math for derivatives
        e=1/len(o)*np.sum((o-l)**2,axis=0)
        nr_correct+=int(np.argmax(o)==np.argmax(l))

        #refining our weights (backpropagation)
        #output -> hidden
        delta_o=o-l
        w_h_o+=-learning_rate*delta_o * np.transpose(h) 
        b_h_o+= -learning_rate*delta_o

        #hidden -> input (activation function derivative)

        delta_h=np.transpose(w_h_o) @ delta_o*(h*(1-h))
        w_i_h+= -learning_rate*delta_h*np.transpose(img)
        b_i_h+= -learning_rate*delta_h

    #accuracy for epoch
    print(f"Accuracy for epoch {epoch}:{round((nr_correct/images.shape[0])*100,2)}%")
    nr_correct=0



#showing result:
while True:
    index=int(input("choose a picture index from 0 to 59999(total 60k):"))
    img=images[index]
    plt.imshow(img.reshape(28,28),cmap="Greys")

    img.shape+=(1,)

    h_pre= w_i_h @img.reshape(784,1)
    h=1/(1+np.exp(-h_pre))

    o_pre=w_h_o @ h + b_h_o
    o=1/(1+np.exp(-o_pre))

    plt.title(f"Our first ocr: Model Says its: {o.argmax()} ")
    plt.show()













