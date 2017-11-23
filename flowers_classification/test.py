from skimage import io,transform
import tensorflow as tf
import numpy as np
import os
import glob

imgs=[]
for i in glob.glob("test/*.jpg"):
    imgs.append(i)

flower_dict={0:'daisy',1:'dandelion',2:'roses',3:'sunflowers',4:'tulips'}

w=100
h=100
c=3
def read_image(path):
    img=io.imread(path)
    img=transform.resize(img,(w,h))
    return np.asarray(img)

with tf.Session() as sess:
    data=[]
    for i in imgs:
        data.append(read_image(i))

    saver=tf.train.import_meta_graph('flower/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint("flower/"))


    graph=tf.get_default_graph()
    x=graph.get_tensor_by_name("x:0")
    print(x)
    feed_dict={x:data}
    logits=graph.get_tensor_by_name("logits_eval:0")

    classifiction_result=sess.run(logits,feed_dict)


    print(classifiction_result)
    print(tf.argmax(classifiction_result,1).eval())

    output=[]
    output=tf.argmax(classifiction_result,1).eval()
    for i in range(len(output)):
        print("第",i+1,"朵花预测："+flower_dict[output[i]])






