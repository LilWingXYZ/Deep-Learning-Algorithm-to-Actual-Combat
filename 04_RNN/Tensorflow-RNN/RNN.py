# coding=utf-8
import tensorflow as tf
from tensorflow.contrib import learn
import numpy as np
from tensorflow.python.ops.rnn import static_rnn
from tensorflow.python.ops.rnn_cell_impl import BasicLSTMCell

# 数据
positive_texts = [
    "我 今天 很 高兴",
    "我 很 开心",
    "他 很 高兴",
    "他 很 开心"
]
negative_texts = [
    "我 不 高兴",
    "我 不 开心",
    "他 今天 不 高兴",
    "他 不 开心"
]

label_name_dict = {
    0: "正面情感",
    1: "负面情感"
}

# 配置信息
embedding_size = 50
num_classes = 2

# 将文本和label数值化
all_texts = positive_texts + negative_texts
labels = [0] * len(positive_texts) + [1] * len(negative_texts)

max_document_length = 4
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)

datas = np.array(list(vocab_processor.fit_transform(all_texts)))
vocab_size = len(vocab_processor.vocabulary_)

# 容器，存放输入输出
datas_placeholder = tf.placeholder(tf.int32, [None, max_document_length])
labels_placeholder = tf.placeholder(tf.int32, [None])

# 词向量表
embeddings = tf.get_variable("embeddings", [vocab_size, embedding_size], initializer=tf.truncated_normal_initializer)

# 将词索引号转换为词向量[None, max_document_length] => [None, max_document_length, embedding_size]
embedded = tf.nn.embedding_lookup(embeddings, datas_placeholder)

# 转换为LSTM的输入格式，要求是数组，数组的每个元素代表某个时间戳一个Batch的数据
rnn_input = tf.unstack(embedded, max_document_length, axis=1)

# 定义LSTM，20是状态的维度
lstm_cell = BasicLSTMCell(20, forget_bias=1.0)
rnn_outputs, rnn_states = static_rnn(lstm_cell, rnn_input, dtype=tf.float32)

# 利用LSTM最后的输出进行预测
logits = tf.layers.dense(rnn_outputs[-1], num_classes)

predicted_labels = tf.argmax(logits, axis=1)

# 定义损失和优化器
losses = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(labels_placeholder, num_classes),
    logits=logits
)

mean_loss = tf.reduce_mean(losses)
optimizer = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(mean_loss)

with tf.Session() as sess:
    # 初始化变量
    sess.run(tf.global_variables_initializer())

    print("开始训练")
    # 定义训练要填充的数据
    train_feed_dict = {
        datas_placeholder: datas,
        labels_placeholder: labels
    }
    for step in range(100):
        _, mean_loss_val = sess.run([optimizer, mean_loss], feed_dict=train_feed_dict)
        if step % 10 == 0:
            print("step = {}\tmean loss = {}".format(step, mean_loss_val))

    print("训练结束，进行预测")
    # 定义测试要填充的数据
    test_feed_dict = {
        datas_placeholder: datas
    }
    predicted_labels_val = sess.run(predicted_labels, feed_dict=test_feed_dict)
    for i, text in enumerate(all_texts):
        label = predicted_labels_val[i]
        label_name = label_name_dict[label]
        print("{} => {}".format(text, label_name))
