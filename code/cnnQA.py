# -*- coding: utf-8 -*-
# encoding='utf-8'

import sys
import os
import tensorflow as tf
import numpy as np
import time
import datetime
import operator
import random

max_question_length = 100
max_answer_length = 100
training_file = './data/training.data'
development_file  = './data/develop.data'
testing_file  = './data/testing.data'

working_type = 3
'''
working_type 的取值 
1  为开始新的training，根据训练集训练神经网络，神经网络模型保存在model下本次运行的目录当中
2  为继续旧的training，读取上一次的model，继续进行训练
3  为developing，读取model，根据开发集测试神经网络效果
4  为testing，读取model，利用测试集测试神经网络，输出result文件为每行中答案对于问题的得分（即认为是正确答案的确率）
'''

# 重要!!使用的model(神经网络的参数状态)，进行新的训练后需要修改使用模型才可以应用训练好的新模型
# 由于本程序机制，每当更换training或deveing集就需要进行新的训练


model = "./models/1501225714/model-9000"
modelstamp = "1501225714" #与model所在文件夹名称一致


# 一些参数变量
tf.flags.DEFINE_integer("embedding_size", 100, "word embedding时目标空间的维度")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5", "filter(卷积核)的大小")
tf.flags.DEFINE_integer("number_of_filters", 500, "单位filter大小对应的filter数目")
tf.flags.DEFINE_float("dropout", 1.0, "Dropout保持概率")
tf.flags.DEFINE_integer("batch_size", 30, "每批(Batch)样本数目")
tf.flags.DEFINE_integer("epochs", 25000, "Epochs次数")
tf.flags.DEFINE_integer("checkpoint_interval", 1000, "保存模型参数的检查点之间的迭代次数间隔")
#session相关
tf.flags.DEFINE_boolean("allow_soft_placement", True, "允许op在指定设备不可用时分配到当前可用设备")
tf.flags.DEFINE_boolean("log_device_placement", False, "记录op节点的分配")




'''def test_util():
	characterset = build_character_set(training_file,development_file)
	ans = read_answer_list(training_file)
	rans = read_right_answer(training_file)
	wans = read_wrong_answer(training_file)
	q,ra,wa = load_training_data(characterset,wans,ans,rans,50)'''


def build_character_set(training_file, development_file):
	characters = {}
	code = int(0)
	characters[u'夨'] = code
	code += 1
	for line in open(training_file,'r', encoding='UTF-8'):
		cols = line.strip().split('\t')
		for i in range(0, 2):
			for ch in cols[i]:
				if not ch in characters:
					characters[ch] = code
					code += 1
	for line in open(development_file,'r', encoding='UTF-8'):
		cols = line.strip().split('\t')
		for i in range(0, 2):
			for ch in cols[i]:
				if not ch in characters:
					characters[ch] = code
					code += 1
	for line in open(testing_file,'r', encoding='UTF-8'):
		cols = line.strip().split('\t')
		for i in range(0, 2):
			try:
				for ch in cols[i]:
					if not ch in characters:
						characters[ch] = code
						code += 1
			except Exception as e:
				print(e)
	return characters


def read_answer_list(dataset):
	ans = []
	for line in open(dataset,'r', encoding='UTF-8'):
		cols = line.strip().split('\t')
		ans.append(cols[1])
	return ans

def read_right_answer(dataset):
	rans = []
	for line in open(dataset,'r', encoding='UTF-8'):
		cols = line.strip().split('\t')
		if cols[2] == '1':
			rans.append(cols)
	return rans

def read_wrong_answer(dataset):
	wans = {}
	for line in open(dataset,'r', encoding='UTF-8'):
		cols = line.strip().split('\t')
		if cols[2] == '0':
			if not cols[0] in wans:
				wans[cols[0]] = []
			wans[cols[0]].append(cols[1])
	return wans

def random_read(foolist):
	idx = random.randint(0, len(foolist) - 1)
	return foolist[idx]

def encode(characterset, string, size):
	x = []
	words = string
	for i in range(0, size):
		if (i < len(words)):
			if words[i] in characterset:
				x.append(characterset[words[i]])
			else:
				x.append(characterset[u'夨'])
				print("该汉字未能编码：",words[i],"可能是因为没有在文件开头指定相应的数据集")
		else:
			x.append(characterset[u'夨'])
	return x

def preload_data(characterset, ans, rans, size):
	q = []
	for i in range(0,size):
		items = rans[random.randint(0, len(rans) - 1)]
		q.append(encode(characterset,items[0],max_question_length))
	return np.array(q)


def load_training_data(characterset, wans, ans, rans, size):
	feed_q = []
	feed_ra = []
	feed_wa = []
	for i in range(0, size):
		items = rans[random.randint(0, len(rans) - 1)]
		nega = random_read(ans)
		if items[0] in wans:
			nega = random_read(wans[items[0]])
		feed_q.append(encode(characterset, items[0], max_question_length))
		feed_ra.append(encode(characterset, items[1], max_answer_length))
		feed_wa.append(encode(characterset, nega, max_answer_length))
	return np.array(feed_q), np.array(feed_ra), np.array(feed_wa)

def load_testing_data(characterset, question, answer, batch_size):
	test_q = []
	test_ra = [] #实际上是某一个候选答案
	test_wa = [] #实际上与上一项相同
	for i in range(0, batch_size):
		test_q.append(encode(characterset, question, max_question_length))
		test_ra.append(encode(characterset, answer, max_question_length))
		test_wa.append(encode(characterset, answer, max_question_length))
	return np.array(test_q), np.array(test_ra), np.array(test_wa)

'''
CNN的Graph定义
'''

class QACNN(object):

	def __init__(
			self, sequence_length, batch_size,
			vocab_size, embedding_size,
			filter_sizes, num_filters, l2_reg_lambda=0.0):
		'''
		输入占位符，分别为问题，正向答案，负向答案，shape：[30,100] type:int32
		'''
		self.input_question = tf.placeholder(
			tf.int32, [batch_size, sequence_length], name="input_question")
		self.input_r_answer = tf.placeholder(
			tf.int32, [batch_size, sequence_length], name="input_r_answer")
		self.input_w_answer = tf.placeholder(
			tf.int32, [batch_size, sequence_length], name="input_w_answer")
		'''
		不使用dropout，运行时为1
		'''
		self.dropout_keep_prob = tf.placeholder(
			tf.float32, name="dropout_keep_prob")

		l2_loss = tf.constant(0.0)  # 不使用L2正则

		'''
		word embedding层，w为映射空间，作为参数可训练
		'''
		with tf.device('/gpu:0'), tf.name_scope("embedding"):
			W = tf.Variable(
				tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0),
				name="W")  # 生成W参数矩阵，范围-1.0到1.0 shape：【8754(对于目前数据集)，100】
			self.embedded_chars_1 = tf.nn.embedding_lookup(
				W, self.input_question)
			self.embedded_chars_2 = tf.nn.embedding_lookup(
				W, self.input_r_answer)
			self.embedded_chars_3 = tf.nn.embedding_lookup(
				W, self.input_w_answer)
			# 30*100*100 -》 30*100*100*1 拓展维度以进行卷积运算
		self.embedded_chars_expanded_1 = tf.expand_dims(
			self.embedded_chars_1, -1)
		self.embedded_chars_expanded_2 = tf.expand_dims(
			self.embedded_chars_2, -1)
		self.embedded_chars_expanded_3 = tf.expand_dims(
			self.embedded_chars_3, -1)

		pooled_outputs_1 = []
		pooled_outputs_2 = []
		pooled_outputs_3 = []

		'''
		卷积-maxpooling层，进行卷积运算滤波，共4个filter
		'''

		for i, filter_size in enumerate(filter_sizes):
			with tf.name_scope("conv-maxpool-%s" % filter_size):
				# filter的shape，每个filter不同，此次为[1/2/3/5,100,1,500]
				filter_shape = [filter_size, embedding_size, 1, num_filters]
				W = tf.Variable(tf.truncated_normal(
					filter_shape, stddev=0.1), name="W")  # 可训练filter参数矩阵w
				b = tf.Variable(tf.constant(
					0.1, shape=[num_filters]), name="b")  # bias，取0.1
				conv = tf.nn.conv2d(
					self.embedded_chars_expanded_1,
					W,
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="conv-1"
				)
				h = tf.nn.relu(tf.nn.bias_add(conv, b),
							   name="relu-1")  # 用ReLU作为激活函数
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, sequence_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="poll-1"
				)
				# shape[30,1,1,500],4个filter[30,1,1,2000]
				pooled_outputs_1.append(pooled)

				conv = tf.nn.conv2d(
					self.embedded_chars_expanded_2,
					W,
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="conv-2"
				)
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-2")
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, sequence_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="poll-2"
				)
				pooled_outputs_2.append(pooled)

				conv = tf.nn.conv2d(
					self.embedded_chars_expanded_3,
					W,
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="conv-3"
				)
				h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu-3")
				pooled = tf.nn.max_pool(
					h,
					ksize=[1, sequence_length - filter_size + 1, 1, 1],
					strides=[1, 1, 1, 1],
					padding='VALID',
					name="poll-3"
				)
				pooled_outputs_3.append(pooled)
		num_filters_total = num_filters * len(filter_sizes)
		#[30,1,1,2000] -> [30,2000]
		pooled_reshape_1 = tf.reshape(
			tf.concat(axis=3, values=pooled_outputs_1), [-1, num_filters_total])
		pooled_reshape_2 = tf.reshape(
			tf.concat(axis=3, values=pooled_outputs_2), [-1, num_filters_total])
		pooled_reshape_3 = tf.reshape(
			tf.concat(axis=3, values=pooled_outputs_3), [-1, num_filters_total])

		pooled_flat_1 = tf.nn.dropout(pooled_reshape_1, self.dropout_keep_prob)
		pooled_flat_2 = tf.nn.dropout(pooled_reshape_2, self.dropout_keep_prob)
		pooled_flat_3 = tf.nn.dropout(pooled_reshape_3, self.dropout_keep_prob)

		# 计算向量长度
		pooled_len_1 = tf.sqrt(tf.reduce_sum(
			tf.multiply(pooled_flat_1, pooled_flat_1), 1))
		pooled_len_2 = tf.sqrt(tf.reduce_sum(
			tf.multiply(pooled_flat_2, pooled_flat_2), 1))
		pooled_len_3 = tf.sqrt(tf.reduce_sum(
			tf.multiply(pooled_flat_3, pooled_flat_3), 1))

		# 计算向量的点乘
		pooled_mul_12 = tf.reduce_sum(
			tf.multiply(pooled_flat_1, pooled_flat_2), 1)
		pooled_mul_13 = tf.reduce_sum(
			tf.multiply(pooled_flat_1, pooled_flat_3), 1)

		'''
		输出层:
		计算余弦相似度
		计算损失函数
		'''

		with tf.name_scope("output"):
			# 得到向量夹角cos值
			self.cos_12 = tf.div(pooled_mul_12, tf.multiply(
				pooled_len_1, pooled_len_2), name="scores")
			self.cos_13 = tf.div(pooled_mul_13, tf.multiply(
				pooled_len_1, pooled_len_3))

		zero = tf.constant(0, shape=[batch_size], dtype=tf.float32)
		margin = tf.constant(0.05, shape=[batch_size], dtype=tf.float32)
		# m值，设为0.05

		# 计算损失函数
		with tf.name_scope("loss"):
			self.losses = tf.maximum(zero, tf.subtract(
				margin, tf.subtract(self.cos_12, self.cos_13)))
			self.loss = tf.reduce_sum(self.losses) + l2_reg_lambda * l2_loss
			print('loss ', self.loss)

		
		with tf.name_scope("accuracy"):
			self.correct = tf.equal(zero, self.losses)
			self.accuracy = tf.reduce_mean(
				tf.cast(self.correct, "float"), name="accuracy")






'''
主程序部分
'''



FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()

print("\n当前参数列表:")
for attr, value in sorted(FLAGS.__flags.items()):
	print("{}={}".format(attr.upper(), value))

characterset = build_character_set(training_file,development_file)
answerlist = read_answer_list(training_file)
rightanswer = read_right_answer(training_file)
wronganswer = read_wrong_answer(training_file)

q = preload_data(characterset,answerlist,rightanswer,FLAGS.batch_size)

with tf.Graph().as_default():
	with tf.device("/gpu:0"):
		session_config = tf.ConfigProto(
			allow_soft_placement=FLAGS.allow_soft_placement,
			log_device_placement=FLAGS.log_device_placement)
		sess = tf.Session(config=session_config)
		with sess.as_default():
			qanet = QACNN(
				sequence_length = q.shape[1],
				batch_size = FLAGS.batch_size,
				vocab_size = len(characterset),
				embedding_size = FLAGS.embedding_size,
				filter_sizes = list(map(int, FLAGS.filter_sizes.split(","))),
				num_filters=FLAGS.number_of_filters,
				l2_reg_lambda=0.0)

			# 设置优化器使用AdamOptimizer，global_step用于记录总步数，保存为Variable可以随模型保存，因为是无关变量所以不能训练
			global_step = tf.Variable(0, name="global_step", trainable=False)
			optimizer = tf.train.AdamOptimizer(1e-1)#学习速率0.1，其余参照标准参数
			grad = optimizer.compute_gradients(qanet.loss)#利用loss函数计算梯度
			train_op = optimizer.apply_gradients(grad, global_step=global_step)#train_op表示进行梯度下降训练的op结点

			sess.run(tf.global_variables_initializer())
			if working_type == 1:
				timestamp = str(int(time.time()))
				out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", timestamp))
				print("此次训练模型保存至：{}\n".format(out_dir))
				checkpoint_dir = os.path.abspath(os.path.join(out_dir))
				checkpoint_prefix = os.path.join(checkpoint_dir, "model")
				if not os.path.exists(checkpoint_dir):
					os.makedirs(checkpoint_dir)
			else:
				out_dir = os.path.abspath(os.path.join(os.path.curdir, "models", modelstamp))
				print("读取{}\n".format(out_dir),"中的模型")
				checkpoint_dir = os.path.abspath(os.path.join(out_dir))
				checkpoint_prefix = os.path.join(checkpoint_dir, "model")
				if not os.path.exists(checkpoint_dir):
					os.makedirs(checkpoint_dir)

			saver = tf.train.Saver(tf.global_variables())

			if working_type != 1:
				saver.restore(sess, model)

			def training_step(input_question, input_r_answer, input_w_answer):
				feed = {
					qanet.input_question: input_question,
					qanet.input_r_answer: input_r_answer,
					qanet.input_w_answer: input_w_answer,
					qanet.dropout_keep_prob: FLAGS.dropout
				}

				_, step, loss, accuracy = sess.run(
					[train_op, global_step, qanet.loss, qanet.accuracy],
					feed)
				time_str = datetime.datetime.now().isoformat()
				print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))

			def test(testfile):#运行测试集
				def get_score_by_qa(question, answer):
					input_question, input_r_answer, input_w_answer = load_testing_data(characterset,question,answer,FLAGS.batch_size)
					feed = {
						qanet.input_question: input_question,
						qanet.input_r_answer: input_r_answer,
						qanet.input_w_answer: input_w_answer,
						qanet.dropout_keep_prob: 1.0
					}
					batch_score = sess.run([qanet.cos_12], feed)
					return batch_score[0][0]

				file1 = './data/result.txt'
				i = 1
				f = open(file1,"w+")
				for line in open(testfile,'r',encoding='UTF-8'):
					print (str(i)+"/"+str(122531))
					i = i + 1
					items = line.strip().split('\t')
					try:
						f.write(str(get_score_by_qa(items[0],items[1]))+"\n")
					except:
						f.write(str(0)+"\n")
					if (i % 1000) == 0 :
						try:
							print("每隔1000定期保存")
							f.flush()
						except:
							print("写缓冲出错")

			def develop(file_to_test): #评估开发集

				def get_score_by_qa(question, answer):
					input_question, input_r_answer, input_w_answer = load_testing_data(characterset,question,answer,FLAGS.batch_size)
					feed = {
						qanet.input_question: input_question,
						qanet.input_r_answer: input_r_answer,
						qanet.input_w_answer: input_w_answer,
						qanet.dropout_keep_prob: 1.0
					}
					batch_score = sess.run([qanet.cos_12], feed)
					return batch_score[0][0]
				def dict2list(dic:dict):
					keys = dic.keys()
					vals = dic.values()
					lst = [(key, val) for key, val in zip(keys, vals)]
					return lst
				def sortedDictValues(adict): 
					return sorted(dict2list(adict), key=lambda x:x[0], reverse=True)

				def mmr_calc(final_score, final_score_count, answer_score):
					answer_score = sortedDictValues(answer_score)
					cnt = int(1)
					rank = int(0)
					for item in answer_score:

						if (right_answer == item[1]):
							rank = cnt
						cnt = cnt + 1
					if rank >= 1:
						final_score = final_score + 1.0/float(rank)
						final_score_count = final_score_count + 1
						MRR = final_score/float(final_score_count)
						if cnt != 1:
							print(u"这是第{}个问题".format(final_score_count))
							print(u"正确回答顺位/答案池总数：{}/{}".format(rank,cnt-1))
							print(u"目前MRR ： {}\n".format(MRR))
					return final_score, final_score_count

				temp_question_vector = u"null"
				right_answer = u"null"
				answer_score = {}
				final_score = 0.0
				final_score_count = 0

				for line in open(file_to_test,'r', encoding='UTF-8'):
					items = line.strip().split('\t')
					question = items[0]
					answer   = items[1]
					score    = get_score_by_qa(question, answer)
					
					if question != temp_question_vector:
						print("问题:{}".format(temp_question_vector))
						final_score, final_score_count = mmr_calc(final_score, final_score_count, answer_score)

						answer_score.clear()
						answer_score = {}
						temp_question_vector = question
						answer_score[score] = answer
					else:
						answer_score[score] = answer

					if items[2] == '1':
						right_answer = answer
					else:
						pass

				print("问题:{}".format(temp_question_vector))
				final_score, final_score_count = mmr_calc(final_score, final_score_count, answer_score)

				MRR = final_score/float(final_score_count)
				print("已结束，问题总数:{}".format(final_score_count))
				print("最终MRR:{}".format(MRR))
				

			def train():
				for i in range(FLAGS.epochs):
					try:
						#在这里才是真格地开始加载训练数据，一次30个，
						q, ra, wa = load_training_data(characterset,wronganswer,answerlist,rightanswer,FLAGS.batch_size)
						training_step(q, ra, wa)
						current_step = tf.train.global_step(sess, global_step)
						if current_step % FLAGS.checkpoint_interval == 0:
							path = saver.save(sess, checkpoint_prefix, global_step=current_step)
							print("模型检查点已保存到{}\n".format(path))
					except Exception as e:
						print("异常:",e)

			if working_type == 1:
				train()
			if working_type == 2:
				train()
			if working_type == 3:
				develop(development_file)
			if working_type == 4:
				test(testing_file)
