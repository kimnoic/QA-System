代码复现需要
python 3.5+
tensorflow 1.0+
numpy 1.2+

供参考的windows系统下的具体做法：
1、安装python 3.6 64bit版本（必须64位），保证.../python36/scripts文件夹在系统路径中
2、通过pip安装tensorflow（打开命令行->输入pip install tensorflow）
3、此时应该可以运行cnnQA.py，如果出错可以考虑安装Visual C++ Redistributable for Visual Studio 2015

其他系统下只要能正常在python3.x环境中运行tensorflow且tensorflow版本满足要求即可


使用说明：
在程序的开始部分，通过修改working_type变量的值修改程序工作模式
working_type 的取值 
1  为开始新的training，根据训练集训练神经网络，神经网络模型保存在model下本次运行的目录当中
2  为继续旧的training，读取上一次的model，继续进行训练
3  为developing，读取model，根据开发集测试神经网络效果
4  为testing，读取model，利用测试集测试神经网络，输出result文件为每行中答案对于问题的得分（即认为是正确答案的确率）

通过修改model、modelstamp选择已有的网络模型

跑测试集的输出在/data/result.txt中，跑测试集需要时间特别长，可以跑develop集预览一下