import svmutil
import svm
tran = [[40, 0.4], [10, 2.0], [50, 0.6], [13, 2.5], [15, 3.0], [46, 0.8]]
label = [1, -1, 1, -1, -1, 1]
testTran = [[35, 1.0], [12, 2.0], [11, 2.0], [30, 1.2]]
testLabel = [1, -1, -1, 1]

classF = [[35, 1.0], [12, 2.0], [11, 2.0], [30, 1.2]]
labelF = [0, 0, 0, 0]
model = svmutil.svm_train(label, tran, "-s 0 -t 2")
p_labels = svmutil.svm_predict(labelF, classF, model)
print(p_labels)
