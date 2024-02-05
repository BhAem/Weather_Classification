
file_resnet = "log/resnet.txt"
x_resnet = []
train_loss_resnet = []
test_loss_resnet = []
acc_resnet = []

f = open(file_resnet, encoding='utf8')
line = f.readline()
while line:
    x_resnet.append(int(line[7:9]))
    train_loss_resnet.append(float(line[24:33]))
    test_loss_resnet.append(float(line[47:55]))
    acc_resnet.append(float(line[63:]))
    line = f.readline()
f.close()

file_resnet_BAM = "log/resnet_BAM.txt"
x_resnet_BAM = []
train_loss_resnet_BAM = []
test_loss_resnet_BAM = []
acc_resnet_BAM = []

f = open(file_resnet_BAM, encoding='utf8')
line = f.readline()
while line:
    x_resnet_BAM.append(int(line[7:9]))
    train_loss_resnet_BAM.append(float(line[24:33]))
    test_loss_resnet_BAM.append(float(line[47:55]))
    acc_resnet_BAM.append(float(line[63:]))
    line = f.readline()
f.close()

file_resnet_CA = "log/resnet_CA.txt"
x_resnet_CA = []
train_loss_resnet_CA = []
test_loss_resnet_CA = []
acc_resnet_CA = []

f = open(file_resnet_CA, encoding='utf8')
line = f.readline()
while line:
    x_resnet_CA.append(int(line[7:9]))
    train_loss_resnet_CA.append(float(line[24:33]))
    test_loss_resnet_CA.append(float(line[47:55]))
    acc_resnet_CA.append(float(line[63:]))
    line = f.readline()
f.close()

file_resnet_CBAM = "log/resnet_CBAM.txt"
x_resnet_CBAM = []
train_loss_resnet_CBAM = []
test_loss_resnet_CBAM = []
acc_resnet_CBAM = []

f = open(file_resnet_CBAM, encoding='utf8')
line = f.readline()
while line:
    x_resnet_CBAM.append(int(line[7:9]))
    train_loss_resnet_CBAM.append(float(line[24:33]))
    test_loss_resnet_CBAM.append(float(line[47:55]))
    acc_resnet_CBAM.append(float(line[63:]))
    line = f.readline()
f.close()

file_resnet_scSE = "log/resnet_scSE.txt"
x_resnet_scSE = []
train_loss_resnet_scSE = []
test_loss_resnet_scSE = []
acc_resnet_scSE = []

f = open(file_resnet_scSE, encoding='utf8')
line = f.readline()
while line:
    x_resnet_scSE.append(int(line[7:9]))
    train_loss_resnet_scSE.append(float(line[24:33]))
    test_loss_resnet_scSE.append(float(line[47:55]))
    acc_resnet_scSE.append(float(line[63:]))
    line = f.readline()
f.close()

file_resnet_simam = "log/resnet_simam.txt"
x_resnet_simam = []
train_loss_resnet_simam = []
test_loss_resnet_simam = []
acc_resnet_simam = []

f = open(file_resnet_simam, encoding='utf8')
line = f.readline()
while line:
    x_resnet_simam.append(int(line[7:9]))
    train_loss_resnet_simam.append(float(line[24:33]))
    test_loss_resnet_simam.append(float(line[47:55]))
    acc_resnet_simam.append(float(line[63:]))
    line = f.readline()
f.close()

file_resnet_TA = "log/resnet_TA.txt"
x_resnet_TA = []
train_loss_resnet_TA = []
test_loss_resnet_TA = []
acc_resnet_TA = []

f = open(file_resnet_TA, encoding='utf8')
line = f.readline()
while line:
    x_resnet_TA.append(int(line[7:9]))
    train_loss_resnet_TA.append(float(line[24:33]))
    test_loss_resnet_TA.append(float(line[47:55]))
    acc_resnet_TA.append(float(line[63:]))
    line = f.readline()
f.close()


import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_resnet, train_loss_resnet, label='Benchmark')
ax.plot(x_resnet, train_loss_resnet_BAM, label='BAM')
ax.plot(x_resnet, train_loss_resnet_CA, label='CA')
ax.plot(x_resnet, train_loss_resnet_CBAM, label='CBAM')
ax.plot(x_resnet, train_loss_resnet_scSE, label='scSE')
ax.plot(x_resnet, train_loss_resnet_simam, label='simam')
ax.plot(x_resnet, train_loss_resnet_TA, label='TA')
leg = ax.legend()
fig.suptitle("Training Loss", fontsize=16, x=0.5, y=0.95)
plt.savefig('./log/training_loss.png',bbox_inches='tight')

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_resnet, test_loss_resnet, label='Benchmark')
ax.plot(x_resnet, test_loss_resnet_BAM, label='BAM')
ax.plot(x_resnet, test_loss_resnet_CA, label='CA')
ax.plot(x_resnet, test_loss_resnet_CBAM, label='CBAM')
ax.plot(x_resnet, test_loss_resnet_scSE, label='scSE')
ax.plot(x_resnet, test_loss_resnet_simam, label='simam')
ax.plot(x_resnet, test_loss_resnet_TA, label='TA')
leg1 = ax.legend()
fig.suptitle("Test Loss", fontsize=16, x=0.5, y=0.95)
plt.savefig('./log/test_loss.png',bbox_inches='tight')

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(x_resnet, acc_resnet, label='Benchmark')
ax.plot(x_resnet, acc_resnet_BAM, label='BAM')
ax.plot(x_resnet, acc_resnet_CA, label='CA')
ax.plot(x_resnet, acc_resnet_CBAM, label='CBAM')
ax.plot(x_resnet, acc_resnet_scSE, label='scSE')
ax.plot(x_resnet, acc_resnet_simam, label='simam')
ax.plot(x_resnet, acc_resnet_TA, label='TA')
leg2 = ax.legend()
fig.suptitle("Accuracy", fontsize=16, x=0.5, y=0.95)
plt.savefig('./log/acc.png',bbox_inches='tight')