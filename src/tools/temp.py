lr=1.25e-4
# lr_step=[20,40]
lr_step='20,40'
print('*')
for epoch in range(0 + 1, 100+1):
    if epoch in lr_step:
        lr = lr * (0.1 ** (lr_step.index(epoch) + 1))
        print('*'*10)
        print(lr_step.index(epoch))
        print(lr)