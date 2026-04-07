
p=0.9
s=0.1
n=[1,2,4,8,16,100,1000]
for n in n:
    # if n > 0:
        a=(1/((1-p)+ (p/n)))
        print(f'n={n} a={a}')