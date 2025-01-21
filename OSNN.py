# %%
from NGD import *
import sys
# %%
MCSizeIn = int(6e4)
MCsizeB = int(5e3)//8
LearningRateStart = 1e-3


DecayRate = 0.5
Epoach = int(6e4)
EpoachDecay = int(8000)
alpha = 100
mu = 2
lam = 0.01
DimInput = 4
NumLayer = 4
Width = 80
Activation = nn.tanh


def yData(ynn: Function, x: Tensor, para) -> Array:
    # return (1.0+16*lam*(pi**4))*torch.prod(torch.sin(x),dim=1).reshape(-1,1)
    return (1.0+16*lam*(pi**4))*np.prod(np.sin(x*pi), axis=1).reshape(-1, 1)


def LossPinn(ynn: Function, lapY: Function, pnn: Function, _x: Tensor, para) -> Array:
    # _x = torch.rand(MCsizeB, DimInput, device=Device, dtype=Dtype)
    # _laplaceY = lapY(_x)
    # _laplaceP=Laplace(pnn,_x,DimInput)
    return L2Norm(((lapY(_x, para['yNet'])).reshape(-1, 1))-((1.0/lam)*pnn(_x, para['pNet']).reshape(-1, 1)))


def LossP(ynn: Function, pnn: Function, lapP: Function, _x: Tensor, para) -> Array:
    _laplaceP = lapP(_x, para['pNet']).reshape(-1, 1)
    return L2Norm(_laplaceP+ynn(_x, para['yNet']).reshape(-1, 1)-yData(ynn, _x, para))


def LossBoundary(fnn: Function, para, _key) -> Array:
    _y = np.zeros(1).reshape()
    for idx in range(DimInput*2):
        _x = random.uniform(_key, (MCsizeB, DimInput))
        _x = _x.at[:, idx//2].set(idx % 2)
        _y = L2Norm(fnn(_x, para))+_y
    # _y=(fnn(_x))
    return _y


def LossAll(ynn, pnn, lapY, lapP, paras, _key):
    _x = random.uniform(_key, (MCSizeIn, DimInput))
    # print(LossPinn(ynn,lapY,pnn,_x,paras))
    # print(LossP(ynn,pnn,lapP,_x,paras))

    return LossPinn(ynn, lapY, pnn, _x, paras)+10*LossP(ynn, pnn, lapP, _x, paras)+alpha*(LossBoundary(ynn, paras['yNet'], _key)+10*LossBoundary(pnn, paras['pNet'], _key))


def LossJ(ynn: NN, unn: NN, Paras: Any, key) -> Array:
    _x = random.uniform(key, (MCSizeIn, DimInput))
    return 0.5*(L2Norm(ynn(_x,Paras['yNet']) - yData(ynn, _x,Paras)) + L2Norm(lam*(unn(_x,Paras['pNet']))))


# %%
yNet, yPara = CreateNN(MLP, DimInput, 1, NumLayer, Width, Activation)
pNet, pPara = CreateNN(MLP, DimInput, 1, NumLayer, Width, Activation)
Paras = {'yNet': yPara, 'pNet': pPara}


def ynn(x, para): return yNet.apply(para, x)
def pnn(x, para): return pNet.apply(para, x)
def unn(x, para): return -(1.0/lam)*pnn(x, para)


LapY = CreateLaplaceNN(ynn, DimInput)
LapP = CreateLaplaceNN(pnn, DimInput)
lr_decay_fn = optax.exponential_decay(
    init_value=LearningRateStart,
    transition_steps=EpoachDecay,
    decay_rate=DecayRate
)

optimizer = optax.adam(
    learning_rate=lr_decay_fn)
opt = optimizer.init(Paras)
# %%

# %%
lossFn = jit(lambda _para, key: LossAll(ynn, pnn, LapY, LapP, _para, key))
gradFn = jit(value_and_grad(lossFn, argnums=0))
JFn = jit(lambda _para, key: LossJ(ynn, unn, _para, key))
Pfn= jit(lambda _para,key:LossP(ynn,pnn,LapP,random.uniform(key,(MCSizeIn, DimInput)),_para))
PinnFn= jit(lambda _para,key:LossPinn(ynn,LapY,pnn,random.uniform(key,(MCSizeIn, DimInput)),_para))
# %%
LstLoss=[0.0]*Epoach
LstJ=[0.0]*Epoach
LstP=[0.0]*Epoach
LstPinn=[0.0]*Epoach
#%%
ProcessBar = tqdm(range(Epoach))

for idx in ProcessBar:
    
    value, grads = gradFn(Paras, key)
    updates, opt = optimizer.update(grads, opt)
    ProcessBar.set_postfix(Loss=value)
    Paras = optax.apply_updates(Paras, updates)
    LstLoss[idx]=value
    LstJ[idx]=JFn(Paras,key)
    LstP[idx]=Pfn(Paras,key)
    LstPinn[idx]=PinnFn(Paras,key)
    if idx % 1000 == 0:
        checkpointer.save(checkpath/f'{idx}',Paras)
    key = random.split(key)[0]
# LapP(x)
#%%
checkpointer.save(checkpath/f'final',Paras)
#%%
checkpointer.save(checkpath/f'Loss',{'Loss' : np.array(LstLoss)})
checkpointer.save(checkpath/f'J',{'J':np.array(LstJ)})
checkpointer.save(checkpath/f'P',{'P':np.array(LstP)})
checkpointer.save(checkpath/f'Pinn',{'Pinn':np.array(LstPinn)})
# %%
dpi = 300
X = np.linspace(0, 1, dpi)
X = np.stack(np.meshgrid(X, X), axis=-1).reshape(-1, 2)
X2 = np.zeros((dpi*dpi, 2))+0.5
X = np.concatenate([X, X2], axis=1)
Y = ynn(X, Paras['yNet'])
U = unn(X, Paras['pNet'])
# %%
fig = plt.figure()
plt.imshow(Y.reshape(dpi, dpi), cmap='coolwarm')
plt.colorbar()
plt.axis('off')
plt.tight_layout()
plt.savefig(f'./figs/Y{Timetxt}.pdf')
# %%
fig = plt.figure()
plt.imshow(U.reshape(dpi, dpi), cmap='coolwarm')
plt.colorbar()
plt.axis('off')
plt.tight_layout()
plt.savefig(f'./figs/U{Timetxt}.pdf')
# %%



# %%
