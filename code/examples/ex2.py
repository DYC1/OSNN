# %%
from header import *
from jax.numpy import cos, sin, sqrt, arccos, square
from flax.linen import relu
import argparse
# %%
parser = argparse.ArgumentParser(description='Hyperparameters for the model')
parser.add_argument('--MCSizeIn', type=int, default=int(6e4), help='Size of MC input')
parser.add_argument('--MCsizeB', type=int, default=int(5e3)//8, help='Size of MC boundary')
parser.add_argument('--LearningRateStart', type=float, default=1e-3, help='Initial learning rate')
parser.add_argument('--DecayRate', type=float, default=0.5, help='Decay rate for learning rate')
parser.add_argument('--Epoach', type=int, default=int(6e4), help='Number of epochs')
parser.add_argument('--EpoachDecay', type=int, default=int(8000), help='Epochs before decay')
parser.add_argument('--alpha', type=float, default=100, help='Alpha parameter')
parser.add_argument('--mu', type=float, default=2, help='Mu parameter')

args = parser.parse_args()

MCSizeIn = args.MCSizeIn
MCsizeB = args.MCsizeB
LearningRateStart = args.LearningRateStart
DecayRate = args.DecayRate
Epoach = args.Epoach
EpoachDecay = args.EpoachDecay
alpha = args.alpha
mu = args.mu

lam = 0.01
DimInput = 4
NumLayer = 4
Width = 80
Activation = nn.tanh


@jit
def Polar2Cartesian4D(r: Tensor) -> Array:
    x = np.zeros_like(r)
    x = x.at[:, 0].set((r[:, 0]*cos(r[:, 1])))
    x = x.at[:, 1].set((r[:, 0]*sin(r[:, 1])*cos(r[:, 2])))
    x = x.at[:, 2].set((r[:, 0]*sin(r[:, 1])*sin(r[:, 2])*cos(r[:, 3])))
    x = x.at[:, 3].set((r[:, 0]*sin(r[:, 1])*sin(r[:, 2])*sin(r[:, 3])))
    return x


@jit
def Cartesion2Polar4D(x: Tensor) -> Array:
    r = np.zeros_like(x)
    r = r.at[:, 0].set(sqrt(square(x).sum(axis=1)))
    r = r.at[:, 1].set(arccos(x[:, 0]/r[:, 0]))
    r = r.at[:, 2].set(arccos(x[:, 1]/(r[:, 0]*sin(r[:, 1]))))
    r = r.at[:, 3].set(arccos(x[:, 3]/(r[:, 0]*(sin(r[:, 1]))*(sin(r[:, 2])))))
    return r


def Sampler(key) -> Array:
    _x = random.uniform(key, (MCSizeIn, DimInput))*2.0*pi
    _x = _x.at[:, 0].set(sqrt(_x[:, 0]*(1/pi)*4.0+1.0))
    _x = Polar2Cartesian4D(_x)
    return _x


def SamplerBoundary(key) -> Array:
    _x = random.uniform(key, (MCSizeB, DimInput))*2.0*pi
    _x = _x.at[:, 0].set(1.0)
    _x = Polar2Cartesian4D(_x)
    _y = random.uniform(key, (MCSizeB*9, DimInput))*2.0*pi
    _y = _y.at[:, 0].set(3.0)
    _y = Polar2Cartesian4D(_y)
    return _x, _y


def yBoundary(x: Tensor) -> Array:
    return (square(x).sum(axis=1)).reshape(-1, 1)


def pBoundary(x: Tensor) -> Array:
    r = sqrt(square(x).sum(axis=1).reshape(-1))
    return lam*(r-1.0)*(r-3.0)*(x[:, 0]/r).reshape(-1, 1)


def Project(x: Tensor) -> Array:
    return -relu(-(relu(x+0.5)-0.5-0.7))+0.7



def ProjectT(x: Tensor) -> Array:
    return -relu(-(relu(x+0.5)-0.5-0.7))+0.7



def yData(ynn: Function, x: Tensor, para) -> Array:
    rq = np.sum(square(x), axis=1).reshape(-1)

    return (rq+lam*((5.0-(9.0/rq))*(x[:, 0]/sqrt(rq)))).reshape(-1, 1)


def Source(x: Tensor) -> Array:
    r = sqrt(square(x).sum(axis=1).reshape(-1))
    return (-8.0+ProjectT(-(r-1.0)*(r-3.0)*(x[:, 0]/r))).reshape(-1, 1)


def LossPinn(ynn: Function, lapY: Function, pnn: Function, _x: Tensor, para) -> Array:
    return L2Norm(((lapY(_x, para['yNet'])).reshape(-1, 1))+Source(_x)+(-(1.0/lam)*pnn(_x, para['pNet']).reshape(-1, 1)))


def LossP(ynn: Function, pnn: Function, lapP: Function, _x: Tensor, para) -> Array:
    _laplaceP = lapP(_x, para['pNet']).reshape(-1, 1)
    return L2Norm(_laplaceP+ynn(_x, para['yNet']).reshape(-1, 1)-yData(ynn, _x, para))


def LossBoundaryY(fnn: Function, para, key) -> Array:
    _y = np.zeros(1).reshape()
    for _x in SamplerBoundary(key):
        _y = L2Norm(fnn(_x, para)-yBoundary(_x))+_y
    return _y


def LossBoundaryP(fnn: Function, para, key) -> Array:
    re = np.zeros(1).reshape()
    for _x in SamplerBoundary(key):
        re = L2Norm(fnn(_x, para)-pBoundary(_x))+re
    return re


def LossAll(ynn, pnn, lapY, lapP, paras, key):
    _x = Sampler(key)

    return LossPinn(ynn, lapY, pnn, _x, paras)+LossP(ynn, pnn, lapP, _x, paras)+alpha*(LossBoundaryY(ynn, paras['yNet'], key)+LossBoundaryP(pnn, paras['pNet'], key))


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
JFn=jit(lambda _para,key:LossJ(ynn,unn,_para,key))
Pfn=jit(lambda _para,key:LossP(ynn,pnn,LapP,random.uniform(key,(MCSizeIn, DimInput)),_para))
PinnFn=jit(lambda _para,key:LossPinn(ynn,LapY,pnn,random.uniform(key,(MCSizeIn, DimInput)),_para))
# %%
LstLoss=[0.0]*Epoach
LstJ=[0.0]*Epoach
LstP=[0.0]*Epoach
LstPinn=[0.0]*Epoach
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
    key = random.split(key)[0]
    if idx % 1000 == 0:
        checkpointer.save(checkpath/f'{idx}', Paras)

# LapP(x)
checkpointer.save(checkpath/f'final', Paras)
checkpointer.save(checkpath/f'Loss',{'Loss' : np.array(LstLoss)})
checkpointer.save(checkpath/f'J',{'J':np.array(LstJ)})
# %%

dpi = 300
X = np.linspace(-3, 3, dpi)
X = np.stack(np.meshgrid(X, X), axis=-1).reshape(-1, 2)
Mask = np.ones((dpi, dpi), dtype=np.bool_).reshape(-1)
Mask = Mask.at[X[:, 0]**2+X[:, 1]**2 < 1.0].set(False)
Mask = Mask.at[X[:, 0]**2+X[:, 1]**2 > 9.0].set(False)
Mask = ~Mask
X2 = np.zeros((dpi*dpi, 2))
X = np.concatenate([X, X2], axis=1)
Y = ynn(X, Paras['yNet']).reshape(-1)
Y = Y.at[Mask].set(np.nan)
U = ProjectT(unn(X, Paras['pNet']).reshape(-1))
U = U.at[Mask].set(np.nan)

# %%
fig = plt.figure()
plt.imshow(Y.reshape(dpi, dpi), cmap='coolwarm')
plt.colorbar()
plt.axis('off')
plt.tight_layout()
plt.savefig(f'./figs/Y{Timetxt}.svg')
# %%
fig = plt.figure()
plt.imshow(U.reshape(dpi, dpi), cmap='coolwarm', vmin=-0.5, vmax=0.7)
plt.colorbar()
plt.axis('off')
plt.tight_layout()
plt.savefig(f'./figs/U{Timetxt}.svg')

# %%
