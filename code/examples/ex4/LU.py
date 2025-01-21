# %%
from header import *
from jax.numpy import cos, sin, sqrt, arccos, square
from flax.linen import relu
import argparse
# %%
parser = argparse.ArgumentParser(description='Hyperparameters for the model')
parser.add_argument('--MCSizeIn', type=int,
                    default=int(6e4), help='Size of MC input')
parser.add_argument('--MCsizeB', type=int, default=int(5e3) //
                    8, help='Size of MC boundary')
parser.add_argument('--LearningRateStart', type=float,
                    default=1e-3, help='Initial learning rate')
parser.add_argument('--DecayRate', type=float, default=0.5,
                    help='Decay rate for learning rate')
parser.add_argument('--Epoach', type=int, default=int(6e4),
                    help='Number of epochs')
parser.add_argument('--EpoachDecay', type=int,
                    default=int(6000), help='Epochs before decay')
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
DimInput = 6
NumLayer = 4
Width = 80
Activation = nn.tanh


def yData(ynn: Function, x: Tensor, para: Any) -> Array:

    return (6*lam*(pi**4)+1)*sin(pi*x).prod(axis=1).reshape(-1, 1)


def Fx(x: Tensor) -> Array:
    return 5*pi*pi*sin(pi*x).prod(axis=1).reshape(-1, 1)


def LossPinn(ynn: NN, lapY: NN, pnn: NN, _x: Tensor, para: Any) -> Array:

    return L2Norm(((lapY(_x, para['yNet'])).reshape(-1, 1))+Fx(_x)-((1.0/lam)*pnn(_x, para['pNet']).reshape(-1, 1)))


def LossP(ynn: Function, pnn: Function, lapP: Function, _x: Tensor, para) -> Array:
    _laplaceP = lapP(_x, para['pNet']).reshape(-1, 1)
    return L2Norm(_laplaceP+ynn(_x, para['yNet']).reshape(-1, 1)-yData(ynn, _x, para))


def LossBoundary(fnn: Function, para, key) -> Array:
    _y = np.zeros(1).reshape()
    for idx in range(DimInput*2):
        _x = random.uniform(key, (MCsizeB, DimInput))
        _x = _x.at[:, idx//2].set(idx % 2)
        _y = L2Norm(fnn(_x, para))+_y

    return _y


def LossAll(ynn, pnn, lapY, lapP, paras, key):
    _x = random.uniform(key, (MCSizeIn, DimInput))

    return LossPinn(ynn, lapY, pnn, _x, paras)+LossP(ynn, pnn, lapP, _x, paras)+alpha*(LossBoundary(ynn, paras['yNet'], key)+LossBoundary(pnn, paras['pNet'], key))


def LossJ(ynn: NN, unn: NN, Paras: Any, key) -> Array:
    _x = random.uniform(key, (MCSizeIn, DimInput))
    return 0.5*(L2Norm(ynn(_x, Paras['yNet']) - yData(ynn, _x, Paras)) + L2Norm(lam*(unn(_x, Paras['pNet']))))


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
    end_value=LearningRateEnd,
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
PFn = jit(lambda _para, key: LossP(ynn, pnn, LapP,
          random.uniform(key, (MCsizeB, DimInput)), _para))
YFn = jit(lambda _para, key: LossPinn(ynn, LapY, pnn,
          random.uniform(key, (MCsizeB, DimInput)), _para))
# %%
LstLoss = [0.0]*Epoach
LstJ = [0.0]*Epoach
LstY = [0.0]*Epoach
LstP = [0.0]*Epoach
ProcessBar = tqdm(range(Epoach))
for idx in ProcessBar:
    value, grads = gradFn(Paras, key)
    updates, opt = optimizer.update(grads, opt)
    LstLoss[idx] = value
    LstJ[idx] = JFn(Paras, key)
    LstP[idx] = PFn(Paras, key)
    LstY[idx] = YFn(Paras, key)
    key = random.split(key)[0]
    ProcessBar.set_postfix(Loss=value)
    Paras = optax.apply_updates(Paras, updates)
    if idx % 1000 == 0:
        checkpointer.save(checkpath/f'{idx}', Paras)
checkpointer.save(checkpath/f'final', Paras)
checkpointer.save(checkpath/f'Loss', {'Loss': np.array(LstLoss)})
checkpointer.save(checkpath/f'J', {'J': np.array(LstJ)})
checkpointer.save(checkpath/f'P', {'P': np.array(LstP)})
checkpointer.save(checkpath/f'Y', {'Y': np.array(LstY)})
# LapP(x)
# %%
dpi = 300
X = np.linspace(0, 1, dpi)
X = np.stack(np.meshgrid(X, X), axis=-1).reshape(-1, 2)
X2 = np.zeros((dpi*dpi, 4))+0.5
X = np.concatenate([X, X2], axis=1)
Y = ynn(X, Paras['yNet'])
U = unn(X, Paras['pNet'])
# %%
fig = plt.figure()
plt.imshow(Y.reshape(dpi, dpi), cmap='coolwarm')
plt.colorbar()
plt.axis('off')
plt.tight_layout()
plt.savefig(f'./figs/Y{Timetxt}.svg')
# %%
fig = plt.figure()
plt.imshow(U.reshape(dpi, dpi), cmap='coolwarm')
plt.colorbar()
plt.axis('off')
plt.tight_layout()
plt.savefig(f'./figs/U{Timetxt}.svg')
# %%
