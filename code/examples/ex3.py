# %%
from header import *
from jax.numpy import cos,sin,sqrt,arccos,square,exp
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

def trueY(x:Tensor)->Array:
    return (exp(x[:,0]*(1-x[:,0]))*sin(pi*x[:,1])+exp(x[:,1]*(1-x[:,1]))*sin(pi*x[:,2])+exp(x[:,2]*(1-x[:,3]))*sin(pi*x[:,2])+exp(x[:,3]*(1-x[:,3]))*sin(pi*x[:,0])).reshape(-1,1)

def trueP(x:Tensor)->Array:
    return np.prod(x*(1+cos(pi*x)),axis=1).reshape(-1,1)

_LapP=CreateLaplace(trueP,DimInput)
_LapY=CreateLaplace(trueY,DimInput)
def trueU(x:Tensor)->Array:
    return -trueP(x)/lam
def Fx(x:Tensor)->Array:
    return -_LapY(x).reshape(-1,1)+trueY(x)+trueY(x)**3-trueU(x)
def Yd(x:Tensor)->Array:
    return (-3*(trueY(x)**2)-1)*trueP(x)+trueY(x)+_LapP(x).reshape(-1,1)

def LossY(ynn: NN, lapY: NN, pnn: NN, _x: Tensor,para) -> Array:
    _Y=ynn(_x,para['yNet'])
    # return L2Norm((lapY(_x,['yNet'])).reshape(-1, 1))
    return L2Norm((lapY(_x,para['yNet'])).reshape(-1, 1)-(1.0/lam)*pnn(_x,para['pNet']).reshape(-1, 1)-_Y.reshape(-1, 1)-_Y.reshape(-1, 1)**3+Fx(_x))
def LossP(ynn:NN, pnn:NN,lapP:NN,_x:Tensor,para)->Array:
    _Y=ynn(_x,para['yNet'])
    return L2Norm(lapP(_x,para['pNet']).reshape(-1, 1)+_Y.reshape(-1, 1)-Yd(_x)-(3*square(_Y)+1)*pnn(_x,para['pNet']))



def LossBoundary(fnn:NN,funb:Function,para)->Array:
    _y = np.zeros(1).reshape()
    for idx in range(DimInput*2):
        _x = uniform((MCsizeB, DimInput))
        _x=_x.at[:, idx//2].set(idx%2)
        _y = L2Norm(fnn(_x,para)-funb(_x))+_y
    return _y


def LossAll(ynn, pnn,lapY,lapP,paras):
    _x=uniform((MCSizeIn,DimInput))

    return LossY(ynn,lapY, pnn,_x,paras)+LossP(ynn, pnn,lapP,_x,paras)+alpha*(LossBoundary(ynn,trueY,paras['yNet'])+LossBoundary(pnn,trueP,paras['pNet']))



def LossJ(ynn: NN, unn: NN, Paras: Any, key) -> Array:
    _x = random.uniform(key, (MCSizeIn, DimInput))
    return 0.5*(L2Norm(ynn(_x,Paras['yNet']) - Yd(_x)) + L2Norm(lam*(unn(_x,Paras['pNet']))))

#%%
yNet,yPara=CreateNN(MLP,DimInput,1,NumLayer,Width,Activation)
pNet,pPara=CreateNN(MLP,DimInput,1,NumLayer,Width,Activation)
Paras={'yNet':yPara,'pNet':pPara}

ynn=lambda x,para:yNet.apply(para,x)
pnn=lambda x,para:pNet.apply(para,x)
unn=lambda x,para:-(1.0/lam)*pnn(x,para)
LapY=CreateLaplaceNN(ynn,DimInput)
LapP=CreateLaplaceNN(pnn,DimInput)
lr_decay_fn = optax.exponential_decay(
        init_value=LearningRateStart,
        end_value=LearningRateEnd,
        transition_steps=EpoachDecay,
        decay_rate=DecayRate
)

optimizer = optax.adam(
            learning_rate=lr_decay_fn)
opt=optimizer.init(Paras)
# %%

# %%
lossFn=jit(lambda _para:LossAll(ynn,pnn,LapY,LapP,_para))
gradFn=jit(value_and_grad(lossFn))
JFn=jit(lambda _para,key:LossJ(ynn,unn,_para,key))
YFn=jit(lambda _para,key:LossY(ynn,LapY,pnn,random.uniform(key,(MCSizeIn,DimInput)),_para))
PFn=jit(lambda _para,key:LossP(ynn,pnn,LapP,random.uniform(key,(MCSizeIn,DimInput)),_para))
# %%
LstLoss=[0.0]*Epoach
LstJ=[0.0]*Epoach
LstY=[0.0]*Epoach
LstP=[0.0]*Epoach
ProcessBar=tqdm(range(Epoach))
for idx in ProcessBar:
    value,grads=gradFn(Paras)
    updates,opt=optimizer.update(grads,opt)
    ProcessBar.set_postfix(Loss=value)
    Paras = optax.apply_updates(Paras, updates)
    LstLoss[idx]=value
    LstJ[idx]=JFn(Paras,key)
    LstY[idx]=YFn(Paras,key)
    LstP[idx]=PFn(Paras,key)
    if idx%1000==0:
        checkpointer.save(checkpath/f'{idx}',Paras)

# LapP(x)
checkpointer.save(checkpath/f'final',Paras)
checkpointer.save(checkpath/f'Loss',{'Loss' : np.array(LstLoss)})
checkpointer.save(checkpath/f'J',{'J':np.array(LstJ)})
checkpointer.save(checkpath/f'Y',{'Y':np.array(LstY)})
checkpointer.save(checkpath/f'P',{'P':np.array(LstP)})
# %%
dpi=300
X = np.linspace(0, 1, dpi)
X = np.stack(np.meshgrid(X, X), axis=-1).reshape(-1, 2)
X2 = np.zeros((dpi*dpi, 2))+0.5
X = np.concatenate([X, X2], axis=1)
Y=ynn(X,Paras['yNet'])
U=unn(X,Paras['pNet'])
# %%
fig=plt.figure()
plt.imshow(Y.reshape(dpi,dpi),cmap='coolwarm')
plt.colorbar()
plt.axis('off')
plt.tight_layout()
plt.savefig(f'./figs/Y{Timetxt}.svg')
# %%
fig=plt.figure()
plt.imshow(U.reshape(dpi,dpi),cmap='coolwarm')
plt.colorbar()
plt.axis('off')
plt.tight_layout()
plt.savefig(f'./figs/U{Timetxt}.svg')

# %%
