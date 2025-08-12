import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
import matplotlib.pyplot as plt


from vortex0408 import PINN, PINN_train, mse


model = PINN()

data_ini = jnp.ones(1000)[:,None]

params = model.init(jax.random.key(1), data_ini)

truc = PINN_train(params=params, model=model, ori = 0.001,  n=1, infin  = 7)

params = truc.train(1000, lr = 1e-1)
#params = truc.train(1000, lr = 1e-2)
#params = truc.train(1000, lr=5e-3)
#params = truc.train(1000, lr=1e-3)



paramsa = model.init(jax.random.key(2), data_ini)

truca = PINN_train(params=paramsa, model=model, ori = 0.01,  n=5, infin=7.5)

paramsa = truca.train(1000, lr=1e-1)
#paramsa = truca.train(1000, lr=1e-2)
#paramsa = truca.train(1000, lr=5e-3)
#paramsa = truca.train(1000, lr=1e-3)

paramsb = model.init(jax.random.key(3), data_ini)

trucb = PINN_train(params=paramsb, model=model, ori=0.01 , n=10, infin=9)

paramsb = trucb.train(1000, lr=1e-1)
#paramsb = trucb.train(1000, lr=1e-2)
#paramsb = trucb.train(1000, lr=5e-3)
#paramsb = trucb.train(1000, lr=1e-3)
#paramsb = trucb.train(500, lr=1e-4)

paramsc = model.init(jax.random.key(4), data_ini)

trucc = PINN_train(params=paramsc, model=model, ori = 0.1,  n=20, infin=10)

paramsc = trucc.train(1000, lr=1e-1)
#paramsc = trucc.train(1000, lr=1e-2)
#paramsc = trucc.train(1000, lr=5e-3)
#paramsc = trucc.train(1000, lr=1e-3)
#paramsc = trucc.train(500, lr=1e-4)
#
data_plot = jnp.linspace(0, 10, 500)[:, None]

fig = plt.figure()
ax = fig.add_subplot()



ax.plot(data_plot, jnp.tanh(data_plot) ** (truc.n) * truc.U(data_plot), color='blue', label=f"n = {truc.n}")
ax.plot(data_plot, jnp.tanh(data_plot) ** (truca.n) * truca.U(data_plot), color= 'red', label= f"n = {truca.n}")
ax.plot(data_plot, jnp.tanh(data_plot) ** (trucb.n) * trucb.U(data_plot), color='green', label=f"n = {trucb.n}")
ax.plot(data_plot, jnp.tanh(data_plot) ** (trucc.n) * trucc.U(data_plot), color='orange', label=f"n = {trucc.n}")
ax.legend()
ax.set_xlabel("sinh r")
ax.set_ylabel("U_n")
ax.set_title("Profiles of G-P vortices")
fig.savefig("talkFRGntegrta.pdf", dpi=400)



