import jax
import jax.numpy as jnp
import flax
from flax import linen as nn
from flax.training.train_state import TrainState
import optax
import matplotlib.pyplot as plt



class PINN(nn.Module):
    def setup(self):
        self.layer1 = nn.Dense(20)
        self.layer2 = nn.Dense(20)
        self.layer3 = nn.Dense(20)
        self.layer4 = nn.Dense(1)

    def __call__(self, y):
        x = nn.tanh(self.layer1(y))
        x = nn.tanh(self.layer2(x))
        x = nn.tanh(self.layer3(x))
        return self.layer4(x)


def _dy(fun):
    return lambda x: jax.grad(lambda x: jnp.sum(fun(x)))(x)

@jax.jit
def mse(x):
    return jnp.mean(x ** 2)


alpha = 0.1

class PINN_train():
    def __init__(self, params, model, n=1, ori = 0.2, infin = 40):
        self.params = params
        self.model = model
        self.n=n
        self.ori = ori
        self.infin = infin
        self.data_x = jnp.linspace(ori, infin, 1000)[:,None]
        self.data_ori = jnp.linspace(0, ori, 500)[:,None]
        self.data_infin = jnp.linspace(infin, 5 * infin, 200)[:,None]

    def U(self, y):
        forward = lambda x: self.model.apply(self.params, x)
        return forward(y)

    def train(self, num_epochs, lr=1e-3):

        @jax.jit
        def loss_iter(params):

            def residue(fct, y):
                U = fct
                U_y = _dy(U)
                U_yy = _dy(_dy(U))


                loss = lambda y : jnp.tanh(y)**(3-self.n)* ( jnp.cosh(y)**(-2)*U_yy(y) + jnp.sinh(y)**(-1)*jnp.cosh(y)**(-3)*U_y(y)
                - self.n**2*jnp.sinh(y)**(-2)*U(y) + U(y)*(1-U(y)**2) )


                return loss(y), jnp.tanh(y)*_dy(loss)(y)

            forward = lambda x : jnp.tanh(x)**(self.n) *self.model.apply(params, x)
            forward_bis = lambda x : self.model.apply(params, x)

            loss_1, loss_2= residue(forward, self.data_x)

            loss = mse(loss_1) + 0.1 * mse(loss_2)
            loss_bc = mse(forward_bis(self.data_ori) - 1. + self.data_ori**2/(4*self.n+4))
            loss_infin = mse(forward(self.data_infin) - 1. + 0.5*(self.n)**2*jnp.sinh(self.data_infin)**(-2))

            return loss_bc + loss_infin + 0.1 * loss

        optimi = optax.adam(lr)
        opt_state = optimi.init(self.params)

        for epoch in range(num_epochs+1):

            loss_tr, grads = jax.value_and_grad(loss_iter)(self.params)
            updates, opt_state = optimi.update(grads, opt_state)
            # Updates the parameters.
            self.params = optax.apply_updates(self.params, updates)
            if epoch % 50 == 0:
                print(f"The loss at epoch {epoch} is {loss_tr}")

        return self.params

if __name__ == "__main__":

    model = PINN()
    data_ini = jnp.ones(1000)[:,None]
    params = model.init(jax.random.key(1), data_ini)
    truc = PINN_train(params=params, model=model, ori = 0.001,  n=1, infin  = 7)
    params = truc.train(1000, lr = 1e-1)
