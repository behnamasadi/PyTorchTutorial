# https://www.youtube.com/watch?v=8L11aMN5KY8
# https://theaisummer.com/gan-computer-vision-object-generation/
# https://arxiv.org/abs/1511.00561
# https://towardsdatascience.com/variational-autoencoder-demystified-with-pytorch-implementation-3a06bee395ed
# https://github.com/williamFalcon/pytorch-lightning-vae
# https://stats.stackexchange.com/questions/370179/why-binary-crossentropy-can-be-used-as-the-loss-function-in-autoencoders
# conditional gan
# semi supervided gan


# CycleGAN
# StyleGAN
# pixelRNN
# text-2-image
# DiscoGAN
# lsGAN


# Ref: https://www.youtube.com/watch?v=OljTVUVzPpM&ab_channel=AladdinPersson
import torch.nn as nn
import torchvision
import torch
from  torch.utils.tensorboard import SummaryWriter
import wandb


class Discriminator(nn.Module):
    def __init__(self, image_dim):
        super(Discriminator, self).__init__()
        self.disc=nn.Sequential(
            nn.Linear(in_features=image_dim,out_features=128),
            nn.LeakyReLU(0.1),
            nn.Linear(in_features=128, out_features=1),
            nn.Sigmoid()
        )

    def forward(self,x):
        return self.disc(x)
    
    
class Generator(nn.Module):
    def __init__(self, z_dim, image_dim):
        super(Generator, self).__init__()
        self.gen=nn.Sequential(nn.Linear(z_dim,256),
                               nn.LeakyReLU(0.1),
                               nn.Linear( 256,image_dim),
                               nn.Tanh() )

    def forward(self,x):
        return self.gen(x)

# Hyperparameters
device= torch.device("cuda")  if torch.cuda.is_available() else torch.device("cpu")

print('device is: ',device )
lr=3e-4
z_dim=64
image_dim=28*28*1 # mnist
batch_size=32
num_epochs=50

disc=Discriminator(image_dim).to(device)
gen=Generator(z_dim,image_dim).to(device)

# just to track what happens to a particular input in the network
fixed_noise=torch.randn(batch_size,z_dim).to(device)
transform=torchvision.transforms.Compose([torchvision.transforms.ToTensor(),torchvision.transforms.Normalize((0.1307,), (0.3081,))])


MNIST_train_dataset = torchvision.datasets.MNIST('../data', train=True, download=True,transform=transform)

MNIST_trainloader = torch.utils.data.DataLoader(MNIST_train_dataset,batch_size=batch_size, shuffle=True)
dataiter = iter(MNIST_trainloader)



disc_opt=torch.optim.Adam(lr=lr,params=disc.parameters())

gen_opt=torch.optim.Adam(lr=lr,params=gen.parameters())

criterion=nn.BCELoss()

logdir='../runs/GANS'

import shutil
import os
if os.path.exists(logdir):
    shutil.rmtree(logdir, ignore_errors=True)
    print(logdir+ " cleared")

os.mkdir(logdir)

writer_fake= SummaryWriter(logdir+'/fake')
writer_real= SummaryWriter(logdir+'/real')

step=0

wandb.init(project='GAN', entity='behnamasadi')
wandb.watch(gen, log='all')
wandb.watch(disc, log='all')

for epoch in range(num_epochs):
    for batch_idx, (real,_) in enumerate(MNIST_trainloader):
        real=real.view(-1,image_dim).to(device)


        ### 1) Training Discriminator : max log(D(real)) + log(1- D(G(z))) while weight of G(z) are fixed
        noise=torch.randn(batch_size, z_dim).to(device)
        fake=gen(noise)

        # D(real))
        disc_real=disc(real).view(-1)

        # D(G(z)
        #disc_fake=disc(fake).view(-1)
        # we want to only update the discriminator so so remove the weight of generator from graph in this step
        disc_fake = disc(fake.detach()).view(-1)

        # disc_real values should be close to 1 because we passed real image to discriminator

        lossD_real = criterion(disc_real, torch.ones_like(disc_real))

        # disc_fake values should be close to 0 because we passed fake image to discriminator

        lossD_fake = criterion(disc_fake, torch.zeros_like(disc_fake))

        # total loss
        lossD = (lossD_real + lossD_fake) / 2


        disc.zero_grad()

        #lossD.backward(retain_graph=True)
        lossD.backward(retain_graph=True)

        disc_opt.step()

        ### 2) Training Generator min log(1-D(G(z)))
        # the above will cause saturated gradient which will cause slow gradient update so it is better to max log(D(G(z)))

        output = disc(fake).view(-1)
        lossG=criterion(output,torch.ones_like(output))
        # the above should be like this but because we are trying to max log(D(G(z)))
        # lossG = criterion(output, torch.zeros_like(output))
        gen.zero_grad()
        lossG.backward()
        gen_opt.step()

        if batch_idx == 0:
            print(
                f"Epoch [{epoch}/{num_epochs}] Batch {batch_idx}/{len(MNIST_trainloader)} \
                              Loss D: {lossD:.4f}, loss G: {lossG:.4f}"
            )
            #wandb.log({"loss": lossD})
            wandb.log({"loss": lossG})
            with torch.no_grad():
                fake = gen(fixed_noise).reshape(-1, 1, 28, 28)
                data = real.reshape(-1, 1, 28, 28)
                img_grid_fake = torchvision.utils.make_grid(fake, normalize=True)
                img_grid_real = torchvision.utils.make_grid(data, normalize=True)

                writer_fake.add_image(
                    "Mnist Fake Images", img_grid_fake, global_step=step
                )
                writer_real.add_image(
                    "Mnist Real Images", img_grid_real, global_step=step
                )
                step += 1

wandb.finish()
# %load_ext tensorboard
# %tensorboard --logdir runs


# https://github.com/aladdinpersson/Machine-Learning-Collection



