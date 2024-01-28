if '__file__' in globals():
  import os, sys
  sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse
import numpy as np
import matplotlib.pyplot as plt
import time
from torch.utils.data import DataLoader
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.autograd as autograd
import statistics
from wgan_gp.models import Generator, Discriminator
from util import save_loss, to_cpu, save_coords, to_cuda

import math

save_dir = "wgan_gp/results"
# ディレクトリがない場合、作成する
if not os.path.exists(save_dir):
    print("ディレクトリを作成します")
    os.makedirs(save_dir)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate") # 1e-4
parser.add_argument("--b1", type=float, default=0, help="adam: decay of first order momentum of gradient") # 0.0
parser.add_argument("--b2", type=float, default=0.9, help="adam: decay of first order momentum of gradient") # 0.9
parser.add_argument("--latent_dim", type=int, default=3, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=2, help="number of classes for dataset")
parser.add_argument("--data_size", type=int, default=200, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--n_critic", type=int, default=5, help="number of training steps for discriminator per iter")
# parser.add_argument("--sample_interval", type=int, default=10000, help="interval betwen image samples")
opt = parser.parse_args()
# print(opt)

data_shape = (opt.channels, opt.data_size)

cuda = True if torch.cuda.is_available() else False
lambda_gp = 10
# Loss weight for gradient penalty
restart = 10000
if restart>0:
    G_PATH = "{}/generator_params_{}".format(save_dir, restart)
    D_PATH = "{}/discriminator_params_{}".format(save_dir, restart)
    generator = Generator(opt.latent_dim)
    generator.load_state_dict(torch.load(G_PATH, map_location=torch.device('cpu')))
    generator.eval()
    discriminator = Discriminator()
    discriminator.load_state_dict(torch.load(D_PATH, map_location=torch.device('cpu')))
    discriminator.eval()
else:
    generator = Generator(opt.latent_dim)
    discriminator = Discriminator()

if cuda:
    print("use GPU")
    generator.cuda()
    discriminator.cuda()
else:
    print("GPU unavailable")
    

# Configure data loader
data = np.load("dataset/data.npy")
param = np.load("dataset/param.npy")

data_mean = data.mean(axis=0)
data_std = data.std(axis=0)
data_std[data_std==0]=1

data = ( data - data_mean ) / data_std
param_m = param.mean()
param_std = param.std()
param = (param - param_m ) / param_std

dataset = torch.utils.data.TensorDataset(torch.tensor(data), torch.tensor(param))
dataloader = torch.utils.data.DataLoader(
  dataset,
  batch_size=opt.batch_size,
  shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

def sample_image(epoch=None, data_num=12):
    # Sample noise
    z = Variable(FloatTensor(np.random.normal(0, 1, (data_num, opt.latent_dim))))
    label_u = 100*np.random.random_sample(size=(data_num, 1))
    label_theta = 90*np.random.random_sample(size=(data_num, 1))
    labels = np.append(label_u.reshape(-1,1), label_theta.reshape(-1,1), axis=1)
    labels=(labels - param_m ) / param_std
    labels = Variable(FloatTensor(labels))
    gen_coords = to_cpu(generator(z, labels)).detach().numpy()
    labels = to_cpu(labels).detach().numpy()
    if epoch is not None:
        #save_coords(gen_coords*data_std+data_mean, labels, "{}/coords/epoch_{0}".format(save_dir, str(epoch).zfill(3)))
        pass
    else:
        np.savez("{}/final".format(save_dir), labels, gen_coords*data_std+data_mean)
        #print(gen_coords*data_std+data_mean)
        #print((gen_coords*data_std+data_mean).shape)
        np.savetxt('{}/final.csv'.format(save_dir), (gen_coords*data_std+data_mean).reshape(data_num*2,-1), delimiter=",")
        np.savetxt('{}/final_label.csv'.format(save_dir), labels*param_std + param_m, delimiter=",")
        #save_coords(gen_coords*data_std+data_mean, labels, "wgan_gp/coords/final.png")

def compute_gradient_penalty(D, real_samples, fake_samples, labels):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = FloatTensor(np.random.random((real_samples.size(0), 1, 1)))
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates, labels)
    fake = Variable(FloatTensor(real_samples.shape[0], 1).fill_(1.0), requires_grad=False)
    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

def throw(u0, theta, dt=0.1):
    g = 9.8
    t = np.arange(0,10, dt)
    x = u0*math.cos(theta)*t
    y = u0*math.sin(theta)*t - g*t*t*0.5
    return np.append(x,y)

def PGAN_Judge( generated, labels):
    labels = labels*param_std + param_m
    u0 = labels[0]
    theta = labels[1]
    TrueData = throw(u0, theta)
    
    #if( np.linalg.norm(TrueData - generated) < threshold):
    #    return 1
    #else:
    #    return 0
    generated = generated*data_std+data_mean
    generated[0:100] = generated[0:100]/ (TrueData[0:100].max() - TrueData[0:100].min())
    generated[100:] = generated[100:]/ (TrueData[100:].max() - TrueData[100:].min())
    TrueData[0:100] = TrueData[0:100]/ (TrueData[0:100].max() - TrueData[0:100].min())
    TrueData[100:] = TrueData[100:]/ (TrueData[100:].max() - TrueData[100:].min())
    
    return np.linalg.norm( (TrueData - generated ) )
    
def PGAN_TrueData( generated, labels):
    labels = labels*param_std + param_m
    u0 = labels[0]
    theta = labels[1]
    TrueData = throw(u0, theta)
   
    return TrueData


# ----------
#  Training
# ----------
flag_pgan = False

start = time.time()
D_losses, G_losses = [], []
Epoch_Pretrain=10000
tt = 10

num_trials=1

for epoch in range(restart, opt.n_epochs):
    #epoch+=restart
    if(epoch<Epoch_Pretrain):
        threshold = tt
    elif(epoch<Epoch_Pretrain+10000):
        threshold = tt
    elif(epoch<Epoch_Pretrain+20000):
        threshold = tt/2.0
    elif(epoch<Epoch_Pretrain+30000):
        threshold = tt/2.0/2.0
    elif(epoch<Epoch_Pretrain+40000):
        threshold = tt/2.0/2.0/2.0
    elif(epoch<Epoch_Pretrain+50000):
        threshold = tt/2.0/2.0/2.0/2.0
    else:
        threshold = tt/2.0/2.0/2.0/2.0/2.0
    
    if(epoch>Epoch_Pretrain):
        flag_pgan = True
    else:
        flag_pgan=False
    for i, (data, labels) in enumerate(dataloader):
        batch_size = data.shape[0]
        data = data.reshape(batch_size, *data_shape)

        valid = Variable(FloatTensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(FloatTensor(batch_size, 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(data.type(FloatTensor))
        train_imgs = Variable(data.type(FloatTensor))
        labels = to_cuda(Variable(torch.reshape(labels.float(), (batch_size, opt.n_classes))))
        # ---------------------
        #  Train Discriminator
        # ---------------------
        optimizer_D.zero_grad()

        # Sample noise and labels as generator input
        z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))

        # Generate a batch of images
        
        gen_imgs = generator(z, labels)
        
        
        
        if flag_pgan:
            #optimizer_D.zero_grad()
            # generate data
            
            # Deactivate Gradient penalty
            # gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data, labels)
            
            #des_data = to_cuda(Variable(FloatTensor( np.zeros(gen_imgs.shape) )))
            
            des_data = real_imgs
            undes_data =  to_cuda(Variable(FloatTensor( np.zeros(gen_imgs.shape) )))
            All_error =  np.empty((0,0))
            
            z_arr =  [0 for x in range(0, num_trials)]
            gen_imgs_arr = [0 for x in range(0, num_trials)]
            
            Collect_des = np.zeros(batch_size)
            Collect_undes = np.zeros(batch_size)
            
            for pp in range(num_trials):
                z_arr[pp] = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
                gen_imgs_arr[pp] = generator(z_arr[pp], labels)
                
                data_g = gen_imgs_arr[pp].detach().cpu().numpy()
                labels_g = labels.detach().cpu().numpy()
                
                error = np.zeros(batch_size)
                TrueData = np.zeros((batch_size,200))
                
                threshold = tt 
                
                for ii in range(batch_size):
                    error[ii] = PGAN_Judge(data_g[ii], labels_g[ii])
                    TrueData[ii] = PGAN_TrueData(data_g[ii], labels_g[ii])
                    if error[ii] < threshold:
                        des_data[ii] = gen_imgs_arr[pp][ii]
                        Collect_des[ii] += 1
                    else:
                        undes_data[ii] = gen_imgs_arr[pp][ii]
                        Collect_undes[ii] += 1
                All_error = np.append(All_error, error)
                #print(pp, len(Collect_des), np.count_nonzero(Collect_des), len(Collect_undes), np.count_nonzero(Collect_undes))
                if (len(Collect_des) == np.count_nonzero(Collect_des)) & (len(Collect_undes) == np.count_nonzero(Collect_undes)):
                    #print("Prepared all Des/Undes data. Tried {} times".format(pp))
                    UseForTraining = (Collect_des>0.9)
                    break
                elif(pp==num_trials-1):
                    #print("Couldn't prepare sufficient dat. Use train data as desirable data.")
                    UseForTraining = (Collect_undes > 0.9)
                    
            if(i==1):
                print(len(Collect_des), np.count_nonzero(Collect_des), len(Collect_undes), np.count_nonzero(Collect_undes))
            des_data = des_data[UseForTraining]
            undes_data = undes_data[UseForTraining]
            #labels = labels[UseForTraining]
            '''
            
            des_data = real_imgs
            undes_data =  to_cuda(Variable(FloatTensor( np.zeros(gen_imgs.shape) )))
            
            error = np.zeros(batch_size)
            for ii in range(batch_size):
                data_g = gen_imgs.detach().cpu().numpy()
                labels_g = labels.detach().cpu().numpy()
                error[ii] = PGAN_Judge(data_g[ii], labels_g[ii])
            des_data = gen_imgs[error<threshold]
            undes_data = gen_imgs[error>threshold]
            '''
            
            #threshold = error.mean()
            #validity_real = validity[ (validity.reshape(-1)>0).detach().numpy() & (error<threshold*1.01)] 
            #validity_fake = validity[ (validity.reshape(-1)<0).detach().numpy() | (error>threshold*1.01)]
            
            
            if(epoch%10==0  and i==1):
                dummy = np.append( des_data.detach().cpu().numpy(), undes_data.detach().cpu().numpy()).reshape(-1,200)
                error_des = np.zeros(batch_size)
                error_undes = np.zeros(batch_size)
                for ii in range(np.count_nonzero(UseForTraining)):
                    error_des[ii] = PGAN_Judge(des_data[ii].detach().cpu().numpy(), labels_g[ii])
                    error_undes[ii] = PGAN_Judge(undes_data[ii].detach().cpu().numpy(), labels_g[ii])
                np.savetxt('{}/gen_data_e{}.csv'.format(save_dir, epoch), ( dummy *data_std+data_mean).reshape(-1,100), delimiter=",")
                np.savetxt('{}/orig_data_e{}.csv'.format(save_dir, epoch), (TrueData).reshape(-1,100), delimiter=",")
                np.savetxt('{}/gen_orig_errorDes_e{}.csv'.format(save_dir, epoch), (error_des).reshape(-1,1), delimiter=",")
                np.savetxt('{}/gen_orig_errorUndes_e{}.csv'.format(save_dir, epoch), (error_undes).reshape(-1,1), delimiter=",")
                np.savetxt('{}/gen_orig_errorAll_e{}.csv'.format(save_dir, epoch), (All_error).reshape(-1,1), delimiter=",")

            #validity_real = torch.cat( (validity_train, validity[ error<threshold ]), 0)
            #validity_fake = validity[ error>threshold ]

            #print(des_data.shape, labels.shape)
            validity_real = discriminator(des_data, labels[UseForTraining])
            validity_fake = discriminator(undes_data, labels[UseForTraining])
            
            #for ii in range(batch_size):
            #    if error[ii]>threshold
            #        suffix_true.append(ii)
            #    else:
            #        suffix_false.append(ii)
            #        
            #validity_real = validity[suffix_true] 
            #validity_fake = validity[suffix_false]
            
            real_imgs = des_data.detach().cpu().numpy()
            #gen_imgs = undes_data.detach().cpu().numpy()
            train_imgs=Variable(data.type(FloatTensor))
            
            
            
            gradient_penalty = compute_gradient_penalty(discriminator, train_imgs.data, gen_imgs.data, labels)
            # Total discriminator loss
            d_loss = -torch.mean(validity_real) + torch.mean(validity_fake) + lambda_gp * gradient_penalty
            
            #if(len(suffix_true)==0):
            #    d_loss = torch.mean(validity_fake)
            #elif(len(suffix_false)==0):
            #    d_loss = -torch.mean(validity_real)
            #else:
            #    d_loss = -torch.mean(validity_real) + torch.mean(validity_fake)
            #d_loss = -torch.mean(validity_real) + torch.mean(validity_fake)
            d_loss.backward()
            optimizer_D.step()
            
            
            #if(epoch%100==0  and i==1):
            #    np.savetxt('{}/real_data_e{}.csv'.format(save_dir, epoch), (real_imgs*data_std+data_mean).reshape(-1,100), delimiter=",")
            #    np.savetxt('{}/real_validity_e{}.csv'.format(save_dir, epoch), validity_real.detach().numpy(), delimiter=",")
            #    np.savetxt('{}/fake_data_e{}.csv'.format(save_dir, epoch), (gen_imgs*data_std+data_mean).reshape( -1,100), delimiter=",")
            #    np.savetxt('{}/fake_validity_e{}.csv'.format(save_dir, epoch), validity_fake.detach().numpy(), delimiter=",")
            
            #with torch.no_grad():
            #    for param in discriminator.parameters():
            #        param.clamp_(-0.01, 0.01)
            
        else:
            # Loss for real images
            validity_real = discriminator(real_imgs, labels)
            # Loss for fake images
            validity_fake = discriminator(gen_imgs, labels)
            
            data_g = gen_imgs.detach().cpu().numpy()
            labels_g = labels.detach().cpu().numpy()
            
            #suffix_true = []
            #suffix_false = []
            error = np.zeros(batch_size)
            TrueData = np.zeros((batch_size,200))
            for ii in range(batch_size):
                error[ii] = PGAN_Judge(data_g[ii], labels_g[ii])
                TrueData[ii] = PGAN_TrueData(data_g[ii], labels_g[ii])
            threshold = tt #np.sort(error)[ int(batch_size/2)]
            #print("thr= ", threshold)
            if(epoch%100==0  and i==1):
                np.savetxt('{}/gen_data_e{}.csv'.format(save_dir, epoch), (data_g*data_std+data_mean).reshape(-1,100), delimiter=",")
                np.savetxt('{}/orig_data_e{}.csv'.format(save_dir, epoch), (TrueData).reshape(-1,100), delimiter=",")
                np.savetxt('{}/gen_orig_error_e{}.csv'.format(save_dir, epoch), (error).reshape(-1,1), delimiter=",")

            # Gradient penalty
            gradient_penalty = compute_gradient_penalty(discriminator, real_imgs.data, gen_imgs.data, labels)
            # Total discriminator loss
            d_loss = -torch.mean(validity_real) + torch.mean(validity_fake) + lambda_gp * gradient_penalty
            
            
            if(epoch%100==0  and i==1):
                np.savetxt('{}/real_data_e{}.csv'.format(save_dir, epoch), (real_imgs.detach().cpu().numpy()*data_std+data_mean).reshape(-1,100), delimiter=",")
                np.savetxt('{}/real_validity_e{}.csv'.format(save_dir, epoch), validity_real.detach().cpu().numpy(), delimiter=",")
                np.savetxt('{}/fake_data_e{}.csv'.format(save_dir, epoch), (gen_imgs.detach().cpu().numpy()*data_std+data_mean).reshape( -1,100), delimiter=",")
                np.savetxt('{}/fake_validity_e{}.csv'.format(save_dir, epoch), validity_fake.detach().cpu().numpy(), delimiter=",")
            
            d_loss.backward()
            optimizer_D.step()
            #with torch.no_grad():
            #    for param in discriminator.parameters():
            #        param.clamp_(-0.01, 0.01)

        optimizer_G.zero_grad()

        # Train the generator every n_critic iterations
        if i % opt.n_critic == 0:
            # -----------------
            #  Train Generator
            # -----------------

            # Generate a batch of images
            
            label_u = 100*np.random.random_sample(size=(batch_size, 1))
            label_theta = 90*np.random.random_sample(size=(batch_size, 1))
            labels = np.append(label_u.reshape(-1,1), label_theta.reshape(-1,1), axis=1)
            labels=(labels - param_m ) / param_std
            
            gen_labels = Variable(FloatTensor(labels))
            gen_imgs = generator(z, gen_labels)
            # Loss measures generator's ability to fool the discriminator
            validity = discriminator(gen_imgs, gen_labels)
            
            
            if flag_pgan:
                data_g = gen_imgs.detach().cpu().numpy()
                labels_g = labels
                error = np.zeros(batch_size)
                for ii in range(batch_size):
                    error[ii] = PGAN_Judge(data_g[ii], labels_g[ii])
                threshold = np.sort(error)[ int(batch_size/2) ]
                #threshold = error.mean()
                #validity_real = validity[ (validity.reshape(-1)>0).detach().numpy() & (error<threshold*1.01)] 
                #validity_fake = validity[ (validity.reshape(-1)<0).detach().numpy() | (error>threshold*1.01)]
                validity_real = validity[ (error<threshold)] 
                validity_fake = validity[ (error>threshold)]
                '''
                suffix_true = []
                suffix_false = []
                for ii in range(batch_size):
                    threshold=1.0
                    disc = PGAN_Judge(data_g[ii], labels_g[ii], threshold)
                    if disc:
                        suffix_true.append(ii)
                    else:
                        suffix_false.append(ii)
                validity_real = validity[suffix_true] 
                validity_fake = validity[suffix_false]
                '''
                #if(len(suffix_true)==0):
                #    g_loss = -torch.mean(validity_fake)
                #elif(len(suffix_false)==0):
                #    g_loss = torch.mean(validity_real)
                #else:
                #    g_loss = torch.mean(validity_real) - torch.mean(validity_fake)
                #g_loss = torch.mean(validity_real) - torch.mean(validity_fake)
                g_loss = - torch.mean(validity_fake)
            else:
                g_loss = -torch.mean(validity)
                
            g_loss.backward()
            optimizer_G.step()
            

            if i==0:
                print(
                    "[Epoch %d/%d %ds] [D loss: %f] [G loss: %f]"
                    % (epoch+1, opt.n_epochs,  int(time.time()-start), d_loss.item(), g_loss.item())
                )
        
                D_losses.append(d_loss.item())
                G_losses.append(g_loss.item())


            #batches_done += opt.n_critic
        if epoch % 500 == 0:
            torch.save(generator.state_dict(), "{}/generator_params_{}".format(save_dir, epoch))
            torch.save(discriminator.state_dict(), "{}/discriminator_params_{}".format(save_dir, epoch))

torch.save(generator.state_dict(), "{}/generator_params_{}".format(save_dir, opt.n_epochs+restart))
torch.save(discriminator.state_dict(), "{}/discriminator_params_{}".format(save_dir, opt.n_epochs+restart))
sample_image(data_num=100)
save_loss(G_losses, D_losses, path="{}/loss.png".format(save_dir))
