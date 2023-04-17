import time
import torch
import sys
sys.path.append('/Users/brianreicher/Documents/GitHub/cyclegan/cyclegan')
from cyclegan.builder import CycleGAN
from cyclegan.opts.arg_parser import *
from cyclegan.data.data import *
from cyclegan.visuals.visualizer import *


if __name__ == '__main__':
    opt = BaseOptions().parse()   # get training options
    dataset: CustomDatasetDataLoader = create_dataset(opt)  # create a dataset given opt.dataset_mode and other options
    dataset_size: int = len(dataset)    # get the number of images in the dataset.
    print('The number of training images = %d' % dataset_size)
    
    # TODO: initialize model
    model: CycleGAN = CycleGAN(gnet_type='resnet',
                          gnet_kwargs={
                                       'input_nc': 1,
                                       'output_nc': 1,
                                       'activation': torch.nn.SELU,
                                       'ngf': 64,
                                       'n_blocks': 9, 
                                       },
                          g_init_learning_rate=0.00004,
                          dnet_type='patch_gan',

                          dnet_kwargs={
                                         'input_nc': 1
                                        },
                          d_init_learning_rate = 0.00007, 
                          loss_kwargs={
                                         'l1_loss': torch.nn.SmoothL1Loss(), 
                                         'g_lambda_dict': {'A': {'l1_loss': {'cycled': 10, 'identity': 0},
                                                            'gan_loss': {'fake': 1, 'cycled': 0},
                                                            },
                                                        'B': {'l1_loss': {'cycled': 10, 'identity': 0},
                                                            'gan_loss': {'fake': 1, 'cycled': 0},
                                                            },
                                                    },
                                         'd_lambda_dict': {'A': {'real': 1, 'fake': 1, 'cycled': 0},
                                                        'B': {'real': 1, 'fake': 1, 'cycled': 0},
                                                    },
                                         'gan_mode': 'lsgan'
                                         }, 
                          adam_betas = [0.5, 0.999], 
                          ndims = 3)

    visualizer: Visualizer = Visualizer(opt)   # create a visualizer that display/save images and plots
    total_iters:int = 0                # the total number of training iterations
    
    for epoch in range(opt.epoch_count, opt.n_epochs + opt.n_epochs_decay + 1):    # outer loop for different epochs; we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>
        epoch_start_time: float = time.time()  # timer for entire epoch
        iter_data_time: float = time.time()    # timer for data loading per iteration
        epoch_iter:int = 0                  # the number of training iterations in current epoch, reset to 0 every epoch
        visualizer.reset()              # reset the visualizer: make sure it saves the results to HTML at least once every epoch
        # model.update_learning_rate()    # update learning rates in the beginning of every epoch.
        for i, data in enumerate(dataset):  # inner loop within one epoch
            iter_start_time: float = time.time()  # timer for computation per iteration
            if total_iters % opt.print_freq == 0:
                t_data: float = iter_start_time - iter_data_time

            total_iters += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)         # unpack data from dataset and apply preprocessing
            model.optimize_parameters()   # calculate loss functions, get gradients, update network weights

            if total_iters % opt.display_freq == 0:   # display images on visdom and save images to a HTML file
                save_result = total_iters % opt.update_html_freq == 0
                # model.compute_visuals()
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_iters % opt.print_freq == 0:    # print training losses and save logging information to the disk
                losses: dict[str, float] = model.get_current_losses()
                t_comp = (time.time() - iter_start_time) / opt.batch_size
                visualizer.print_current_losses(epoch, epoch_iter, losses, t_comp, t_data)
                if opt.display_id > 0:
                    visualizer.plot_current_losses(epoch, float(epoch_iter) / dataset_size, losses)

            if total_iters % opt.save_latest_freq == 0:   # cache our latest model every <save_latest_freq> iterations
                print('saving the latest model (epoch %d, total_iters %d)' % (epoch, total_iters))
                save_suffix: str = 'iter_%d' % total_iters if opt.save_by_iter else 'latest'
                model.save_networks(save_suffix)

            iter_data_time = time.time()
        if epoch % opt.save_epoch_freq == 0:              # cache our model every <save_epoch_freq> epochs
            print('saving the model at the end of epoch %d, iters %d' % (epoch, total_iters))
            model.save_networks('latest')
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' % (epoch, opt.n_epochs + opt.n_epochs_decay, time.time() - epoch_start_time))
