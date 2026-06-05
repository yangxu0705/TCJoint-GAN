import tensorflow as tf
print("tf version", tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(devices[0], True)

import pickle
import numpy as np
import tensorflow as tf
import random
import time
import matplotlib.pyplot as plt

import config as conf
from gan import ThreeDGAN
import utils

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.config.run_functions_eagerly(True)
#tf.config.experimental_run_functions_eagerly(True)

def main():
    model_spec_name = "%s-model-spec.json" % conf.MODEL_NAME              # MODEL_NAME = "3dGAN"
    model_rslt_name = "%s-results.pickle" % conf.MODEL_NAME
                                                                          # MODEL_SAVE_DIR = "./learned-models/3dGAN-tf2"
    model_save_path = os.path.join(conf.MODEL_SAVE_DIR, conf.MODEL_NAME)  # os.path.join()函数：连接两个或更多的路径名组件
    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)
    
    model_ckpt_path = os.path.join(model_save_path, "model-ckpt")
    model_spec_path = os.path.join(model_save_path, model_spec_name)
    model_rslt_path = os.path.join(model_save_path, model_rslt_name)

    hyparams = conf.HYPARAMS[conf.OBJ]

    latent_depth = conf.LATENT_DEPTH   # LATENT_DEPTH = 100
    data = conf.DATA_PATH              # DATA_PATH = './data/train/joint'
    batch_size = conf.BATCH_SIZE       # BATCH_SIZE = 24
    num_epochs = conf.NUM_EPOCHS       # NUM_EPOCHS = 300
    seed = conf.SEED                   # SEED = np.random.normal(0, 0.33, size=[BATCH_SIZE, LATENT_DEPTH]).astype(np.float32)

    files, iter_counter = utils.grab_files(data)
    model = ThreeDGAN(
        project_shape=hyparams["project_shape"],
        gen_filters_list=hyparams["gen_filters_list"],
        disc_filters_list=hyparams["disc_filters_list"],
    )
    generator_opt = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    discriminator_opt = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5)

    @tf.function
    def train_step_1(x, z):
        with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:
            generator_loss = model.generator_loss(z)
            discriminator_loss = model.discriminator_loss(x, z)

            grads_generator_loss = generator_tape.gradient(
                target=generator_loss, sources=model.generator.trainable_variables
            )
            grads_discriminator_loss = discriminator_tape.gradient(
                target=discriminator_loss, sources=model.discriminator.trainable_variables
            )

            generator_opt.apply_gradients(
                zip(grads_generator_loss, model.generator.trainable_variables)
            )
            discriminator_opt.apply_gradients(
                zip(grads_discriminator_loss, model.discriminator.trainable_variables)
            )

        return generator_loss, discriminator_loss

    def train_step_2(x, z):
        with tf.GradientTape() as generator_tape:
            generator_loss = model.generator_loss(z)
            discriminator_loss = model.discriminator_loss(x, z)

            grads_generator_loss = generator_tape.gradient(
                target=generator_loss, sources=model.generator.trainable_variables
            )
            generator_opt.apply_gradients(
                zip(grads_generator_loss, model.generator.trainable_variables)
            )

        return generator_loss, discriminator_loss

    ckpt = tf.train.Checkpoint(generator=model.generator, discriminator=model.discriminator)
    # 如果有需要(在训练前)，恢复最新的检查点
    ckpt.restore(tf.train.latest_checkpoint(model_save_path))

    generator_losses = []
    discriminator_losses = []
    generator_losses_epoch = []
    discriminator_losses_epoch = []
    x_fakes = []
    save_step = 5
    visualization_step = 1

    for i in range(num_epochs):
        #epoch = i // steps_per_epoch
        epoch = i
        random.shuffle(files)
        # 记录训练开始时间
        start = time.time()
        for idx in range(0, int(len(files)/batch_size)):
            #print("Epoch: %i ====> %i / %i" % (epoch+1, (i+1) % steps_per_epoch, steps_per_epoch), end="\r")  # %d十进制整数 %i十进制整数
            file_batch = files[idx*batch_size:(idx+1)*batch_size]
            x_i, start_time = utils.make_inputs(file_batch)
            z_i = np.random.normal(0, 0.33, size=[batch_size, latent_depth]).astype(np.float32)
            '''
            # 对于每一轮，只有在最后一批的准确率不高于80%时，鉴别器才会更新

            if idx == 134:
                accuracy = model.accuracy(x_i, z_i)
                print(accuracy)

                if accuracy <= 0.8:
                    print('accuracy <= 0.8')
                    generator_loss_i, discriminator_loss_i = train_step_1(x_i, z_i)

                else:
                    print('accuracy >= 0.8')
                    generator_loss_i, discriminator_loss_i = train_step_2(x_i, z_i)

            else:
                generator_loss_i, discriminator_loss_i = train_step_1(x_i, z_i)

            generator_losses.append(generator_loss_i)
            discriminator_losses.append(discriminator_loss_i)

            '''
            accuracy = model.accuracy(x_i, z_i)
            if accuracy <= 0.8:
                print('accuracy <= 0.8')
                generator_loss_i, discriminator_loss_i = train_step_1(x_i, z_i)
                generator_losses.append(generator_loss_i)
                discriminator_losses.append(discriminator_loss_i)

            else:
                print('accuracy >= 0.8')
                generator_loss_i, discriminator_loss_i = train_step_2(x_i, z_i)
                generator_losses.append(generator_loss_i)
                discriminator_losses.append(discriminator_loss_i)


        print('generator_losses = ', generator_losses)
        print('\n')
        print('discriminator_losses = ', discriminator_losses)

        generator_loss_epoch = np.mean(generator_losses[-810:])
        discriminator_loss_epoch = np.mean(discriminator_losses[-810:])
        generator_loss_epoch = float(generator_loss_epoch)
        discriminator_loss_epoch = float(discriminator_loss_epoch)

        print("Epoch: %i, Generator Loss: %f, Discriminator Loss: %f" % \
            (epoch+1, generator_loss_epoch, discriminator_loss_epoch)
        )
        
        generator_losses_epoch.append(generator_loss_epoch)
        discriminator_losses_epoch.append(discriminator_loss_epoch)
        
        print('generator_losses_epoch = ', generator_losses_epoch)
        print('\n')
        print('discriminator_losses_epoch = ', discriminator_losses_epoch)

        if (i+1) % save_step == 0:
            x_fake = model.generator(seed, training=False)

            x_fakes.append(x_fake)
                
            ckpt.save(file_prefix=model_ckpt_path)

            with open(model_rslt_path, "ab") as f:  # wb:以二进制格式打开一个文件只用于写入。如果该文件已存在则将其覆盖。如果该文件不存在，创建新文件。
                pickle.dump((generator_losses_epoch, discriminator_losses_epoch, x_fakes), f)  # pickle.dump(obj, file, protocol)
                f.close()
            utils.save_voxels(model_rslt_path, x_fake, epoch)

        # 确保保存图像的目录存在
        output_dir = "training_visualizations"
        os.makedirs(output_dir, exist_ok=True)

        if (i + 1) % visualization_step == 0:
            plt.xlabel("Epochs")
            plt.ylabel("Loss")
            plt.plot(generator_losses_epoch, 'red', label="Generator Loss")
            plt.plot(discriminator_losses_epoch, 'blue', label="Discriminator Loss")
            plt.legend(loc=1)  # 通过参数loc指定图例位置

            # 保存图像
            plt.savefig(os.path.join(output_dir, f"loss_plot_epoch_{i + 1}.png"))  # 保存为PNG格式的图像
            plt.pause(0.1)  # 使绘图非阻塞
            plt.clf()  # 清空图像以便下一次绘制

            # 显示运行总时间
        end = time.time()
        duration = end - start
        print("Train Finished takes:", "{:.2f}".format(duration))


if __name__ == "__main__":
    main()

