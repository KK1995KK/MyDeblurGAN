import time
import datetime
from Dataloader import DataLoader
from losses import *
from model import *
from utils import *

LR = 0.0002
LRchangLoss = 300
RESHAPE = (256, 256)
BATCH_SIZE = 1
shape = (BATCH_SIZE, 1)
outputTrueBatch = torch.from_numpy(np.ones(shape=shape, dtype='double')).cuda()
outputFalseBatch = torch.from_numpy(-np.ones(shape=shape, dtype='double')).cuda()
BASE_DIR = './res'


def train_multiple_outputs(continue_train=False, Gmodel=None, Dmodel=None, index='00', id=0):
    data = load_images('./images/train', n_images=1000)
    y_train, x_train = data['B'], data['A']
    print(y_train.shape[0])
    dataSet = MyDataset(x_train, y_train)
    loader = DataLoader()
    loader.initialize(dataSet, BATCH_SIZE)
    dataset = loader.load_data()
    testData = load_images('./images/test', -1)
    testDataset = test_dataset(testData)
    print('continue_train: ', continue_train)
    if continue_train:
        if Gmodel == None:
            g = Gnet().double()
            print('creat new Gnet')
        else:
            g = torch.load(Gmodel)
            print('using ', Gmodel)
        if DataLoader == None:
            d = NLayerDiscriminator().double()
            print('creat new Dnet')
        else:
            d = torch.load(Dmodel)
            print('using ', Dmodel)
        # dOnG = torch.load('DOnGnet')
    else:
        g = Gnet().double()
        print('creat new Gnet')
        d = NLayerDiscriminator().double()
        print('creat new Dnet')
        # dOnG = generator_containing_discriminator_multiple_outputs(Gnet=g, Dnet=d)
    g.cuda()
    d.cuda()
    # dOnG.cuda()
    optimizerG = torch.optim.Adam(g.parameters(), lr=LR)
    optimizerD = torch.optim.Adam(d.parameters(), lr=LR)
    loss_function = nn.MSELoss().cuda(device='0')
    l1Loss = nn.L1Loss().cuda(device='0')
    perceptauLoss = PerceptualLoss_v2().cuda(device='0')
    perceptauLoss.initialize(l1Loss)
    wassesrsteinLoss = WassesrsteinLoss().cuda(device='0')
    now = datetime.datetime.now()
    start = time.time()
    for i in range(8):
        GLoss = []
        DLoss = []
        for step, (x, y) in enumerate(dataset):
            x_f = g(x)
            for _ in range(2):
                dReal = d(y)
                dFalse = d(x_f)
                dLossReal = wassesrsteinLoss(dReal, outputTrueBatch)
                dLossFalse = wassesrsteinLoss(dFalse, outputFalseBatch)
                dLoss = 0.5 * (dLossReal + dLossFalse)

                optimizerD.zero_grad()
                dLoss.backward(retain_graph=True)
                optimizerD.step()
                DLoss.append(dLoss.data.cpu().numpy())
            #
            #
            dFalse = d(x_f)
            dLoss = loss_function(dFalse, outputTrueBatch)
            # pLoss = perceptauLoss(x_f, y)
            gLoss = dLoss
            optimizerG.zero_grad()
            gLoss.backward()
            optimizerG.step()
            GLoss.append(gLoss.data.cpu().numpy())
            # print('gloss: ', gLoss.data.cpu().numpy())

            if step % 10 == 0:
                # now = datetime.datetime.now()
                print('-' * 10, ' ', id, ': Saving model\trun time: ',
                      int((time.time() - start) / 60), 'm', int((time.time() - start) % 60), 's ', '-' * 10)
                start = time.time()
                save_dir = os.path.join(BASE_DIR, '{}{}_{}'.format(now.month, now.day, index))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                print('dloss: ', np.mean(DLoss), '\tploss: ', np.mean(GLoss))
                torch.save(g, os.path.join(save_dir, '{}_Gnet_{}'.format(index, id)))
                torch.save(d, os.path.join(save_dir, '{}_Dnet_{}'.format(index, id)))
                with open(os.path.join(save_dir, '{}_log.txt'.format(index)), 'a+') as f:
                    f.write('{}\t{}\t{}\r\n'.format(id, np.mean(DLoss), np.mean(GLoss)))
                # id = id + 1
                test_img(g, id, save_dir, testDataset)
                id = id + 1



def test(model, index=''):
    g = torch.load(model)
    batchSize = 2
    # data = load_images('./images/test', n_images=20)
    data = load_images('./images/test', -1)
    y_train, x_train = data['B'], data['A']
    print(data['A_paths'])
    # print(y_train.shape)
    dataSet = MyDataset(x_train, y_train)
    loader = DataLoader()
    loader.initialize(dataSet, batchSize, shuffle=False)
    dataset = loader.load_data()
    for step, (x, y) in enumerate(dataset):
        img = g(x)
        # print(img.shape)
        # print(y.shape)
        res = np.concatenate((img.data.cpu().numpy(), x.data.cpu().numpy(), y.data.cpu().numpy()), axis=3)
        for i in range(x.shape[0]):
            # print(res[i].shape)
            save_image(res[i], './res/img/res{}_{}.png'.format(index, str(step * x.shape[0] + i)))
            print('./res/img/res{}_{}.png'.format(index, str(step * batchSize + i)))


if __name__ == '__main__':
    train_multiple_outputs(True, Gmodel='./res/123_D/D_Gnet_9', Dmodel='./res/123_D/D_Dnet_9',
                           index='D', id=10)
    # train_multiple_outputs(index='D')
    # test('./res/114_MSE/MSE_Gnet_206_8', '_01')
    # test('./res/1227/02_Gnet_1_248', '_01')
