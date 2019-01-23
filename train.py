import time
import datetime
from Dataloader import DataLoader
from losses import *
from model import *
from utils import *

# LR = 0.000002
LR = 1e-4
# LRchangLoss = 2000
RESHAPE = (256, 256)
BATCH_SIZE = 2
# shape = (BATCH_SIZE, 1, 32, 32)
shape = (BATCH_SIZE, 1, 1)
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
    print('continue_train: ', continue_train)
    testData = load_images('./images/test', -1)
    testDataset = test_dataset(testData)
    LOSS = 1000
    print(LR)
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
    optimizerG = torch.optim.Adam(g.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    optimizerD = torch.optim.Adam(d.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8)
    # optimizerDOnG = torch.optim.Adam(dOnG.parameters(), lr=LR)
    loss_function = nn.MSELoss().cuda(device='0')
    l1Loss = nn.L1Loss().cuda(device='0')
    perceptauLoss = PerceptualLoss_v2().cuda(device='0')
    perceptauLoss.initialize(loss_function)
    wassesrsteinLoss = WassesrsteinLoss().cuda(device='0')
    now = datetime.datetime.now()
    start = time.time()
    for i in range(8):
        #     i = 1
        GLoss = []
        DLoss = []
        # if i % 2 == 0:
        #     lr = LR * (0.1 ** int(i/2))
        #     print('LR: ', lr)
        #     for para_group in optimizerG.param_groups:
        #         para_group['lr'] = lr
        #     for para_group in optimizerD.param_groups:
        #         para_group['lr'] = lr
        for step, (x, y) in enumerate(dataset):
            x_f = g(x)
            for _ in range(5):
                dReal = d(y)
                dFalse = d(x_f)
                # print(dReal.shape)
                # print(type(outputTrueBatch))
                dLossReal = wassesrsteinLoss(dReal, outputTrueBatch)
                # dLossReal = wassesrsteinLoss(dReal, 1)
                # print(dLossReal)
                dLossFalse = wassesrsteinLoss(dFalse, outputFalseBatch)
                # dLossFalse = wassesrsteinLoss(dFalse, -1)
                dLoss = 0.5 * (dLossReal + dLossFalse)

                optimizerD.zero_grad()
                dLoss.backward(retain_graph=True)
                optimizerD.step()
                DLoss.append(dLoss.data.cpu().numpy())
            #
            #
            dFalse = d(x_f)
            dLoss = wassesrsteinLoss(dFalse, outputTrueBatch)
            pLoss = perceptauLoss(x_f, y)
            gLoss = 100 * pLoss + dLoss
            # gLoss = 100 * perceptauLoss(x_f, y) + wassesrsteinLoss(x_f, y)
            optimizerG.zero_grad()
            gLoss.backward()
            optimizerG.step()
            LOSS_NOW = 100 * pLoss.data.cpu().numpy()
            # print('step', step, ': ', lossG.data.cpu().numpy())
            GLoss.append(LOSS_NOW)
            # print('Dloss: ', np.mean(DLoss), 'dloss: ', dLoss.data.cpu().numpy(), '\tgloss: ', gLoss.data.cpu().numpy(), '\tperceptauLoss: ', pLoss.data.cpu().numpy())
            # print('dloss: ', dLoss.data.cpu().numpy(), '\tgloss: ', LOSS_NOW,
            #       '\tperceptauLoss: ', pLoss.data.cpu().numpy())

            if gLoss.data.cpu().numpy() < LOSS:
                # now = datetime.datetime.now()
                save_dir = os.path.join(BASE_DIR, '{}{}_{}'.format(now.month, now.day, index))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                torch.save(g, os.path.join(save_dir, '{}_Gnet_{}_{}'.format(index, 'bestGloss', int(LOSS_NOW))))
                LOSS = LOSS_NOW
                print('best model saved!!\t', save_dir, '{}_Gnet_{}_{}'.format(index, 'bestGloss', LOSS_NOW))
                save_dir = os.path.join(BASE_DIR, '{}{}_{}'.format(now.month, now.day, index))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                test_img(g, '{}_{}'.format('bestGloss', LOSS_NOW), save_dir, testDataset)



            if step % 10 == 0:
                # now = datetime.datetime.now()
                print('-' * 10, ' ', id, ': Saving model\trun time: ',
                      int((time.time() - start) / 60), 'm', int((time.time() - start) % 60), 's ', '-' * 10)
                start = time.time()
                currentLoss = int(np.mean(GLoss))
                save_dir = os.path.join(BASE_DIR, '{}{}_{}'.format(now.month, now.day, index))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                print('dloss: ', np.mean(DLoss), '\tploss: ', np.mean(GLoss))
                torch.save(g, os.path.join(save_dir, '{}_Gnet_{}_{}'.format(index, id, currentLoss)))
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
    y_test, x_test = data['B'], data['A']
    x_path = data['A_paths']
    # print(y_train.shape)
    dataSet = MyDataset(x_test, y_test)
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
            img_name = x_path[i].split('/')[-1]
            save_image(res[i], './res/img/res{}_{}'.format(index, img_name))
            print('./res/img/res{}_{}'.format(index, img_name))



if __name__ == '__main__':
    # train_multiple_outputs(True, Gmodel='./res/1228/03_Gnet_80_301', Dmodel='./res/1228/03_Dnet_80',
    #                        index='03')
    train_multiple_outputs(index='03_v2')
    # print(os.listdir(os.path.join(os.getcwd(), 'res/116_04')))
    # test('./res/117_04/04_Gnet_bestGloss_252')
    # test('./res/1227/02_Gnet_1_248', '_01')
