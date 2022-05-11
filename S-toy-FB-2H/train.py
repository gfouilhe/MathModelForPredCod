from tokenize import Double
from data import Dataset, get_data
from model import MLP
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim as optim
import numpy 

def main():
    model = MLP()

    X_train, X_test, y_train, y_test = get_data(n=2048,std=0.01)
    X_train, X_test, y_train, y_test = torch.from_numpy(X_train.astype(numpy.float32)), torch.from_numpy(X_test.astype(numpy.float32)), \
                                       torch.from_numpy(y_train.astype(numpy.float32)), torch.from_numpy(y_test.astype(numpy.float32))

    dataset_train = Dataset(X_train,y_train)
    dataset_test = Dataset(X_test,y_test)

    batch_size=32

    print('Train dataset size :', X_train.shape[0])
    print('Test dataset size :', X_test.shape[0])


    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=1)

    def train(model, batch_size=8, weight_decay=0.0,
            optimizer="sgd", learning_rate=0.05, momentum=0.9, 
            num_epochs=50,display_graph=True):
        
        # la loss 
        criterion = nn.BCELoss()
        # l'optimiseur
        assert optimizer in ("sgd", "adam")
        if optimizer == "sgd":
            optimizer = optim.SGD(model.parameters(),
                                lr=learning_rate,
                                momentum=momentum,
                                weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(),
                                lr=learning_rate,
                                weight_decay=weight_decay)
        # on track la learning curve avec des listes
        iters, iters_acc, losses, train_acc, val_acc = [], [], [], [], []
        # training
        n = 0 # nombre d'iterations (pour faire des figures)
        for epoch in range(num_epochs):
            for imgs, labels in iter(train_loader):
                # if imgs.size()[0] < batch_size:
                #     continue
                # print(imgs.size())

                model.train() # met le modèle en mode train
                out = model(imgs)

                # print(out)
                # print(labels)
                loss = criterion(out,labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                # sauvegarde iteration et loss
                iters.append(n)
                losses.append(float(loss)/batch_size)             # loss moyen

                if n % 20 == 0 :
                    train_acc.append(get_accuracy(model, train=True)) # training accuracy 
                    val_acc.append(get_accuracy(model, train=False))  # test accuracy
                    iters_acc.append(n)

                n += 1

        p1 = train_acc[-1]
        p2 = val_acc[-1]

        if not display_graph : 
            return [p1,p2]
        # plotting
        plt.title("Courbe d'apprentissage")
        plt.plot(iters, losses, label="Train")
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.show()

        plt.title("Courbe d'apprentissage")
        plt.plot(iters_acc, train_acc, label="Train")
        plt.plot(iters_acc, val_acc, label="Test")
        plt.xlabel("Iterations")
        plt.ylabel("Training Accuracy")
        plt.legend(loc='best')
        plt.show()
        print("Précision finale Train : {}".format(p1))
        print("Précision finale Test : {}".format(p2))

    def get_accuracy(model, train=False):
        if train:
            data = train_loader
        else:
            data = test_loader

        model.eval()
        correct = 0
        total = 0
        for inp, labels in data:
            output = model(inp)
            pred = output.reshape(-1).detach().round()
            #print(pred)
            correct += pred.eq(labels.view_as(pred)).sum().item()
            total += inp.shape[0]
        return correct / total


    train(model,batch_size=batch_size)


if __name__ == "__main__":
    main()

    
